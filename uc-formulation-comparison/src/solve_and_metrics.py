"""
solve_and_metrics.py -- solve a model, collect performance metrics, verify the schedule.

Main entry point: `run_formulation(instance, "B", cfg)`, which

    1. builds the Pyomo model,
    2. counts binaries / continuous variables / constraints,
    3. solves the LP relaxation (integrality dropped)  -> LP bound,
    4. solves the real MIP (time limit, gap, 1 thread, fixed seed),
    5. checks the schedule (demand, limits, up/down times, ramps),
    6. returns one result row (a dict) and the schedule (a DataFrame).

Solver differences live ONLY in `solve_model` (HiGHS via Pyomo's APPSI interface
by default; anything else goes through the classic `SolverFactory`).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pyomo.environ as pyo

from .data_loader import UCInstance
from .formulations import FORMULATIONS


# --------------------------------------------------------------------------
# Configuration and result containers
# --------------------------------------------------------------------------
@dataclass
class SolverConfig:
    """Solver settings, identical for every formulation (fair comparison)."""

    name: str = "appsi_highs"     # "appsi_highs" (default, free) | "gurobi" | "cbc" | "highs"
    time_limit: float = 300.0     # seconds per solve
    mip_gap: float = 1e-4         # relative gap: 1e-4 = 0.01 %
    threads: int = 1
    seed: int = 0
    tee: bool = False             # True = print the solver log


@dataclass
class SolveResult:
    """Outcome of one solve (MIP or LP)."""

    objective: float | None       # best feasible objective found (None if none)
    best_bound: float | None      # proven lower bound (None if unknown)
    gap_pct: float | None         # (obj - bound) / |obj| * 100
    time_s: float                 # wall-clock seconds spent in the solver call
    termination: str              # solver's own termination text
    status: str                   # "optimal" | "time_limit" | "infeasible" | "other"
    nodes: int | None             # branch-and-bound nodes (None if not exposed)

    @property
    def has_solution(self) -> bool:
        return self.objective is not None


# --------------------------------------------------------------------------
# Model size and LP relaxation
# --------------------------------------------------------------------------
def count_model(model: pyo.ConcreteModel) -> dict[str, int]:
    """Number of free binary variables, free continuous variables, constraints.

    Variables that were fixed in advance (history, must-run) are not counted.
    """
    n_bin = n_cont = 0
    for v in model.component_data_objects(pyo.Var, active=True):
        if v.fixed:
            continue
        if v.is_binary():
            n_bin += 1
        else:
            n_cont += 1
    n_con = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))
    return {"n_binary": n_bin, "n_continuous": n_cont, "n_constraints": n_con}


def relaxed_copy(model: pyo.ConcreteModel) -> pyo.ConcreteModel:
    """Copy of the model in which binary variables may take any value in [0, 1]."""
    lp = model.clone()
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(lp)
    return lp


# --------------------------------------------------------------------------
# Solving
# --------------------------------------------------------------------------
def _gap_pct(obj: float | None, bound: float | None) -> float | None:
    if obj is None or bound is None or not math.isfinite(obj) or not math.isfinite(bound):
        return None
    return max(0.0, obj - bound) / max(abs(obj), 1e-9) * 100.0


def _status_from_text(text: str) -> str:
    """Map a solver's termination text to one of our four statuses."""
    t = text.lower().replace("_", "").replace(" ", "")
    if "timelimit" in t:
        return "time_limit"
    if "infeasible" in t:
        return "infeasible"
    if "optimal" in t or "convergencecriteriasatisfied" in t:
        return "optimal"
    return "other"


def solve_model(model: pyo.ConcreteModel, cfg: SolverConfig) -> SolveResult:
    """Solve `model` in place (solution is loaded into the variables if one exists)."""
    if cfg.name == "appsi_highs":
        return _solve_appsi_highs(model, cfg)
    return _solve_generic(model, cfg)


def _solve_appsi_highs(model: pyo.ConcreteModel, cfg: SolverConfig) -> SolveResult:
    """HiGHS through Pyomo's APPSI interface (native class, not the legacy wrapper).

    `SolverFactory("appsi_highs")` would return a legacy-style results object
    without best_feasible_objective / best_objective_bound, so we instantiate
    the APPSI class directly.  A fallback handles legacy results anyway.
    """
    from pyomo.contrib.appsi.solvers import Highs

    opt = Highs()
    opt.config.time_limit = cfg.time_limit
    opt.config.mip_gap = cfg.mip_gap
    opt.config.stream_solver = cfg.tee
    opt.config.load_solution = False          # we load by hand: a time-out may have no solution
    opt.highs_options["threads"] = cfg.threads
    opt.highs_options["random_seed"] = cfg.seed

    t0 = time.perf_counter()
    res = opt.solve(model)
    elapsed = time.perf_counter() - t0

    if hasattr(res, "best_feasible_objective"):          # native APPSI results
        obj = res.best_feasible_objective
        bound = res.best_objective_bound
        if obj is not None:
            res.solution_loader.load_vars()
        text = getattr(res.termination_condition, "name", str(res.termination_condition))
    else:                                                 # legacy results (solution auto-loaded)
        text = str(res.solver.termination_condition)
        try:
            obj = float(pyo.value(model.obj))
        except Exception:                                 # noqa: BLE001 - no solution available
            obj = None
        try:
            bound = float(res.problem.lower_bound)
        except Exception:                                 # noqa: BLE001
            bound = None

    nodes = None
    try:                                       # HiGHS statistics; not guaranteed by Pyomo
        info = opt._solver_model.getInfo()
        nodes = int(info.mip_node_count)
        if nodes < 0:
            nodes = None
        if bound is None or not math.isfinite(bound):
            bound = float(info.mip_dual_bound)
    except Exception:                          # noqa: BLE001 - optional information only
        pass

    if obj is not None and not math.isfinite(obj):
        obj = None
    status = _status_from_text(text)
    if status == "optimal" and obj is not None and (bound is None or not math.isfinite(bound)):
        bound = obj
    return SolveResult(obj, bound, _gap_pct(obj, bound), elapsed, text, status, nodes)


def _solver_options(name: str, cfg: SolverConfig) -> dict[str, Any]:
    """Option names differ per solver; map our common settings."""
    n = name.lower()
    if "gurobi" in n:
        return {"TimeLimit": cfg.time_limit, "MIPGap": cfg.mip_gap,
                "Threads": cfg.threads, "Seed": cfg.seed}
    if "cbc" in n:
        return {"seconds": cfg.time_limit, "ratioGap": cfg.mip_gap,
                "threads": cfg.threads, "randomSeed": cfg.seed}
    if "highs" in n:
        return {"time_limit": cfg.time_limit, "mip_rel_gap": cfg.mip_gap,
                "threads": cfg.threads, "random_seed": cfg.seed}
    return {}


def _solve_generic(model: pyo.ConcreteModel, cfg: SolverConfig) -> SolveResult:
    """Classic Pyomo interface for Gurobi / CBC / other solvers (less tested)."""
    opt = pyo.SolverFactory(cfg.name)
    opt.options.update(_solver_options(cfg.name, cfg))
    t0 = time.perf_counter()
    res = opt.solve(model, tee=cfg.tee, load_solutions=False)
    elapsed = time.perf_counter() - t0

    obj = None
    if len(res.solution) > 0:
        model.solutions.load_from(res)
        obj = float(pyo.value(model.obj))
    bound = None
    try:
        lb = float(res.problem.lower_bound)
        bound = lb if math.isfinite(lb) else None
    except Exception:                          # noqa: BLE001
        pass
    text = str(res.solver.termination_condition)
    return SolveResult(obj, bound, _gap_pct(obj, bound), elapsed, text,
                       _status_from_text(text), None)


# --------------------------------------------------------------------------
# Reading the solution
# --------------------------------------------------------------------------
def extract_schedule(inst: UCInstance, model: pyo.ConcreteModel) -> pd.DataFrame:
    """Long table: one row per (generator, hour) with on/off and MW."""
    rows = []
    for g in inst.gens:
        for t in range(1, inst.n_hours + 1):
            rows.append({
                "generator": g, "hour": t,
                "on": int(round(pyo.value(model.u[g, t]))),
                "power_mw": float(pyo.value(model.power[g, t])),
            })
    return pd.DataFrame(rows)


def verify_solution(inst: UCInstance, model: pyo.ConcreteModel,
                    tol: float = 1e-3) -> dict[str, tuple[bool, str]]:
    """Independent check of a solved model; does not use any constraint of the model.

    Returns {check name: (passed, message)}.  Tolerance `tol` is in MW.
    """
    T = inst.n_hours
    on = {g: [int(round(pyo.value(model.u[g, t]))) for t in range(1, T + 1)] for g in inst.gens}
    p = {g: [float(pyo.value(model.power[g, t])) for t in range(1, T + 1)] for g in inst.gens}
    out: dict[str, tuple[bool, str]] = {}

    # 1. demand met in every hour
    worst = max(abs(sum(p[g][t] for g in inst.gens) - inst.demand[t + 1]) for t in range(T))
    out["demand_met"] = (worst <= tol * 10, f"max |supply - demand| = {worst:.2e} MW")

    # 2. output limits (0 when off, Pmin..Pmax when on)
    worst = 0.0
    for g, gen in inst.gens.items():
        for t in range(T):
            lo, hi = (gen.pmin, gen.pmax) if on[g][t] else (0.0, 0.0)
            worst = max(worst, lo - p[g][t], p[g][t] - hi)
    out["output_limits"] = (worst <= tol, f"worst violation = {worst:.2e} MW")

    # 3. minimum up / down times (including history at the start)
    bad = []
    for g, gen in inst.gens.items():
        seq = on[g]
        if gen.must_run and not all(seq):
            bad.append(f"{g}: must-run unit switched off")
        if gen.u0 == 1 and any(x == 0 for x in seq[:gen.forced_on_hours]):
            bad.append(f"{g}: initial minimum up time violated")
        if gen.u0 == 0 and any(x == 1 for x in seq[:gen.forced_off_hours]):
            bad.append(f"{g}: initial minimum down time violated")
        start = 0
        while start < T:                       # walk through runs of equal state
            end = start
            while end + 1 < T and seq[end + 1] == seq[start]:
                end += 1
            length = end - start + 1
            is_last = end == T - 1
            is_first_continuing = start == 0 and seq[0] == gen.u0
            need = gen.min_up if seq[start] == 1 else gen.min_down
            if not is_last and not is_first_continuing and length < need:
                bad.append(f"{g}: {'on' if seq[start] else 'off'}-run of {length} h "
                           f"starting hour {start + 1} (needs {need})")
            start = end + 1
    out["min_up_down_times"] = (not bad, "; ".join(bad[:3]) if bad else "ok")

    # 4. ramp limits (normal ramps, start-up and shut-down limits)
    worst = 0.0
    for g, gen in inst.gens.items():
        prev_on, prev_p = gen.u0, gen.p0
        for t in range(T):
            cur_on, cur_p = on[g][t], p[g][t]
            if prev_on and cur_on:
                worst = max(worst, cur_p - prev_p - gen.ramp_up, prev_p - cur_p - gen.ramp_down)
            elif cur_on and not prev_on:
                worst = max(worst, cur_p - gen.startup_ramp)
            elif prev_on and not cur_on:
                worst = max(worst, prev_p - gen.shutdown_ramp)
            prev_on, prev_p = cur_on, cur_p
    out["ramp_limits"] = (worst <= tol, f"worst violation = {worst:.2e} MW")

    # 5. spinning reserve (only if required)
    if any(r > 1e-9 for r in inst.reserve.values()):
        worst = 0.0
        for t in range(T):
            head = sum(inst.gens[g].pmax * on[g][t] - p[g][t] for g in inst.gens)
            worst = max(worst, inst.reserve[t + 1] - head)
        out["reserve"] = (worst <= tol * 10, f"worst shortfall = {worst:.2e} MW")
    return out


def checks_passed(checks: dict[str, tuple[bool, str]] | None) -> bool | None:
    return None if checks is None else all(ok for ok, _ in checks.values())


def print_checks(checks: dict[str, tuple[bool, str]]) -> None:
    for name, (ok, msg) in checks.items():
        print(f"      [{'PASS' if ok else 'FAIL'}] {name:<18} {msg}")


# --------------------------------------------------------------------------
# One (instance, formulation) experiment
# --------------------------------------------------------------------------
def run_formulation(inst: UCInstance, key: str, cfg: SolverConfig,
                    verbose: bool = True) -> tuple[dict[str, Any], pd.DataFrame | None, pyo.ConcreteModel]:
    """Build, solve (LP and MIP), verify.  Returns (row, schedule, solved model)."""
    model = FORMULATIONS[key](inst)
    size = count_model(model)
    lp_model = relaxed_copy(model)             # copy BEFORE the MIP solve changes values

    lp = solve_model(lp_model, cfg)
    mip = solve_model(model, cfg)

    schedule, checks = None, None
    if mip.has_solution:
        schedule = extract_schedule(inst, model)
        checks = verify_solution(inst, model)
        if verbose:
            print_checks(checks)

    row = {
        "instance": inst.name, "formulation": key,
        "n_gens": len(inst.gens), "n_hours": inst.n_hours,
        **size,
        "lp_obj": lp.objective, "lp_time_s": lp.time_s,
        "mip_obj": mip.objective, "best_bound": mip.best_bound, "mip_gap_pct": mip.gap_pct,
        "solve_time_s": mip.time_s, "status": mip.status, "termination": mip.termination,
        "nodes": mip.nodes, "checks_passed": checks_passed(checks),
    }
    return row, schedule, model


# --------------------------------------------------------------------------
# Aggregation across formulations
# --------------------------------------------------------------------------
def add_lp_gap(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add `best_known_obj` and `lp_gap_pct`.

    best_known_obj = lowest MIP objective found by ANY formulation on the same
    case, so every formulation's LP bound is measured against the same number.
        lp_gap_pct = 100 * (best_known - LP objective) / best_known
    """
    df = df.copy()
    df["best_known_obj"] = df.groupby(group_cols)["mip_obj"].transform("min")
    df["lp_gap_pct"] = 100.0 * (df["best_known_obj"] - df["lp_obj"]) / df["best_known_obj"].abs()
    return df


def check_objective_agreement(df: pd.DataFrame, group_cols: list[str],
                              rel_tol: float) -> pd.DataFrame:
    """For each case, do all formulations report the same optimal cost?

    Two solves that each stop within `mip_gap` of the optimum can differ by up
    to about 2 x gap, hence `rel_tol` should be >= 2 x gap.  Runs that hit the
    time limit only need their [bound, objective] interval to contain the best
    known objective.
    """
    rows = []
    for key, grp in df.groupby(group_cols):
        solved = grp.dropna(subset=["mip_obj"])
        if solved.empty:
            rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                         "agree": None, "rel_spread": None})
            continue
        best = solved["mip_obj"].min()
        optimal = solved[solved["status"] == "optimal"]["mip_obj"]
        spread = (optimal.max() - optimal.min()) / abs(best) if len(optimal) > 1 else 0.0
        unfinished = solved[solved["status"] != "optimal"]
        bounds_ok = all((b is None or pd.isna(b)) or b <= best * (1 + rel_tol)
                        for b in unfinished["best_bound"])
        rows.append({**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
                     "agree": bool(spread <= rel_tol and bounds_ok), "rel_spread": spread})
    return pd.DataFrame(rows)
