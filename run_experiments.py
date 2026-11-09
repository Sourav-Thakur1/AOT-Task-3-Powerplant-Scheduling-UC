"""
run_experiments.py -- ONE command runs everything:

    python run_experiments.py            # full study (can take a while)
    python run_experiments.py --quick    # small & fast smoke run (minutes)

Steps
  0. sanity check on the tiny hand-checkable instance
  1. MAIN grid      : instances x sizes (N generators) x formulations A/B/C
  2. SENSITIVITY    : demand x 0.9 / 1.0 / 1.1 on one instance
  3. RAMP STRESS    : ramp limits tightened (modified benchmark!) on one instance
  4. write results/results.csv, summary_table.csv, objective_agreement.csv
  5. write figures (PNG) into results/

Change the solver with ONE line below (or `--solver gurobi`).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_loader import (download_pglib_samples, find_instances, load_instance,
                             tiny_instance)
from src.formulations import FORMULATIONS
from src.plots import (plot_generation_stack, plot_lp_gap, plot_ramp_stress,
                       plot_solve_time)
from src.solve_and_metrics import (SolverConfig, add_lp_gap, check_objective_agreement,
                                   run_formulation)

# ---------------------------------------------------------------- ONE-LINE SOLVER SWITCH
SOLVER_NAME = "appsi_highs"        # "appsi_highs" (free, default) | "gurobi" | "cbc"
# ----------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
log = logging.getLogger("experiments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--solver", default=SOLVER_NAME)
    p.add_argument("--time-limit", type=float, default=300.0, help="seconds per solve")
    p.add_argument("--gap", type=float, default=1e-4, help="relative MIP gap (1e-4 = 0.01%%)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 20, 40], help="numbers of generators")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--n-instances", type=int, default=3)
    p.add_argument("--formulations", nargs="+", default=["A", "B", "C"], choices=list(FORMULATIONS))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--no-download", action="store_true", help="never try to download extra instances")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--skip-ramp-stress", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="smoke run: sizes 10/20, 12 hours, 1 instance, 20 s limit")
    p.add_argument("--verbose", action="store_true", help="print every verification check")
    args = p.parse_args()
    if args.quick:
        args.sizes, args.hours, args.n_instances, args.time_limit = [10, 20], 12, 1, 20.0
    return args


def fmt(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def solve_case(inst, case_meta: dict, cfg: SolverConfig, forms: list[str], verbose: bool):
    """Run all formulations on one instance; return (rows, {formulation: schedule})."""
    rows, schedules = [], {}
    for key in forms:
        row, sched, _ = run_formulation(inst, key, cfg, verbose=verbose)
        row.update(case_meta)
        rows.append(row)
        schedules[key] = sched
        ok = {True: "checks PASS", False: "checks FAIL", None: "no solution"}[row["checks_passed"]]
        print(f"    {key}: LP={fmt(row['lp_obj'])}  MIP={fmt(row['mip_obj'])}  "
              f"gap={fmt(row['mip_gap_pct'], 4)}%  time={fmt(row['solve_time_s'])}s  "
              f"[{row['status']}]  {ok}")
    return rows, schedules


def tiny_sanity_check(cfg: SolverConfig) -> None:
    """Solve the 3-generator, 6-hour instance with every formulation."""
    print("\n== Step 0: tiny instance (3 generators, 6 hours) ==")
    inst = tiny_instance()
    costs = []
    for key in FORMULATIONS:
        row, _, _ = run_formulation(inst, key, cfg, verbose=False)
        costs.append(row["mip_obj"])
        print(f"    {key}: optimal cost = {fmt(row['mip_obj'])}  (LP bound {fmt(row['lp_obj'])}) "
              f"checks_passed={row['checks_passed']}")
    if None in costs or max(costs) - min(costs) > 1e-3 * abs(min(costs)):
        print("    WARNING: formulations disagree on the tiny instance - do not trust results!")
    else:
        print("    OK: all formulations agree (hand-computed optimum is 8,245).")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = SolverConfig(name=args.solver, time_limit=args.time_limit, mip_gap=args.gap,
                       threads=1, seed=args.seed)
    forms = args.formulations
    print(f"Solver={cfg.name}  time limit={cfg.time_limit:g}s  gap={cfg.mip_gap:g}  "
          f"threads=1  seed={cfg.seed}")

    tiny_sanity_check(cfg)

    # ---------------------------------------------------------------- find instances
    files = find_instances(args.data_dir)
    if len(files) < args.n_instances and not args.no_download:
        print(f"\nOnly {len(files)} instance file(s) in {args.data_dir}; trying to download more ...")
        download_pglib_samples(args.data_dir)
        files = find_instances(args.data_dir)
    files = files[: args.n_instances]
    if not files:
        raise SystemExit(f"No PGLib-UC .json files found in {args.data_dir}. See data/README.md.")
    if len(files) < 3 and not args.quick:
        print(f"WARNING: only {len(files)} instance(s) available (the study asks for 3). "
              f"Add files to {args.data_dir} and re-run.")

    all_rows: list[dict] = []
    stack_schedule, stack_inst = None, None

    # ---------------------------------------------------------------- 1. main grid
    print("\n== Step 1: main grid ==")
    for f in files:
        for n in args.sizes:
            inst = load_instance(f, n, args.hours)
            case_id = f"main|{f.stem}|N{n}"
            print(f"\n[{case_id}]  capacity={inst.total_capacity:,.0f} MW")
            rows, scheds = solve_case(
                inst, {"experiment": "main", "case_id": case_id, "source_file": f.name,
                       "demand_factor": 1.0, "ramp_scale": 1.0}, cfg, forms, args.verbose)
            all_rows += rows
            # keep one schedule for the stack plot: first file, middle size, tight formulation
            want = args.sizes[len(args.sizes) // 2]
            if f == files[0] and n == want:
                pick = "B" if "B" in scheds else forms[0]
                if scheds.get(pick) is not None:
                    stack_schedule, stack_inst = scheds[pick], inst
                    scheds[pick].to_csv(out / f"schedule_{f.stem}_N{n}_{pick}.csv", index=False)

    # ---------------------------------------------------------------- 2. sensitivity
    # Size for sensitivity / ramp stress: 20 generators if available.  (With the
    # very small subsets, e.g. 10 units, tightened ramps can make the problem infeasible.)
    sens_n = 20 if 20 in args.sizes else max(args.sizes)
    if not args.skip_sensitivity:
        print(f"\n== Step 2: demand sensitivity ({files[0].name}, N={sens_n}) ==")
        base = load_instance(files[0], sens_n, args.hours)
        for factor in (0.9, 1.0, 1.1):
            inst = base.with_demand_factor(factor)
            case_id = f"sensitivity|{files[0].stem}|N{sens_n}|dem{factor:g}"
            print(f"\n[{case_id}]")
            rows, _ = solve_case(
                inst, {"experiment": "sensitivity", "case_id": case_id,
                       "source_file": files[0].name, "demand_factor": factor,
                       "ramp_scale": 1.0}, cfg, forms, args.verbose)
            all_rows += rows

    # ---------------------------------------------------------------- 3. ramp stress
    if not args.skip_ramp_stress:
        print(f"\n== Step 3: ramp stress ({files[0].name}, N={sens_n}) - MODIFIED benchmark ==")
        for scale in (1.0, 0.2, 0.1):
            inst = load_instance(files[0], sens_n, args.hours, ramp_scale=scale)
            case_id = f"rampstress|{files[0].stem}|N{sens_n}|ramp{scale:g}"
            print(f"\n[{case_id}]")
            rows, _ = solve_case(
                inst, {"experiment": "ramp_stress", "case_id": case_id,
                       "source_file": files[0].name, "demand_factor": 1.0,
                       "ramp_scale": scale}, cfg, forms, args.verbose)
            all_rows += rows

    # ---------------------------------------------------------------- 4. tables
    df = pd.DataFrame(all_rows)
    df = add_lp_gap(df, ["case_id"])
    first = ["experiment", "case_id", "source_file", "instance", "formulation", "n_gens", "n_hours",
             "demand_factor", "ramp_scale"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(out / "results.csv", index=False)

    agree = check_objective_agreement(df, ["case_id"], rel_tol=2 * cfg.mip_gap + 1e-9)
    agree.to_csv(out / "objective_agreement.csv", index=False)
    n_ok = int((agree["agree"] == True).sum())            # noqa: E712
    n_bad = int((agree["agree"] == False).sum())          # noqa: E712
    n_none = len(agree) - n_ok - n_bad
    print(f"\nObjective agreement across formulations: {n_ok} cases agree, {n_bad} disagree, "
          f"{n_none} infeasible/unsolved for all formulations.")
    if n_bad:
        print("  <-- DISAGREEMENT: see objective_agreement.csv before trusting any result.")

    main_df = df[df["experiment"] == "main"]
    summary = (main_df.groupby(["formulation", "n_gens"])
               .agg(avg_solve_time_s=("solve_time_s", "mean"),
                    avg_lp_gap_pct=("lp_gap_pct", "mean"),
                    avg_final_mip_gap_pct=("mip_gap_pct", "mean"),
                    avg_nodes=("nodes", "mean"),
                    n_cases=("case_id", "nunique"),
                    n_time_limit_hits=("status", lambda s: int((s == "time_limit").sum())))
               .reset_index().round(4))
    summary.to_csv(out / "summary_table.csv", index=False)
    print("\nSummary (main grid):")
    print(summary.to_string(index=False))

    # ---------------------------------------------------------------- 5. figures
    plot_solve_time(summary, out / "fig1_solve_time.png")
    plot_lp_gap(summary, out / "fig2_lp_gap.png")
    if stack_schedule is not None:
        plot_generation_stack(stack_schedule, stack_inst, out / "fig3_generation_stack.png",
                              title=f"Hourly generation - {stack_inst.name} (formulation B)")
    rs = df[df["experiment"] == "ramp_stress"]
    if not rs.empty:
        plot_ramp_stress(rs, out / "fig4_ramp_stress.png")
    print(f"\nDone. Results are in {out}/")


if __name__ == "__main__":
    main()
