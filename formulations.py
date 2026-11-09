"""
formulations.py -- three Pyomo versions of the SAME unit commitment problem.

    A  "baseline / loose"   total-power variable p, simple generation limits,
                            big-M style ramping  (Arroyo & Conejo 2000;
                            Carrion & Arroyo 2006 style).
    B  "tight"              power-ABOVE-minimum variable p', generation limits
                            that include start-up / shut-down terms, and
                            two-period ramping inequalities
                            (Morales-Espana et al. 2013; Gentile et al. 2017;
                            Damci-Kurt et al. 2016; see Knueven et al. 2020).
    C  "A + tight limits"   formulation A plus the tight generation limits
                            of B written in the p variable.  A stepping stone
                            that shows which ingredient of B matters.

All three describe exactly the same set of feasible schedules, so their optimal
costs must match.  They only differ in how "tight" the LP relaxation is.

Equation numbers in comments, e.g. "(20)", refer to Knueven, Ostrowski & Watson
(2020), INFORMS J. Computing 32(4), the paper this project reproduces in part.

Common to A, B and C (see `_build_core` and `_add_system_and_cost`):
  * binaries u (on), v (start-up), w (shut-down) and the logic
        u[t] - u[t-1] = v[t] - w[t]                                    (2)
  * v[t] + w[t] <= 1   (v and w are then exact transition indicators)
  * minimum up / down time, "turn-on / turn-off" inequalities of Malkin (2003)
    and Rajan & Takriti (2005)                                      (4), (5)
  * history: hours a unit is still forced on / off at the start are fixed
  * demand balance (single bus), optional spinning reserve
  * convex piecewise-linear production cost via epigraph constraints
  * ONE (hot) start-up cost per start-up

Model data is read from `UCInstance` objects (see data_loader.py).  Rules refer
to hour t-1 = 0 through the unit's initial state (u0, p0).
"""
from __future__ import annotations

from typing import Callable

import pyomo.environ as pyo

from .data_loader import Generator, UCInstance


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _u_prev(m: pyo.ConcreteModel, g: str, t: int, gen: Generator):
    """u[g, t-1]; for t = 1 this is the known initial on/off state."""
    return m.u[g, t - 1] if t > 1 else gen.u0


# --------------------------------------------------------------------------
# Parts shared by all formulations
# --------------------------------------------------------------------------
def _build_core(inst: UCInstance, name: str) -> pyo.ConcreteModel:
    """Sets, binary variables, state logic and minimum up/down times."""
    m = pyo.ConcreteModel(name=name)
    m.G = pyo.Set(initialize=list(inst.gens), ordered=True)    # generators
    m.T = pyo.RangeSet(1, inst.n_hours)                         # hours 1..T

    m.u = pyo.Var(m.G, m.T, domain=pyo.Binary)   # 1 if unit is on in hour t
    m.v = pyo.Var(m.G, m.T, domain=pyo.Binary)   # 1 if unit STARTS in hour t
    m.w = pyo.Var(m.G, m.T, domain=pyo.Binary)   # 1 if unit SHUTS DOWN in hour t

    # (2) State logic: a change of state is either a start-up or a shut-down.
    def logic_rule(m, g, t):
        return m.u[g, t] - _u_prev(m, g, t, inst.gens[g]) == m.v[g, t] - m.w[g, t]
    m.logic = pyo.Constraint(m.G, m.T, rule=logic_rule)

    # A unit cannot start and stop in the same hour.
    m.no_start_and_stop = pyo.Constraint(
        m.G, m.T, rule=lambda m, g, t: m.v[g, t] + m.w[g, t] <= 1)

    # (4) Minimum up time: if the unit started in the last UT hours it must be on now.
    def min_up_rule(m, g, t):
        ut = inst.gens[g].min_up
        if t < ut:
            return pyo.Constraint.Skip
        return sum(m.v[g, i] for i in range(t - ut + 1, t + 1)) <= m.u[g, t]
    m.min_up = pyo.Constraint(m.G, m.T, rule=min_up_rule)

    # (5) Minimum down time: if the unit shut down in the last DT hours it must be off now.
    def min_down_rule(m, g, t):
        dt = inst.gens[g].min_down
        if t < dt:
            return pyo.Constraint.Skip
        return sum(m.w[g, i] for i in range(t - dt + 1, t + 1)) <= 1 - m.u[g, t]
    m.min_down = pyo.Constraint(m.G, m.T, rule=min_down_rule)

    # History and must-run: fix some on/off decisions in advance.
    for g, gen in inst.gens.items():
        for t in range(1, inst.n_hours + 1):
            if gen.must_run or t <= gen.forced_on_hours:
                m.u[g, t].fix(1)
            elif t <= gen.forced_off_hours:
                m.u[g, t].fix(0)
    return m


def _add_system_and_cost(m: pyo.ConcreteModel, inst: UCInstance) -> None:
    """Demand balance, reserve, production cost and objective.

    Needs `m.power[g, t]` (an Expression = total MW produced), which each
    formulation defines in its own variables.  Because this part is written in
    terms of `m.power` and `m.u`, it is IDENTICAL for A, B and C.
    """
    # Demand balance (single bus, no network): production = demand every hour.
    m.demand_balance = pyo.Constraint(
        m.T, rule=lambda m, t: sum(m.power[g, t] for g in m.G) == inst.demand[t])

    # Optional spinning reserve: unused capacity of the units that are ON must
    # cover the reserve requirement.  (Simplification: ramp-rate limits on the
    # reserve are ignored.)  Skipped in hours where the requirement is zero.
    def reserve_rule(m, t):
        if inst.reserve[t] <= 1e-9:
            return pyo.Constraint.Skip
        return sum(inst.gens[g].pmax * m.u[g, t] - m.power[g, t] for g in m.G) >= inst.reserve[t]
    m.reserve = pyo.Constraint(m.T, rule=reserve_rule)

    # Production cost: cost[g,t] must lie above every line segment of the convex
    # piecewise-linear curve.  Segment through (mw_a, c_a) and (mw_b, c_b) has
    #   slope s = (c_b - c_a)/(mw_b - mw_a),   intercept b = c_a - s*mw_a
    # and the cut  cost >= s*power + b*u  (the "*u" makes the cost 0 when the
    # unit is off).  Minimisation pushes cost onto the highest cut = the curve.
    m.cost = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    seg_data = {g: _segments(gen) for g, gen in inst.gens.items()}
    m.cut_index = pyo.Set(
        initialize=[(g, t, k) for g in inst.gens for t in range(1, inst.n_hours + 1)
                    for k in range(len(seg_data[g]))],
        dimen=3, ordered=True)

    def cut_rule(m, g, t, k):
        slope, icept = seg_data[g][k]
        return m.cost[g, t] >= slope * m.power[g, t] + icept * m.u[g, t]
    m.cost_cuts = pyo.Constraint(m.cut_index, rule=cut_rule)

    # Objective = running cost (incl. no-load cost) + start-up costs.
    m.obj = pyo.Objective(
        expr=sum(m.cost[g, t] for g in m.G for t in m.T)
        + sum(inst.gens[g].startup_cost * m.v[g, t] for g in m.G for t in m.T),
        sense=pyo.minimize)


def _segments(gen: Generator) -> list[tuple[float, float]]:
    """(slope, intercept) of each linear piece of the cost curve."""
    pts = gen.cost_points
    if len(pts) == 1:
        return [(0.0, pts[0][1])]
    out = []
    for (a, ca), (b, cb) in zip(pts, pts[1:]):
        s = (cb - ca) / (b - a)
        out.append((s, ca - s * a))
    return out


def _tight_upper_bounds(m: pyo.ConcreteModel, gen: Generator, g: str, t: int,
                        last_hour: int) -> list:
    """Right-hand sides R such that (power above minimum) <= R.

    Tighter "variable upper bounds" that account for start-up and shut-down
    limits (Morales-Espana et al. 2013, eq. (20); Gentile et al. 2017, eq. (23)):

    * If the unit has minimum up time >= 2, it can never start in hour t AND
      shut down in hour t+1, so one inequality is valid:
          p'[t] <= (Pmax-Pmin) u[t] - (Pmax-SU) v[t] - (Pmax-SD) w[t+1]     (20)
    * If minimum up time = 1, a start in t and a stop in t+1 CAN happen
      together.  Then (20) would be too strong, and one uses the two weaker
      but still tight inequalities of Gentile et al.                         (23)
          p'[t] <= (Pmax-Pmin) u[t] - (Pmax-SU) v[t] - max(SU-SD,0) w[t+1]
          p'[t] <= (Pmax-Pmin) u[t] - (Pmax-SD) w[t+1] - max(SD-SU,0) v[t]

    w[T+1] lies outside the horizon and is treated as 0.
    """
    span = gen.pmax - gen.pmin
    w_next = m.w[g, t + 1] if t < last_hour else 0
    su, sd, pmax = gen.startup_ramp, gen.shutdown_ramp, gen.pmax
    if gen.min_up >= 2:
        return [span * m.u[g, t] - (pmax - su) * m.v[g, t] - (pmax - sd) * w_next]
    return [
        span * m.u[g, t] - (pmax - su) * m.v[g, t] - max(su - sd, 0.0) * w_next,
        span * m.u[g, t] - (pmax - sd) * w_next - max(sd - su, 0.0) * m.v[g, t],
    ]


# --------------------------------------------------------------------------
# Formulation A: baseline (loose)
# --------------------------------------------------------------------------
def build_formulation_a(inst: UCInstance, _name: str = "A_baseline") -> pyo.ConcreteModel:
    """Baseline: total power p, simple limits, big-M ramping."""
    m = _build_core(inst, _name)
    m.p = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)        # MW produced
    m.power = pyo.Expression(m.G, m.T, rule=lambda m, g, t: m.p[g, t])
    _add_system_and_cost(m, inst)
    _add_loose_generation_limits(m, inst)
    _add_loose_ramping(m, inst)
    return m


def _add_loose_generation_limits(m: pyo.ConcreteModel, inst: UCInstance) -> None:
    """(18) Pmin*u <= p <= Pmax*u : output only between Pmin and Pmax if on."""
    m.gen_min = pyo.Constraint(
        m.G, m.T, rule=lambda m, g, t: m.p[g, t] >= inst.gens[g].pmin * m.u[g, t])
    m.gen_max = pyo.Constraint(
        m.G, m.T, rule=lambda m, g, t: m.p[g, t] <= inst.gens[g].pmax * m.u[g, t])


def _add_loose_ramping(m: pyo.ConcreteModel, inst: UCInstance) -> None:
    """(24), (25) Arroyo & Conejo (2000) ramping with start-up/shut-down limits.

    Ramp up  : p[t] - p[t-1] <= RampUp * u[t-1] + StartUpLimit * v[t]
    Ramp down: p[t-1] - p[t] <= RampDown * u[t] + ShutDownLimit * w[t]
    The terms with v and w act like big-M switches that lift the ramp limit in
    the hour a unit starts or stops.
    """
    def ramp_up(m, g, t):
        gen = inst.gens[g]
        p_prev = m.p[g, t - 1] if t > 1 else gen.p0
        return (m.p[g, t] - p_prev
                <= gen.ramp_up * _u_prev(m, g, t, gen) + gen.startup_ramp * m.v[g, t])

    def ramp_down(m, g, t):
        gen = inst.gens[g]
        p_prev = m.p[g, t - 1] if t > 1 else gen.p0
        return (p_prev - m.p[g, t]
                <= gen.ramp_down * m.u[g, t] + gen.shutdown_ramp * m.w[g, t])

    m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up)
    m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down)


# --------------------------------------------------------------------------
# Formulation B: tight
# --------------------------------------------------------------------------
def build_formulation_b(inst: UCInstance) -> pyo.ConcreteModel:
    """Tight: power above minimum p', tight limits (20)/(23), ramping (35)/(36)."""
    m = _build_core(inst, "B_tight")
    # p'[g,t] = MW produced ABOVE the minimum Pmin (0 when the unit is off).
    m.pp = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    # total power  p = p' + Pmin*u                                          (12)
    m.power = pyo.Expression(
        m.G, m.T, rule=lambda m, g, t: m.pp[g, t] + inst.gens[g].pmin * m.u[g, t])
    _add_system_and_cost(m, inst)

    # Tight generation limits (also imply 0 <= p' <= (Pmax-Pmin) u): (20) / (23)
    m.tight_limits = pyo.ConstraintList()
    for g, gen in inst.gens.items():
        for t in range(1, inst.n_hours + 1):
            for rhs in _tight_upper_bounds(m, gen, g, t, inst.n_hours):
                m.tight_limits.add(m.pp[g, t] <= rhs)

    # Two-period ramping inequalities of Damci-Kurt et al. (2016), in p' form:
    #   (35) p'[t]   - p'[t-1] <= (SU - Pmin - RU) v[t] + RU u[t]
    #   (36) p'[t-1] - p'[t]   <= (SD - Pmin - RD) w[t] + RD u[t-1]
    # They are facets of the two-period ramp polytopes and are as sparse as the
    # simple big-M ramps of formulation A.
    def pp_prev(m, g, t):
        gen = inst.gens[g]
        return m.pp[g, t - 1] if t > 1 else max(gen.p0 - gen.pmin * gen.u0, 0.0)

    def ramp_up(m, g, t):
        gen = inst.gens[g]
        return (m.pp[g, t] - pp_prev(m, g, t)
                <= (gen.startup_ramp - gen.pmin - gen.ramp_up) * m.v[g, t]
                + gen.ramp_up * m.u[g, t])

    def ramp_down(m, g, t):
        gen = inst.gens[g]
        return (pp_prev(m, g, t) - m.pp[g, t]
                <= (gen.shutdown_ramp - gen.pmin - gen.ramp_down) * m.w[g, t]
                + gen.ramp_down * _u_prev(m, g, t, gen))

    m.ramp_up = pyo.Constraint(m.G, m.T, rule=ramp_up)
    m.ramp_down = pyo.Constraint(m.G, m.T, rule=ramp_down)
    return m


# --------------------------------------------------------------------------
# Formulation C: A + tight generation limits
# --------------------------------------------------------------------------
def build_formulation_c(inst: UCInstance) -> pyo.ConcreteModel:
    """A plus the tight generation limits of B (written for the total power p).

    Every added inequality is valid (see `_tight_upper_bounds`), so the set of
    feasible schedules is unchanged; only the LP relaxation gets tighter.
    """
    m = build_formulation_a(inst, _name="C_A_plus_tight_limits")
    m.tight_limits = pyo.ConstraintList()
    for g, gen in inst.gens.items():
        for t in range(1, inst.n_hours + 1):
            for rhs in _tight_upper_bounds(m, gen, g, t, inst.n_hours):
                # p = p' + Pmin*u  ->  p <= Pmin*u + (bound on p')
                m.tight_limits.add(m.p[g, t] <= gen.pmin * m.u[g, t] + rhs)
    return m


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
FORMULATIONS: dict[str, Callable[[UCInstance], pyo.ConcreteModel]] = {
    "A": build_formulation_a,
    "B": build_formulation_b,
    "C": build_formulation_c,
}

FORMULATION_LABELS: dict[str, str] = {
    "A": "A: baseline (loose)",
    "B": "B: tight",
    "C": "C: A + tight limits",
}
