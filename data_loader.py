"""
data_loader.py -- read PGLib-UC JSON files into clean Python objects.

Pipeline (each step is a separate function so you can see what happens):

    read_json()        -> raw dict exactly as stored in the PGLib-UC file
    reduce_instance()  -> smaller raw dict (first N generators, first T hours,
                          demand scaled to the capacity that is kept)
    parse_instance()   -> `UCInstance`: validated, easy-to-use dataclasses
    load_instance()    -> convenience: the three steps above in one call

FIELD NAMES
-----------
The names below were taken from the real PGLib-UC sample
`2014-09-01_reserves_0.json` (not guessed).  If your file uses different names,
change ONLY the constants in the block below.

Things worth knowing about PGLib-UC files:
  * `thermal_generators` is a dict {name: {...}} (not a list).
  * There is NO separate `fixed_cost` field.  The no-load cost (cost of simply
    being on) is already inside `piecewise_production`: the first point of the
    curve is (Pmin, cost of running at Pmin).
  * `startup` is a list of {lag, cost} steps (hot -> cold).  This project uses
    only the first (hottest) step, i.e. ONE startup cost per startup.
  * Costs in the files look like "thousand dollars" (e.g. 2.2 for 49 MW).  We
    multiply by `cost_multiplier` (default 1000) so objectives are in dollars
    and numerically friendlier for the solver.  Change it if you disagree.
"""
from __future__ import annotations

import copy
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# FIELD NAMES (edit here if your JSON differs)
# --------------------------------------------------------------------------
K_PERIODS = "time_periods"
K_DEMAND = "demand"
K_RESERVES = "reserves"
K_THERMAL = "thermal_generators"
K_RENEWABLE = "renewable_generators"

G_PMIN = "power_output_minimum"
G_PMAX = "power_output_maximum"
G_RAMP_UP = "ramp_up_limit"
G_RAMP_DOWN = "ramp_down_limit"
G_RAMP_SU = "ramp_startup_limit"
G_RAMP_SD = "ramp_shutdown_limit"
G_MIN_UP = "time_up_minimum"
G_MIN_DOWN = "time_down_minimum"
G_P_T0 = "power_output_t0"
G_ON_T0 = "unit_on_t0"
G_UP_T0 = "time_up_t0"
G_DOWN_T0 = "time_down_t0"
G_STARTUP = "startup"            # list of {"lag": int, "cost": float}
G_PWL = "piecewise_production"   # list of {"mw": float, "cost": float}
G_MUST_RUN = "must_run"

DEFAULT_COST_MULTIPLIER = 1000.0


# --------------------------------------------------------------------------
# Clean data classes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Generator:
    """One thermal unit, with all parameters the models need."""

    name: str
    pmin: float                  # minimum output when on (MW)
    pmax: float                  # maximum output when on (MW)
    ramp_up: float               # max increase between two hours, both on (MW/h)
    ramp_down: float             # max decrease between two hours, both on (MW/h)
    startup_ramp: float          # max output in the hour the unit starts (MW)
    shutdown_ramp: float         # max output in the last hour before shut-down (MW)
    min_up: int                  # minimum up time (hours), capped at horizon
    min_down: int                # minimum down time (hours), capped at horizon
    u0: int                      # on (1) / off (0) just before hour 1
    p0: float                    # output just before hour 1 (MW)
    forced_on_hours: int         # hours from t=1 that the unit MUST stay on
    forced_off_hours: int        # hours from t=1 that the unit MUST stay off
    must_run: bool               # must be on in every hour
    startup_cost: float          # cost of one (hot) startup
    cost_points: tuple[tuple[float, float], ...]  # convex curve (MW, cost/h) from Pmin to Pmax


@dataclass(frozen=True)
class UCInstance:
    """A complete single-bus unit commitment instance (hours are 1..n_hours)."""

    name: str
    n_hours: int
    demand: dict[int, float]
    reserve: dict[int, float]
    gens: dict[str, Generator] = field(default_factory=dict)

    @property
    def total_capacity(self) -> float:
        return sum(g.pmax for g in self.gens.values())

    def with_demand_factor(self, factor: float) -> "UCInstance":
        """Copy of the instance with demand (and reserve) multiplied by `factor`."""
        return replace(
            self,
            name=f"{self.name}_dem{factor:g}",
            demand={t: d * factor for t, d in self.demand.items()},
            reserve={t: r * factor for t, r in self.reserve.items()},
        )

    def with_reserve_fraction(self, fraction: float) -> "UCInstance":
        """Copy where the spinning reserve requirement is `fraction` x demand
        (or the file's own requirement if that is larger)."""
        res = {t: max(self.reserve[t], fraction * self.demand[t]) for t in self.demand}
        return replace(self, name=f"{self.name}_res{fraction:g}", reserve=res)


# --------------------------------------------------------------------------
# Step 1: read
# --------------------------------------------------------------------------
def read_json(path: str | Path) -> dict[str, Any]:
    """Read a PGLib-UC JSON file and return the raw dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Step 2: reduce
# --------------------------------------------------------------------------
def reduce_instance(
    raw: dict[str, Any],
    n_generators: int = 20,
    n_hours: int = 24,
    hot_start_only: bool = False,
    max_load_share: float = 0.95,
    ramp_scale: float = 1.0,
) -> dict[str, Any]:
    """Return a smaller raw instance in the SAME JSON format.

    * keeps the first `n_generators` generators (in file order) and the first
      `n_hours` hours;
    * scales demand and reserves by (capacity kept / capacity of full system),
      so the smaller system sees a proportionally smaller load;
    * repairs demand where the scaling alone would make the problem infeasible:
         - demand below the total Pmin of kept must-run units is raised to it,
         - demand (+reserve) above `max_load_share` x kept capacity is lowered.
      Every repair is logged as a warning;
    * optionally (`hot_start_only`) keeps only the hottest startup step - used
      when comparing with EGRET so both tools see exactly one startup cost;
    * optionally (`ramp_scale` < 1) makes ramp limits tighter: ramp-up/down are
      multiplied by `ramp_scale` and the start-up / shut-down limits are reset
      to Pmin + ramp (the convention used inside PGLib-UC).  This is a "ramp
      stress test" and MODIFIES the benchmark - label results accordingly.
      Reason: in the original CA files ramp limits almost never bind, so
      formulations that differ only in how they model ramping look identical.
    """
    gens_all = raw[K_THERMAL]
    names = list(gens_all)[: max(1, n_generators)]
    kept = {n: copy.deepcopy(gens_all[n]) for n in names}
    if hot_start_only:
        for g in kept.values():
            g[G_STARTUP] = sorted(g[G_STARTUP], key=lambda s: s["lag"])[:1]

    if ramp_scale != 1.0:
        for g in kept.values():
            g[G_RAMP_UP] = g[G_RAMP_UP] * ramp_scale
            g[G_RAMP_DOWN] = g[G_RAMP_DOWN] * ramp_scale
            g[G_RAMP_SU] = g[G_PMIN] + g[G_RAMP_UP]
            g[G_RAMP_SD] = g[G_PMIN] + g[G_RAMP_DOWN]

    cap_all = sum(g[G_PMAX] for g in gens_all.values())
    cap_kept = sum(g[G_PMAX] for g in kept.values())
    share = cap_kept / cap_all

    n_total = len(raw[K_DEMAND])
    T = min(n_hours, n_total)
    demand = [d * share for d in raw[K_DEMAND][:T]]
    reserves_raw = raw.get(K_RESERVES) or [0.0] * n_total
    reserves = [r * share for r in reserves_raw[:T]]

    must_pmin = sum(g[G_PMIN] for g in kept.values() if g.get(G_MUST_RUN, 0))
    n_fix = 0
    for t in range(T):
        upper = max_load_share * cap_kept - reserves[t]
        if must_pmin > upper:
            raise ValueError(
                f"Reduced instance cannot be feasible: must-run Pmin ({must_pmin:.1f} MW) "
                f"exceeds usable capacity ({upper:.1f} MW). Try a different N."
            )
        new_d = min(max(demand[t], must_pmin), upper)
        if abs(new_d - demand[t]) > 1e-9:
            n_fix += 1
            demand[t] = new_d
    if n_fix:
        log.warning("reduce_instance: repaired demand in %d of %d hours "
                    "(must-run minimum / capacity limit).", n_fix, T)

    if raw.get(K_RENEWABLE):
        log.warning("Renewable generators are ignored by this project.")

    return {
        K_PERIODS: T,
        K_DEMAND: demand,
        K_RESERVES: reserves,
        K_THERMAL: kept,
        K_RENEWABLE: {},
    }


# --------------------------------------------------------------------------
# Step 3: parse into clean objects
# --------------------------------------------------------------------------
def _check_convex(points: list[tuple[float, float]], name: str) -> None:
    """The epigraph cost model is only exact for CONVEX piecewise-linear curves."""
    slopes = [(c1 - c0) / (m1 - m0) for (m0, c0), (m1, c1) in zip(points, points[1:])]
    for s0, s1 in zip(slopes, slopes[1:]):
        if s1 < s0 - 1e-9 * max(1.0, abs(s0)):
            raise ValueError(f"Generator {name}: production cost curve is not convex.")


def parse_instance(raw: dict[str, Any], name: str = "instance",
                   cost_multiplier: float = DEFAULT_COST_MULTIPLIER) -> UCInstance:
    """Validate a raw PGLib-UC dict and convert it to a `UCInstance`."""
    demand_list = list(raw[K_DEMAND])
    T = len(demand_list)
    if raw.get(K_PERIODS, T) != T:
        raise ValueError("time_periods does not match the length of demand")
    reserve_list = list(raw.get(K_RESERVES) or [0.0] * T)
    if len(reserve_list) != T:
        raise ValueError("reserves must have the same length as demand")

    gens: dict[str, Generator] = {}
    for gname, g in raw[K_THERMAL].items():
        pmin, pmax = float(g[G_PMIN]), float(g[G_PMAX])
        u0 = int(g[G_ON_T0])
        p0 = float(g[G_P_T0]) if u0 else 0.0

        # Effective ramp limits.  Capping at Pmax never changes the feasible set
        # (output can never exceed Pmax anyway) but avoids huge coefficients.
        ramp_up = min(float(g[G_RAMP_UP]), pmax)
        ramp_down = min(float(g[G_RAMP_DOWN]), pmax)
        su = min(max(float(g[G_RAMP_SU]), pmin), pmax)   # a unit must be able to reach Pmin
        sd = min(max(float(g[G_RAMP_SD]), pmin), pmax)

        ut_raw, dt_raw = int(g[G_MIN_UP]), int(g[G_MIN_DOWN])
        up_t0, down_t0 = int(g.get(G_UP_T0, 0)), int(g.get(G_DOWN_T0, 0))
        forced_on = max(0, min(T, ut_raw - up_t0)) if u0 else 0
        forced_off = max(0, min(T, dt_raw - down_t0)) if not u0 else 0

        pts = sorted((float(s["mw"]), float(s["cost"]) * cost_multiplier) for s in g[G_PWL])
        if abs(pts[0][0] - pmin) > 1e-6 or abs(pts[-1][0] - pmax) > 1e-6:
            raise ValueError(f"Generator {gname}: cost curve must span exactly [Pmin, Pmax].")
        _check_convex(pts, gname)

        hot = sorted(g[G_STARTUP], key=lambda s: s["lag"])[0]
        gens[gname] = Generator(
            name=gname, pmin=pmin, pmax=pmax,
            ramp_up=ramp_up, ramp_down=ramp_down, startup_ramp=su, shutdown_ramp=sd,
            min_up=max(1, min(ut_raw, T)), min_down=max(1, min(dt_raw, T)),
            u0=u0, p0=p0, forced_on_hours=forced_on, forced_off_hours=forced_off,
            must_run=bool(g.get(G_MUST_RUN, 0)),
            startup_cost=float(hot["cost"]) * cost_multiplier,
            cost_points=tuple(pts),
        )

    return UCInstance(
        name=name, n_hours=T,
        demand={t + 1: float(d) for t, d in enumerate(demand_list)},
        reserve={t + 1: float(r) for t, r in enumerate(reserve_list)},
        gens=gens,
    )


def load_instance(path: str | Path, n_generators: int = 20, n_hours: int = 24,
                  cost_multiplier: float = DEFAULT_COST_MULTIPLIER,
                  name: str | None = None, ramp_scale: float = 1.0) -> UCInstance:
    """read_json -> reduce_instance -> parse_instance in one call."""
    path = Path(path)
    raw = reduce_instance(read_json(path), n_generators, n_hours, ramp_scale=ramp_scale)
    return parse_instance(raw, name or f"{path.stem}_N{n_generators}_T{n_hours}", cost_multiplier)


# --------------------------------------------------------------------------
# Tiny hand-checkable instance (data lives in code)
# --------------------------------------------------------------------------
def tiny_raw() -> dict[str, Any]:
    """3 generators, 6 hours, made-up numbers (PGLib-UC format).

    * G_base : big, cheap (10-12 $/MWh), slow (min up/down 3 h), on at the start
    * G_mid  : medium, 20 $/MWh, min up/down 2 h, off at the start
    * G_peak : small, expensive (40 $/MWh), no min times, off at the start
    Costs are in dollars (use cost_multiplier=1).
    """
    def gen(pmin, pmax, ru, rd, su, sd, ut, dt, on, p0, up0, down0, su_cost, pts):
        return {
            G_PMIN: pmin, G_PMAX: pmax, G_RAMP_UP: ru, G_RAMP_DOWN: rd,
            G_RAMP_SU: su, G_RAMP_SD: sd, G_MIN_UP: ut, G_MIN_DOWN: dt,
            G_ON_T0: on, G_P_T0: p0, G_UP_T0: up0, G_DOWN_T0: down0,
            G_MUST_RUN: 0,
            G_STARTUP: [{"lag": 1, "cost": su_cost}],
            G_PWL: [{"mw": m, "cost": c} for m, c in pts],
        }

    return {
        K_PERIODS: 6,
        K_DEMAND: [60.0, 80.0, 130.0, 150.0, 100.0, 50.0],
        K_RESERVES: [0.0] * 6,
        K_THERMAL: {
            "G_base": gen(30, 100, 40, 40, 60, 60, 3, 3, 1, 60.0, 10, 0, 500.0,
                          [(30, 450.0), (70, 870.0), (100, 1230.0)]),
            "G_mid": gen(20, 60, 30, 30, 40, 40, 2, 2, 0, 0.0, 0, 5, 200.0,
                         [(20, 460.0), (60, 1260.0)]),
            "G_peak": gen(10, 40, 40, 40, 40, 40, 1, 1, 0, 0.0, 0, 3, 40.0,
                          [(10, 410.0), (40, 1610.0)]),
        },
        K_RENEWABLE: {},
    }


def tiny_instance() -> UCInstance:
    """The tiny instance as a `UCInstance` (costs in dollars)."""
    return parse_instance(tiny_raw(), name="tiny", cost_multiplier=1.0)


# --------------------------------------------------------------------------
# Finding / downloading benchmark files
# --------------------------------------------------------------------------
PGLIB_RAW_BASE = "https://raw.githubusercontent.com/power-grid-lib/pglib-uc/master/"
DEFAULT_DOWNLOADS = [
    "ca/2014-12-01_reserves_0.json",
    "ca/2015-03-01_reserves_0.json",
    "ca/2015-06-01_reserves_0.json",
]


def find_instances(data_dir: str | Path) -> list[Path]:
    """All benchmark JSON files below `data_dir`, sorted by name."""
    return sorted(p for p in Path(data_dir).rglob("*.json") if "_reduced" not in p.name)


def download_pglib_samples(data_dir: str | Path, files: list[str] | None = None,
                           timeout: float = 60.0) -> list[Path]:
    """Best-effort download of extra PGLib-UC files into `data_dir`.

    Never raises: network problems are reported and the files that did arrive
    are returned.  File names/paths are from memory of the pglib-uc repository;
    if a download fails, copy files there manually (see data/README.md).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for rel in files or DEFAULT_DOWNLOADS:
        dest = data_dir / Path(rel).name
        if dest.exists():
            got.append(dest)
            continue
        try:
            with urllib.request.urlopen(PGLIB_RAW_BASE + rel, timeout=timeout) as resp:
                dest.write_bytes(resp.read())
            json.loads(dest.read_text(encoding="utf-8"))   # sanity check
            got.append(dest)
            print(f"  downloaded {dest.name}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if dest.exists():
                dest.unlink()
            print(f"  could not download {rel}: {exc}")
    return got
