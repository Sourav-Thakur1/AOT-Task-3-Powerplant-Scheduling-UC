"""
Tests on the tiny hand-checkable instance (3 generators, 6 hours).

Hand calculation of the optimum (cost in $):
    G_base stays on all day (min up/down 3 h), G_mid is switched on in hours 3-5,
    G_peak is never used.
        hour 1: G_base 60 MW                          765
        hour 2: G_base 80 MW                          990
        hour 3: G_base 100 + G_mid 30 (start-up 200)  1230 + 660
        hour 4: G_base 100 + G_mid 50                 1230 + 1060
        hour 5: G_base  80 + G_mid 20                  990 +  460
        hour 6: G_base  50                             660
        sum of hourly costs 8,045 + one start-up 200 = 8,245
    Why does G_mid stay on in hour 5 although G_base could cover 100 MW?
    Because it produces 50 MW in hour 4 and its shut-down limit is 40 MW.
    (Optimum also confirmed by brute-force enumeration of all schedules.)

Run with:  pytest -q
"""
import pytest

pytest.importorskip("pyomo.environ")
pytest.importorskip("highspy")

from src.data_loader import (load_instance, parse_instance, reduce_instance, read_json,  # noqa: E402
                             tiny_instance, tiny_raw, find_instances)
from src.formulations import FORMULATIONS  # noqa: E402
from src.solve_and_metrics import SolverConfig, run_formulation, verify_solution  # noqa: E402

EXPECTED_COST = 8245.0
CFG = SolverConfig(time_limit=60, mip_gap=1e-6)


@pytest.fixture(scope="module")
def results():
    inst = tiny_instance()
    return {k: run_formulation(inst, k, CFG, verbose=False) for k in FORMULATIONS}


def test_parse_tiny():
    inst = tiny_instance()
    assert inst.n_hours == 6 and len(inst.gens) == 3
    assert inst.gens["G_base"].u0 == 1 and inst.gens["G_mid"].u0 == 0


@pytest.mark.parametrize("key", list(FORMULATIONS))
def test_optimal_cost_matches_hand_calculation(results, key):
    row, _, _ = results[key]
    assert row["status"] == "optimal"
    assert row["mip_obj"] == pytest.approx(EXPECTED_COST, rel=1e-4)


@pytest.mark.parametrize("key", list(FORMULATIONS))
def test_schedule_passes_independent_checks(results, key):
    row, schedule, _ = results[key]
    assert row["checks_passed"] is True
    assert schedule is not None


def test_all_formulations_same_cost(results):
    costs = [results[k][0]["mip_obj"] for k in FORMULATIONS]
    assert max(costs) - min(costs) < 1e-3 * EXPECTED_COST


def test_lp_bounds(results):
    lp = {k: results[k][0]["lp_obj"] for k in FORMULATIONS}
    mip = EXPECTED_COST
    for k in lp:                                  # a relaxation never exceeds the optimum
        assert lp[k] <= mip + 1e-6
    assert lp["C"] >= lp["A"] - 1e-6              # C = A + extra valid inequalities
    assert lp["B"] >= lp["A"] - 1e-6              # holds on this instance (8133.75 vs 8087.67)
    assert lp["B"] > lp["A"] + 1.0                # and B is strictly tighter here


def test_checker_detects_a_broken_schedule(results):
    _, _, model = results["A"]
    inst = tiny_instance()
    model.p["G_base", 3].set_value(1000.0)       # impossible output
    checks = verify_solution(inst, model)
    assert not checks["demand_met"][0]
    assert not checks["output_limits"][0]


def test_reducer_scales_demand_and_keeps_requested_size():
    files = find_instances("data")
    if not files:
        pytest.skip("no benchmark file in data/")
    raw = read_json(files[0])
    red = reduce_instance(raw, n_generators=10, n_hours=12)
    assert len(red["thermal_generators"]) == 10 and len(red["demand"]) == 12
    cap = sum(g["power_output_maximum"] for g in red["thermal_generators"].values())
    assert max(red["demand"]) < cap
    must = sum(g["power_output_minimum"] for g in red["thermal_generators"].values()
               if g["must_run"])
    assert min(red["demand"]) >= must - 1e-9
    inst = parse_instance(red, "reduced")
    assert inst.n_hours == 12


def test_reduced_real_instance_agrees_across_formulations():
    files = find_instances("data")
    if not files:
        pytest.skip("no benchmark file in data/")
    inst = load_instance(files[0], 8, 8)
    costs = []
    for k in FORMULATIONS:
        row, _, _ = run_formulation(inst, k, SolverConfig(time_limit=60, mip_gap=1e-6), verbose=False)
        assert row["status"] == "optimal" and row["checks_passed"] is True
        costs.append(row["mip_obj"])
    assert max(costs) - min(costs) < 1e-4 * abs(min(costs))
