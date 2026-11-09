"""
validate_with_egret.py -- compare OUR objective value with EGRET's on one small instance.

EGRET (https://github.com/grid-parity-exchange/Egret) is used ONLY as a referee,
never as the implementation.  If EGRET is not installed the script says so and
exits normally.

    pip install git+https://github.com/grid-parity-exchange/Egret.git
    python validate_with_egret.py --file data/2014-09-01_reserves_0.json --gens 10 --hours 24

Fairness: we feed both tools the same reduced instance in which every generator
has ONE (hot) startup cost, because that is what this project models.  EGRET
may still differ slightly (it has its own modelling conventions), so a small
relative difference is reported as a warning, not an error.

NOTE: the EGRET calls below follow EGRET's documented API from memory and could
not be executed where this script was written; adjust them if your EGRET version differs.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

try:
    import egret  # noqa: F401
    from egret.models.unit_commitment import solve_unit_commitment
    from egret.parsers.pglib_uc_parser import create_ModelData
except ImportError:
    print("EGRET is not installed - skipping validation.\n"
          "Install it with:  pip install git+https://github.com/grid-parity-exchange/Egret.git")
    sys.exit(0)

from src.data_loader import DEFAULT_COST_MULTIPLIER, parse_instance, read_json, reduce_instance
from src.solve_and_metrics import SolverConfig, run_formulation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/2014-09-01_reserves_0.json")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--solver", default="appsi_highs")
    ap.add_argument("--tol", type=float, default=1e-3, help="allowed relative difference")
    args = ap.parse_args()

    raw = reduce_instance(read_json(args.file), args.gens, args.hours, hot_start_only=True)
    inst = parse_instance(raw, "egret_check", DEFAULT_COST_MULTIPLIER)

    ours = {}
    cfg = SolverConfig(name=args.solver, time_limit=300, mip_gap=1e-6)
    for key in ("A", "B"):
        row, _, _ = run_formulation(inst, key, cfg, verbose=False)
        ours[key] = row["mip_obj"]
        print(f"our formulation {key}: {row['mip_obj']:,.4f} $  ({row['status']})")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reduced.json"
        path.write_text(json.dumps(raw))
        md = create_ModelData(str(path))
        sol = solve_unit_commitment(md, args.solver, mipgap=1e-6, timelimit=300,
                                    solver_tee=False)
    egret_cost = float(sol.data["system"]["total_cost"]) * DEFAULT_COST_MULTIPLIER
    print(f"EGRET              : {egret_cost:,.4f} $")

    for key, val in ours.items():
        rel = abs(val - egret_cost) / abs(egret_cost)
        verdict = "MATCH" if rel <= args.tol else "DIFFERENT (check modelling conventions)"
        print(f"  {key} vs EGRET relative difference = {rel:.2e}  -> {verdict}")


if __name__ == "__main__":
    main()
