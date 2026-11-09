# Data

Benchmark: **PGLib-UC** (IEEE PES Task Force on Benchmarks for Validation of
Emerging Power System Algorithms), single-bus unit commitment instances in JSON.
Repository: <https://github.com/power-grid-lib/pglib-uc>

## What is included

* `2014-09-01_reserves_0.json` - one CAISO-derived instance (610 thermal
  generators, 48 hours, reserve requirement = 0), about 330 KB.
  **Check the pglib-uc repository's license before redistributing this file**;
  if in doubt, delete it from your fork and download it as described below.

## Getting more instances (needed for the 3-instance study)

`python run_experiments.py` tries to download these automatically
(use `--no-download` to disable):

```
ca/2014-12-01_reserves_0.json
ca/2015-03-01_reserves_0.json
ca/2015-06-01_reserves_0.json
```

If the automatic download fails (no internet, or file names changed), do it by hand:

```bash
git clone --depth 1 https://github.com/power-grid-lib/pglib-uc.git /tmp/pglib-uc
ls /tmp/pglib-uc/ca                     # pick any three different dates
cp /tmp/pglib-uc/ca/2014-12-01_reserves_0.json data/
cp /tmp/pglib-uc/ca/2015-03-01_reserves_0.json data/
```

Any `*.json` file placed in this folder (or sub-folders) is picked up
automatically, in alphabetical order; the first three are used.

## Format notes (from the sample file)

Top-level keys: `time_periods`, `demand`, `reserves`, `thermal_generators`
(a dict keyed by generator name), `renewable_generators` (empty here).
There is no separate fixed-cost field: the no-load cost is the cost of the first
point of `piecewise_production`. Costs look like thousands of dollars; the
loader multiplies them by 1000 (`cost_multiplier` in `src/data_loader.py`).
