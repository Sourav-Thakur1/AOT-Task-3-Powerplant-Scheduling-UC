# Scheduling Power Plants at Lowest Cost: Which Model Solves Faster?

*Unit Commitment formulation comparison on the PGLib-UC benchmark, implemented in Pyomo.*
BMN 616 - Advanced Optimization Techniques for Managers, IIT Roorkee (Option B: reproduce a problem from the literature).

---

## 1. The problem in plain English

Every day a grid operator must decide **which power plants to switch on in each hour** and **how much each one should produce**, so that demand is met at the lowest total cost. This is the *unit commitment* (UC) problem.

Costs: **fuel** (cheaper plants first) and **startup** (firing up a plant costs money).
Technical rules: a running plant must stay between a minimum and maximum output; it cannot change output too fast (**ramp limits**); once started it must stay on for a minimum time, and once stopped it must stay off for a minimum time.

The on/off decisions are yes/no, so UC is a *mixed-integer linear program* (MILP). The **same** problem can be written in several algebraically different ways ("formulations"). They have identical optimal answers, but a solver may need very different effort. Intuition: a solver first solves an easy "relaxed" version (letting a plant be 30 % on). If the relaxed answer is already close to the true answer, little search is left. A formulation whose relaxation is closer is called **tighter**. This project measures how much tightness is worth.

| | Idea | Where it comes from |
|---|---|---|
| **A** baseline (loose) | total power `p`, simple limits, "big-M" style ramping | Arroyo & Conejo 2000; Carrion & Arroyo 2006 style |
| **B** tight | power *above minimum* `p'`, limits that include startup/shutdown, two-period ramping inequalities | Morales-Espana et al. 2013; Gentile et al. 2017; Damci-Kurt et al. 2016; see Knueven et al. 2020 |
| **C** A + tight limits | A plus B's tight generation limits (in the `p` variable) | stepping stone that separates the effect of the limits from the ramping |

All three share the same objective, demand balance, cost model, minimum up/down time constraints. They **must** return the same optimal cost; the code checks this.

## 2. Mathematical formulations

**Sets**: generators `g in G`, hours `t in {1..T}`. Hour 0 is the known initial state.

**Parameters** (per generator `g`, effective values after the preprocessing in `data_loader.py`):

| symbol | meaning |
|---|---|
| `Pmin, Pmax` | minimum / maximum output when on (MW) |
| `RU, RD` | maximum ramp up / down between two hours when on in both (MW/h) |
| `SU, SD` | maximum output in the hour of startup / the last hour before shutdown (MW), with `Pmin <= SU, SD <= Pmax` |
| `UT, DT` | minimum up / down time (h) |
| `u0, p0` | on/off state and output at hour 0 |
| `cSU` | startup cost (one, hot, startup cost) |
| `(s_k, b_k)` | slope and intercept of linear piece `k` of the convex production-cost curve (no-load cost included) |
| `D_t, R_t` | demand and spinning-reserve requirement in hour `t` (MW) |

**Binary variables** (all formulations): `u[g,t]` on, `v[g,t]` startup, `w[g,t]` shutdown.
Units still bound by their history, and must-run units, have `u` fixed in advance.

### Constraints common to A, B, C

```
(1) u[g,t] - u[g,t-1] = v[g,t] - w[g,t]                     state logic            (Garver 1962)
(2) v[g,t] + w[g,t] <= 1                                    no start and stop in the same hour
(3) sum_{i=t-UT+1..t} v[g,i] <= u[g,t]            t >= UT   minimum up time        (Malkin 2003; Rajan & Takriti 2005)
(4) sum_{i=t-DT+1..t} w[g,i] <= 1 - u[g,t]        t >= DT   minimum down time
(5) sum_g power[g,t] = D_t                                  demand balance (single bus)
(6) sum_g (Pmax u[g,t] - power[g,t]) >= R_t                 spinning reserve (only if R_t > 0)
(7) cost[g,t] >= s_k * power[g,t] + b_k * u[g,t]   for all pieces k   production cost (epigraph, convex curve)
objective:  minimise  sum_{g,t} ( cost[g,t] + cSU[g] * v[g,t] )
```
`power[g,t]` is the MW produced; it is the variable `p` in A and C, and `p' + Pmin*u` in B.
Constraint (7): `u` makes the cost zero when the unit is off. For hour 1, `u[g,0] = u0`.

### Formulation A (baseline)

```
variables:  p[g,t] >= 0      total MW produced
(A1)  Pmin u[g,t] <= p[g,t] <= Pmax u[g,t]                                  generation limits         (18)
(A2)  p[g,t]   - p[g,t-1] <= RU u[g,t-1] + SU v[g,t]                         ramp up, big-M startup     (24)
(A3)  p[g,t-1] - p[g,t]   <= RD u[g,t]   + SD w[g,t]                         ramp down, big-M shutdown  (25)
```
with `p[g,0] = p0`. Weakness: with fractional `u`, `p` can sit near `Pmax*u` while `v` and `w` are small, so the startup/shutdown limits SU, SD hardly bind in the relaxation.

### Formulation B (tight)

```
variables:  p'[g,t] >= 0     MW above minimum;   p = p' + Pmin u
Generation limits including startup and shutdown terms (w[T+1] := 0):
  if UT >= 2 :   p'[g,t] <= (Pmax-Pmin) u[g,t] - (Pmax-SU) v[g,t] - (Pmax-SD) w[g,t+1]               (20)
  if UT  = 1 :   p'[g,t] <= (Pmax-Pmin) u[g,t] - (Pmax-SU) v[g,t] - max(SU-SD,0) w[g,t+1]            (23a)
                 p'[g,t] <= (Pmax-Pmin) u[g,t] - (Pmax-SD) w[g,t+1] - max(SD-SU,0) v[g,t]            (23b)
Two-period ramping (Damci-Kurt et al. 2016):
  (B2)  p'[g,t]   - p'[g,t-1] <= (SU - Pmin - RU) v[g,t] + RU u[g,t]                                   (35)
  (B3)  p'[g,t-1] - p'[g,t]   <= (SD - Pmin - RD) w[g,t] + RD u[g,t-1]                                 (36)
```
Numbers in brackets are equation numbers in Knueven, Ostrowski & Watson (2020).

* **Why `UT = 1` is special.** With a minimum up time of 1 hour a unit may start in hour `t` and stop in `t+1`. Then `v[t]` and `w[t+1]` are both 1 and (20) would subtract both terms and wrongly cut off feasible schedules. For `UT >= 2` this cannot happen (the minimum-up-time constraint forbids it), so (20) is valid. For `UT = 1` the two weaker inequalities (23) are used instead.
* **Why it is the same problem as A.** Each inequality of B is valid for exactly the schedules A allows (checked numerically; see Section 6). B only removes fractional points.
* `SU, SD` are capped at `Pmax` in the loader; otherwise coefficients like `(Pmax - SU)` would turn negative and the inequalities would be useless (though still valid).

### Formulation C

A (A1-A3) plus, for every `g, t`:  `p[g,t] <= Pmin u[g,t] + RHS`, where `RHS` is the right-hand side of (20) or (23) above. Valid, so same optimal cost; its LP relaxation lies between A and B.

## 3. What is measured

For each (instance, size, formulation): number of binary / continuous variables and constraints; LP-relaxation objective; **LP gap** `= 100 * (best known MIP objective - LP objective) / best known MIP objective` (best known = lowest MIP objective found by any formulation on that case); MIP objective, best bound, final MIP gap; solve time (wall clock of the solver call, includes Pyomo's hand-over of the model); status; branch-and-bound nodes when HiGHS reports them.

Solver settings (identical for all): HiGHS via Pyomo `appsi_highs`, 300 s limit, relative gap 0.01 %, 1 thread, seed 0. A run that hits the time limit is recorded with its gap and bound; it never crashes the study.

Experiments (all in `run_experiments.py`):

1. **Main grid**: up to 3 PGLib-UC instances x N = 10, 20, 40 generators x T = 24 hours x formulations A, B, C.
2. **Demand sensitivity**: demand x 0.9, 1.0, 1.1 on the first instance (N = 20).
3. **Ramp stress (extra, modifies the benchmark)**: ramp limits x 1.0, 0.2, 0.1 on the first instance (N = 20). *Why:* in the original California files the ramp limits almost never bind, so formulations that differ in how they treat ramping look identical there. (In my pre-checks the LP bounds of A, B and C on the original data were exactly equal for N = 20 and 40.) Tightening ramps makes the difference visible.

## 4. Install and run

Python 3.10 - 3.12.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # tests (tiny instance, ~seconds)
python run_experiments.py --quick                      # smoke run (minutes)
python run_experiments.py                              # full study
```

Useful options: `--solver gurobi` (or edit `SOLVER_NAME` at the top of `run_experiments.py`), `--time-limit 60`, `--sizes 10 20`, `--n-instances 1`, `--no-download`, `--verbose` (print every schedule check).

Only one benchmark file is committed. The full study wants three: `run_experiments.py` tries to download two more automatically; otherwise see `data/README.md`.

## 5. Reproducing the results

After a full run, `results/` contains:

| file | content |
|---|---|
| `results.csv` | every metric for every run (`experiment` column: main / sensitivity / ramp_stress) |
| `summary_table.csv` | formulation x size: average solve time, average LP gap, average final gap, nodes, time-limit hits |
| `objective_agreement.csv` | per case: do all formulations report the same optimal cost? |
| `fig1_solve_time.png`, `fig2_lp_gap.png`, `fig3_generation_stack.png`, `fig4_ramp_stress.png` | figures |
| `schedule_*.csv` | the solved schedule used for the stack plot |

After every solve the code independently verifies demand met in each hour, output between Pmin and Pmax, minimum up/down times (including history) and ramp limits, and records `checks_passed`.

Optional: `python validate_with_egret.py` compares our objective with EGRET on a small instance (skips politely if EGRET is not installed).

## 6. Results summary

*Fill this section in with your own run - the numbers depend on your machine and solver version.*

What I verified before writing this README (with an independent HiGHS-based stand-in, **not** your Pyomo pipeline):

* Tiny instance: A, B, C all give 8,245 (hand-checked). LP bounds: A 8,087.7 < B = C 8,133.8.
* 192 random small instances with minimum up/down times 1-4, random startup/shutdown limits, initial histories, must-run units and reserves: A, B, C always gave identical optimal costs; B's LP bound was never lower than A's and strictly higher in 105 of them.
* On the first 20 generators of `2014-09-01_reserves_0` with ramps scaled by 0.2: LP bound A 566,994 < C 567,277 < B 567,726 (all below the common MIP optimum).

## 7. Limitations

* **Single bus**: no transmission network, no losses, no congestion.
* **Reduced instances**: first N generators of a 610-unit system, first 24 of 48 hours, demand scaled by kept capacity (and repaired where scaling made it infeasible). "First N" in file order is arbitrary: small subsets can be dominated by one lumpy unit (e.g. N = 10 contains a 126 MW unit whose Pmax - Pmin is only about 1 MW), which inflates LP gaps for *all* formulations.
* **Simplified startup cost**: one hot startup cost per startup; the cooler startup steps in the data are ignored. No shutdown costs (none in the data).
* **Simplified reserve**: nominal headroom of committed units; ramp-rate limits on reserve deliverability are not modelled. The sample file has zero reserve requirement.
* **Cost units**: file costs are multiplied by 1000 (they look like thousand dollars); this is an assumption, easily changed.
* Our B is a compact tight formulation (similar to "Co" in Knueven et al.), not their full "T" formulation: no extended startup-cost variables, no perspective cost formulation.
* Timing comparisons with one thread, one seed, a handful of instances are indicative, not statistically conclusive.

## 8. References

* Carrion, M. and Arroyo, J. M. (2006). A computationally efficient mixed-integer linear formulation for the thermal unit commitment problem. *IEEE Trans. Power Systems* 21(3), 1371-1378.
* Knueven, B., Ostrowski, J. and Watson, J.-P. (2020). On mixed-integer programming formulations for the unit commitment problem. *INFORMS Journal on Computing* 32(4), 857-876.
* Morales-Espana, G., Latorre, J. M. and Ramos, A. (2013). Tight and compact MILP formulation of start-up and shut-down ramping in unit commitment. *IEEE Trans. Power Systems* 28(2), 1288-1296.
* Gentile, C., Morales-Espana, G. and Ramos, A. (2017). A tight MIP formulation of the unit commitment problem with start-up and shut-down constraints. *EURO J. Computational Optimization* 5, 177-201.
* Damci-Kurt, P., Kucukyavuz, S., Rajan, D. and Atamturk, A. (2016). A polyhedral study of production ramping. *Mathematical Programming* 158, 175-205.
* Arroyo, J. M. and Conejo, A. J. (2000). Optimal response of a thermal unit to an electricity spot market. *IEEE Trans. Power Systems* 15(3), 1098-1104.
* Rajan, D. and Takriti, S. (2005). Minimum up/down polytopes of the unit commitment problem with start-up costs. IBM Research Report RC23628. Malkin, P. (2003), same topic (technical report).
* **Benchmark**: PGLib-UC, IEEE PES Task Force on Benchmarks for Validation of Emerging Power System Algorithms. <https://github.com/power-grid-lib/pglib-uc>. The data are described in Knueven et al. (2020); please also check the repository README for its preferred citation.
* Pyomo: Bynum et al., *Pyomo - Optimization Modeling in Python*, 3rd ed., Springer 2021. HiGHS: Huangfu & Hall (2018), *Mathematical Programming Computation* 10, 119-142.
* EGRET (validation only): <https://github.com/grid-parity-exchange/Egret>.

*(Volume/page numbers are from memory - verify them before submitting.)*

## 9. Repository layout

```
README.md  requirements.txt  pytest.ini  LICENSE  .gitignore
run_experiments.py          one command runs everything
validate_with_egret.py      optional referee check against EGRET
src/data_loader.py          read / reduce / parse PGLib-UC JSON, tiny instance
src/formulations.py         Pyomo models A, B, C
src/solve_and_metrics.py    solver wrapper, LP relaxation, checks, metrics
src/plots.py                figures
tests/test_tiny_instance.py hand-checkable instance + agreement tests
data/                       README + one small sample instance
results/                    generated
```
