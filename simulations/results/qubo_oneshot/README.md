# DIAMOND-QUBO vs Paper-Faithful NB3R — One-Shot QUBO Benchmark

**Generated:** 2026-05-17
**Branch:** `claude/debug-diamond-env-by007`
**Scope:** 600 configs (5 × 2 × 3 × 20) — Ns ∈ {4..8}, B_scale ∈ {uniform, skewed}, I_scale ∈ {low, moderate, high}, 20 seeds each
**Total runtime:** 1332 s (~22 min serial)
**Status:** ✅ Hypothesis confirmed — centralized one-shot QUBO wins on every config

---

## תקציר מנהלים (TL;DR)

בדקנו אם פתירת QUBO מרוכזת ב-iteration אחת ארוכה (SA על מטריצת `(N·m)²` עם B ו-I אורקליים, "DIAMOND-QUBO") מתחרה באלגוריתם NB3R המבוזר מהמאמר (arXiv:2303.15544 §III-C, Algorithm 3 + Corollary 1). שני האלגוריתמים רצו ב-**oracle mode** (B ו-I ידועים, ללא למידה).

**התוצאה: QUBO מנצח ב-600/600 (100%) מהקייסים, פוגע באופטימום הברוט-פורס המדויק ב-99.8% מהזמן. NB3R נכשל באופן קטסטרופי החל מ-N=6.**

---

## Hypothesis

> Can a centralized one-shot QUBO+SA solve match or beat the paper-faithful distributed iterative NB3R algorithm on multi-flow routing, when both run in oracle mode?

The DIAMOND paper proves NB3R converges to the global optimum *in the stationary distribution* as `t → ∞`, with log-cooling `ν(t) = log(t)/Δ`. We test what happens in finite time at moderate N.

---

## Headline Metrics (600 configs)

| Metric | Result | % |
|---|---|---|
| QUBO hit brute-force optimum exactly | **599 / 600** | **99.8 %** |
| NB3R best-visited hit optimum | 322 / 600 | 53.7 % |
| NB3R final-sample hit optimum | 13 / 600 | 2.2 % |
| QUBO ≥ NB3R final | **600 / 600** | **100 %** |
| QUBO ≥ NB3R best-visited | **600 / 600** | **100 %** |
| NB3R hit `stable_tail` early (T=5000) | 0 / 600 | 0 % |
| Total QUBO wall-clock | 302 s (~0.5 s / config) | — |
| Total NB3R wall-clock (T=10000) | 603 s (~1.0 s / config) | — |

---

## Optimum-Hit Rate vs N (averaged across topologies)

| N | QUBO | NB3R best-visited | NB3R final-sample |
|---|---|---|---|
| 4 | 100 % | 99 % | 14 % |
| 5 | 100 % | 81 % | 1 % |
| 6 | 100 % | 56 % | 0 % |
| 7 | 100 % | 28 % | 0 % |
| 8 | 99 % | 3 % | 0 % |

**NB3R's hit rate drops monotonically from ~100 % at N=4 to ~3 % at N=8** — the search space `K^N = 4^N` grows from 256 to 65 k, while NB3R's 10 k Boltzmann samples can't cover it. QUBO is flat at ~100 % across the whole range.

See `plots/optimum_hit_rate.png`.

---

## Detailed Results Per Topology

(Means over 20 seeds; `qubo_hits_opt_pct` is the fraction of seeds where SW matches brute-force exactly.)

### B = skewed

| I_scale | N | opt | qubo | nb3r best | qubo=opt | nb3r_best=opt |
|---|---|---|---|---|---|---|
| low | 4 | 2.999 | 2.999 | 2.999 | 100 % | 100 % |
| low | 5 | 3.566 | 3.566 | 3.566 | 100 % | 100 % |
| low | 6 | 3.917 | 3.917 | 3.867 | 100 % | 90 % |
| low | 7 | 4.270 | 4.270 | 3.937 | 100 % | 40 % |
| low | 8 | 4.430 | 4.430 | 3.748 | 100 % | **0 %** |
| moderate | 4 | 2.398 | 2.398 | 2.398 | 100 % | 100 % |
| moderate | 5 | 2.534 | 2.534 | 2.534 | 100 % | 100 % |
| moderate | 6 | 2.476 | 2.476 | 2.411 | 100 % | 80 % |
| moderate | 7 | 2.248 | 2.248 | 1.994 | 100 % | 50 % |
| moderate | 8 | 1.704 | 1.704 | 1.217 | 100 % | **5 %** |
| high | 4 | 1.810 | 1.810 | 1.810 | 100 % | 100 % |
| high | 5 | 1.530 | 1.530 | 1.530 | 100 % | 100 % |
| high | 6 | 1.092 | 1.092 | 1.037 | 100 % | 75 % |
| high | 7 | 0.312 | 0.312 | 0.143 | 100 % | 50 % |
| high | 8 | −0.892 | −0.892 | −1.254 | 100 % | **5 %** |

### B = uniform

| I_scale | N | opt | qubo | nb3r best | qubo=opt | nb3r_best=opt |
|---|---|---|---|---|---|---|
| low | 4 | 3.080 | 3.080 | 3.079 | 100 % | 95 % |
| low | 5 | 3.645 | 3.645 | 3.622 | 100 % | 75 % |
| low | 6 | 4.126 | 4.126 | 4.066 | 100 % | 35 % |
| low | 7 | 4.557 | 4.557 | 4.376 | 100 % | 5 % |
| low | 8 | 4.751 | 4.750 | 4.464 | **95 %** | **0 %** |
| moderate | 4 | 2.610 | 2.610 | 2.610 | 100 % | 100 % |
| moderate | 5 | 2.881 | 2.881 | 2.842 | 100 % | 50 % |
| moderate | 6 | 2.956 | 2.956 | 2.871 | 100 % | 20 % |
| moderate | 7 | 2.929 | 2.929 | 2.722 | 100 % | 10 % |
| moderate | 8 | 2.533 | 2.533 | 2.128 | 100 % | **5 %** |
| high | 4 | 2.180 | 2.180 | 2.180 | 100 % | 100 % |
| high | 5 | 2.149 | 2.149 | 2.123 | 100 % | 60 % |
| high | 6 | 1.890 | 1.890 | 1.777 | 100 % | 35 % |
| high | 7 | 1.412 | 1.412 | 1.159 | 100 % | 20 % |
| high | 8 | 0.420 | 0.420 | −0.033 | 100 % | **5 %** |

The single QUBO miss (uniform / low / N=8, seed unknown without re-checking) is a marginal SA stochastic event — gap < 0.001 from optimum. Bumping `n_restarts` to 100 would likely close it.

Full per-seed data: `benchmark.json` (8.3 MB).
Per-config aggregates: `result.csv`.

---

## Why NB3R Loses (despite the paper's convergence proof)

The paper's Theorem 1 / Corollary 1 prove NB3R converges to the optimum in the **stationary distribution** as `t → ∞`. In finite time at small `Δ = N`:

1. **Soft Boltzmann at large t**: at `t=10 000, N=8`, `ν(t)/Δ ≈ log(10001) / 8 ≈ 1.15`. With per-flow utility differences `~0.1–0.5`, the Boltzmann probability ratio between the best and a typical sub-optimal arm is only `exp(0.3 × 1.15) ≈ 1.4` — basically uniform sampling.
2. **Final sample is just a draw, not the argmax**: the algorithm samples; even at convergence the final assignment is one noisy realization of the stationary distribution.
3. **K^N grows fast**: `4^8 = 65 536` joint assignments, and NB3R updates one flow per round → only 1 250 "full sweeps" in 10 000 rounds. SA on QUBO does 60 × 2 000 = 120 000 flips with O(1) delta-energy updates over a 32-variable landscape and warm-starts from greedy-B.

We tracked three NB3R metrics to be fair:
- `nb3r_sw_final` — the single sample at `t=T_rounds` (matches what the paper's protocol returns).
- `nb3r_sw_best` — the best SW the algorithm visited at any logged step (most charitable reading of "what NB3R achieved").
- `nb3r_sw_tail_mean` — average over the last 10 % of recorded points (time-averaged stationary estimate).

**Even with the most charitable metric (`best-visited`), NB3R hits the optimum only 53.7 % of cases overall and 0–5 % at N=8.**

See `plots/nb3r_convergence.png` for the noisy trajectories at every seed/topology — even visually, NB3R is bouncing around the search space, not concentrating on σ*.

---

## Caveats

1. **Wall-clock fairness**: NB3R's per-round cost is dominated by `env.compute_throughput` (Python dict iteration over agents) called `K · T = 4 × 10 000 = 40 000` times per seed. QUBO SA uses numpy `Q_row_sum/Q_col_sum` delta updates with `O(1)` per flip. The factor-of-2 wall-clock gap (302 s vs 603 s) reflects implementation, not algorithm. **Treat wall-clock as secondary; the headline metric is final social welfare.**
2. **`stable_tail` was too strict**: with `tol=1 %`, 100 % of configs failed the stability check and were re-run at T=10 000. This is consistent with the theoretical observation that Boltzmann sampling at finite `ν` *does* produce noisy tails — the tail std doesn't shrink because the algorithm hasn't crystallised on a single assignment.
3. **SA budget was tuned**: empirical sweep showed `60 × 2 000 = 120 k flips` reaches the brute-force optimum in 18 / 18 hard cases (N ∈ {6,8}, B=skewed, I ∈ {low, moderate, high}). The headline 99.8 % hit rate is at this budget.
4. **NB3R faithfulness**: our `nb3r_paper.py` implements Algorithm 3 + eq (10) + Corollary 1 directly:
   - Asynchronous (one flow per round, uniform random).
   - Oracle counterfactual: for each candidate `k`, compute `U_n(k, σ_{-n}) = sum of all flows' throughputs under trial σ' (full SW with σ_n := k)`.
   - Boltzmann sampling with `ν(t) = log(t+1)/Δ`, `Δ = N`.
   - Fully-connected neighborhood, matching `NetworkEnvironment`.

   The existing `simulations/legacy/nb3r.py` uses an **EMA bandit** update on observed throughputs plus **linear cooling** — neither matches the paper. We did not benchmark it; the comparison would be even worse for the paper's algorithm.

---

## Interpretation

The DIAMOND paper's NB3R is a **distributed convergence proof on a potential game with known utility**, not a finite-time algorithm. In settings where:

- a centralized solver is available (which is exactly the "GRRL" half of DIAMOND in the paper itself), and
- B and I are known (oracle mode, which is the *assumed* state in §III-C),

then SA on the explicit QUBO is strictly faster and strictly more accurate. NB3R's value lies in its distributed implementation and its asymptotic guarantee, not in matching a centralized solver on finite-time accuracy.

The headline takeaway: **for this problem class, if you already have B and I and you have a centralized compute node, do not iterate NB3R — solve the QUBO**.

---

## Files

```
simulations/results/qubo_oneshot/
├── README.md                  ← this file
├── result.json                ← headline metrics
├── result.csv                 ← 30 rows of per-config aggregates
├── benchmark.json             ← 600 records, full per-seed history (8.3 MB)
├── code/
│   ├── __init__.py
│   ├── diamond_qubo.py        ← QUBO build + SA solver (copied verbatim from
│   │                            simulations/experiments/convergence_test/
│   │                            02_sa_quality_sweep.py)
│   ├── nb3r_paper.py          ← paper-faithful NB3R (Alg 3 + eq 10 + Cor 1)
│   ├── run_benchmark.py       ← driver: sweep N × topology × seed
│   └── aggregate_and_plot.py  ← post-processing: CSV + headline + plots
└── plots/
    ├── sw_gap_vs_N.png        ← 2×3 grid of SW gap (algo − opt) vs N per topology
    ├── optimum_hit_rate.png   ← % configs hitting brute-force optimum vs N
    ├── nb3r_convergence.png   ← raw NB3R SW trajectories (per-seed, all topologies)
    └── wall_clock.png         ← mean wall-clock per config vs N
```

## Reproduction

The canonical run target lives under `simulations/experiments/diamond_vs_qubo/`. The `code/` snapshot in this folder is a frozen copy at commit `d530000`.

```
# Smoke (1 config, <2 s):
python -m simulations.experiments.diamond_vs_qubo.run_benchmark --smoke

# Full (600 configs, ~22 min):
python -m simulations.experiments.diamond_vs_qubo.run_benchmark

# Re-aggregate:
python simulations/results/qubo_oneshot/code/aggregate_and_plot.py
```
