# DIAMOND-QUBO vs Paper-Faithful NB3R

## Hypothesis

When both algorithms run in **oracle mode** (B and I known, no learning), can a centralized one-shot QUBO+SA solve ("DIAMOND-QUBO") match or beat the paper's distributed iterative NB3R, given NB3R enough rounds to converge?

## What's here

| File | Role |
|---|---|
| `diamond_qubo.py` | Builds the full `(N·m)²` oracle QUBO, runs heavy SA (200 restarts × 5000 iters with O(1) delta updates). One-shot. Returns `(assignment, sw, seconds)`. |
| `nb3r_paper.py` | Paper-faithful NB3R per arXiv:2303.15544 §III-C: asynchronous (one flow per round), oracle counterfactual queries for every candidate path, Boltzmann sampling with **log cooling** `ν(t) = log(t+1)/Δ`. Returns `(assignment, sw_history, seconds)`. |
| `run_benchmark.py` | Driver: sweep `N × B_scale × I_scale × seed`, compute brute-force optimum, run both algorithms, dump JSON. Auto-bumps NB3R rounds from 5 k → 10 k if the SW tail isn't stable. |
| `__init__.py` | Empty marker. |

## Code reused

- `simulations/legacy/simulation_core.py` — `NetworkEnvironment` (env with `B`, `I`, `compute_throughput`, `social_welfare`).
- `simulations/experiments/convergence_test/02_sa_quality_sweep.py` — `build_oracle_qubo`, `sa_solve`, `brute_force`, `social_welfare` were copied verbatim into `diamond_qubo.py` (source filename begins with a digit so it isn't importable).

The existing `simulations/legacy/nb3r.py` is **not** reused — its EMA-bandit update rule and linear cooling don't match the paper.

## How to run

Smoke test (~5 s):
```
python -m simulations.experiments.diamond_vs_qubo.run_benchmark --smoke
```

Full sweep (~30 min serial, 600 configs):
```
python -m simulations.experiments.diamond_vs_qubo.run_benchmark
```

Override output path:
```
python -m simulations.experiments.diamond_vs_qubo.run_benchmark --out path/to/file.json
```

## Output JSON schema

```
{
  "metadata": {
    "smoke": bool,
    "Ns": [4, 5, 6, 7, 8],
    "m": 4,
    "B_scales": ["uniform", "skewed"],
    "I_scales": ["low", "moderate", "high"],
    "n_seeds": 20,
    "T_rounds": 5000,
    "bumped_T_rounds": 10000,
    "total_seconds": float,
    "n_records": int
  },
  "records": [
    {
      "N": int, "B_scale": str, "I_scale": str, "seed": int,
      "opt_sw": float, "opt_assignment": [int],
      "qubo_sw": float, "qubo_assignment": [int], "qubo_time": float,
      "nb3r_sw": float, "nb3r_time": float,
      "nb3r_history": [[t, sw], ...],
      "nb3r_T_rounds": int
    },
    ...
  ]
}
```

## Caveats

- **Wall-clock fairness**: NB3R uses `dict`-based `env.compute_throughput` (Python loops); QUBO SA uses numpy delta updates. The QUBO times will look artificially fast — treat wall-clock as a secondary metric. The headline metric is **final social welfare** vs the brute-force optimum.
- **SA budget**: 200 × 5000 was chosen as a heavy default. For `N=8` high-interference an ablation may be needed to confirm we're on the plateau.
- **NB3R convergence at small Δ**: at `N=8`, `ν(5000)/Δ ≈ 1.06`, still soft. The driver's `stable_tail` check auto-bumps `T_rounds` to 10 000 when the tail SW isn't stable.

## Headline question to answer from the JSON

Per `(B_scale, I_scale)` pair, on what fraction of seeds does `qubo_sw >= nb3r_sw`, and how do both compare to `opt_sw`?
