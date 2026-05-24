# Cat 10: SA Algorithm Comparison — Bit-Flip vs Route-Flip

**Status:** ✅ DONE — bit-flip wins decisively (6/6 seeds)

## What We Did
Two SA implementations exist in `sa_solver_physical.py`:

| Function | Method | Encoding | Returns |
|----------|--------|----------|---------|
| `sa_sweep` | bit-flip | (M=N×K,) binary vector | `(best_x, best_energy)` |
| `sa_solve` | route-flip (alias: `sa_onehot`) | (N,) route indices | `(N,) paths` |

Both solve the same QUBO at each step. "Winning" = lower QUBO energy found.

## Design
- Both algorithms run on the **same QUBO** at each step (same shared world topology)
- At each step: build QUBO → run both SA variants → compare energies
- Config: N=4, K=4, P=10, T=30, sigma=0.1, n_seeds=6
- Route-flip params: n_restarts=20, n_iters=200, T0=2.0, decay=0.995
- Bit-flip params: n_reads=20, n_sweeps=200, T∈[2.0→0.05]

## Results

```
Total energy — bit-flip (old):   -24622.55 ± 1277.80
Total energy — route-flip (new): -23197.66 ± 1707.16
Gap (old - new):                -1424.89 ± 587.98
Route-flip wins:                0/6 seeds  ❌
Bit-flip wins:                  6/6 seeds  ✅
```

**bit-flip wins 6/6 seeds** — consistently finds solutions with ~6.1% lower QUBO energy.

### Per-Epoch Gap (bit-flip − route-flip)
| Epoch | Gap |
|-------|-----|
| 0 | −2.52 |
| 1 | −0.95 |
| 2 | −6.39 |
| 3 | −3.04 |
| 4 | −8.22 |
| 5 | −5.34 |
| 6 | −3.22 |
| 7 | −3.98 |
| 8 | −5.56 |
| 9 | −8.27 |

The gap is **consistently negative** across all epochs — bit-flip never loses.

## Analysis: Why Bit-Flip Wins

**Bit-flip (sa_sweep):**
- Works on (M=NK=16,) binary vector — 16 independent bits
- 200 sweeps × 16 bits per restart = 3200 bit-flip attempts per restart × 20 restarts = 64,000 proposals
- Random sweep order each temperature step — mixes bits thoroughly
- Linear cooling from T=2.0 → 0.05 over 200 sweeps — fast exploitation

**Route-flip (sa_onehot):**
- Works on (N=4,) route indices — 4 flows, each with K=4 choices
- 200 iterations × 1 route-flip per iteration × 20 restarts = 4,000 proposals
- Coarser proposal granularity: 1 proposal flips exactly 1 of 4 routes for 1 of 4 flows
- Same transition (flow n: route a→b) requires 1 route-flip (1 proposal) but explores less per proposal

**Effective search density:** Even though both have ~16 degrees of freedom (16 bits vs 4 routes × 4 states), bit-flip's fine-grained proposals allow more targeted moves. A "cross" move in route-flip (flipping multiple flows simultaneously) requires sequential proposals, losing time at temperature T where acceptance is easier.

## Interpretation for Thesis

> **Finding:** The bit-flip SA solver (`sa_sweep`) significantly outperforms the route-flip SA solver (`sa_onehot`) on this QUBO problem. Route-flip was hypothesized to better match the problem structure (one route per flow), but the coarser proposal granularity and fewer total proposals per restart negate this advantage. **Recommendation:** Keep `sa_sweep` as the default QA-MAB solver. `sa_solve` (alias for `sa_onehot`) remains available but is not used by the agent.

## Files
- Script: `simulations/physical_validation/run_compare_sa.py`
- Results: `simulations/physical_validation/results/compare_sa_regret/`
  - `sa_comparison_plots.png` — rolling energy, cumulative gap, per-epoch bars
  - `summary.json` — full numerical results

## Updates
- `qa_mab_physical.py` — unchanged, continues to use bit-flip SA (`sa_solve` which points to `sa_onehot` on remote, but local file uses `sa_sweep`)
- `sa_solver_physical.py` — remote has `sa_solve = sa_onehot`, but this comparison shows `sa_sweep` outperforms

**Note:** Local `sa_solver_physical.py` was out of sync with remote. Pulled from origin/main for this test. Remote uses `sa_solve = sa_onehot` (route-flip as default); local file was older (bit-flip only).