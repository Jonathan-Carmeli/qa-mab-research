# QA-MAB Publication Results

**Date:** 2026-04-18  
**Experiments:** N=[5,10,12,15,20,30,50], m=4, T=1000, 20 runs each

## Final Results

| N | NB3R | QA-MAB | QA-MAB+FixB | Random | Δ (QA-NB3R) | p-value |
|---|------|--------|-------------|--------|-------------|---------|
| 5 | +2.84 | +2.29 | +2.27 | +1.81 | **-0.55** | <0.001 *** |
| 10 | +0.79 | +0.68 | +0.57 | -1.43 | **-0.11** | 0.28 (NS) |
| **12** | **-1.51** | **-1.11** | **-1.03** | **-4.12** | **+0.40** | **<0.001 *** ← CROSSOVER** |
| 15 | -6.65 | -5.80 | -5.79 | -9.80 | **+0.85** | <0.001 *** |
| 20 | -19.41 | -16.97 | -16.93 | -23.00 | **+2.44** | <0.001 *** |
| 30 | -60.52 | -54.66 | -54.83 | -64.82 | **+5.86** | <0.001 *** |
| 50 | -201.76 | -186.80 | -186.31 | -207.78 | **+14.96** | <0.001 *** |

## Key Findings

### 1. Crossover at N≈12
QA-MAB statistically significantly outperforms NB3R from N=12 onwards (p<0.001). At N=10 there is no significant difference. At N<10, NB3R is better.

### 2. Advantage Grows with N
The advantage grows monotonically:
- N=12: +0.40
- N=15: +0.85
- N=20: +2.44
- N=30: +5.86
- N=50: +14.96

### 3. QA-MAB+FixB Slightly Better at Large N
At N=50: FixB=-186.31 vs Baseline=-186.80 (+0.49 advantage).

### 4. NB3R Collapses at Scale
At N=50: NB3R=-201.76, Random=-207.78. NB3R is barely above random! The shared SW signal has completely collapsed.

## Statistical Notes
- All comparisons use paired t-test (same seeds)
- NS = not significant (p>0.05)
- *** = p<0.001
- Error bars represent 95% CI over 20 runs

## Plots
See `results/` directory for:
- `scaling.png` — SW vs N with error bars
- `convergence_N20.png` — SW(t) trajectories at N=20
- `delta_qamab_vs_nb3r.png` — advantage vs N

## Citation
This work: QA-MAB extension of DIAMOND (arXiv:2303.15544)
