# QA-MAB Round 2 — Final Conclusions

## Experiment Results

### Round 1 (6 "bug fixes" — all made things worse)
Every "fix" degraded QA-MAB below baseline 0.98. The double-counting is a feature, not a bug.

### Round 2 (6 new approaches — none beat NB3R)
```
NB3R:                    1.03 ✓
Baseline (no fixes):     0.98 (-0.05)
Fix B + I:               0.96 (-0.07)
Fix L (no I_hat):        0.92 (-0.11)
```

**Fix L insight:** Removing I_hat entirely (pure argmax u_hat) scores 0.92. The interference QUBO term adds only +0.06 over doing nothing with it.

## The Fundamental Problem (Opus Analysis)

### NB3R has MORE information than QA-MAB

NB3R's W[i,k] directly estimates:
`E[Social Welfare | agent i picks route k, others play current policy]`

This **single scalar** encodes B, I, AND the joint policy distribution. It's the exact quantity to maximize.

QA-MAB tries to reconstruct this from two noisy pieces:
- u_hat ≈ B - E[interference] (already contains I implicitly)
- I_hat ≈ noisy, inflated estimate of pairwise interference (already in u_hat)

The QUBO optimizer has **nothing to optimize** that NB3R isn't already doing via best-response.

### Can QA-MAB(SA) ever beat NB3R?

**At N=10: No.** The optimization is trivial (4^10 ≈ 1M states, SA handles it). The bottleneck is estimation quality, and NB3R has an inherent information advantage.

**At any N with SA: Unlikely.** SA ≈ best-response for moderate frustration. NB3R IS softmaxed best-response with the true reward signal.

### Where QA WOULD help

The crossover requires:
- N ≥ 100-200 with **frustrated** interference (spin-glass structure)
- A QUBO with **accurate** estimates (so global minimum ≈ true SW maximum)
- **Real QA** (not SA) — quantum tunneling through barriers SA can't cross

## Correct Thesis Narrative

### ❌ Don't claim: "QA-MAB beats NB3R"

### ✅ Do claim:

1. **Framework contribution:** Map multi-flow routing → QUBO → annealer-ready. Show τ(t) unifies both algorithms under Gibbs sampling.

2. **Strong baseline result:** NB3R with direct SW feedback is **surprisingly strong** — implicitly encodes B and I in one sufficient statistic. Many papers overclaim centralized wins without this comparison.

3. **QA regime characterization:** Analytically + empirically (QAOA/BF at small N) identify where QUBO's global minimum differs from NB3R's coordinate-ascent fixed point. This happens with: frustrated interference, dense conflict topology, large N.

4. **Honest empirical finding:** QA-MAB(SA) ≈ NB3R in tested regimes. The gap is reserved for real quantum hardware on frustrated instances.

5. **Future work:** D-Wave evaluation on Advantage2 for N=50-100 sparse networks.

> This is a stronger, more defensible thesis than a contrived win. It motivates quantum correctly: not "QA solves what nothing else can," but "QA is a tool for a specific regime where distributed best-response fails."

## Links
- [[qa-mab-opus-deep-analysis]] — Bug analysis
- [[qa-mab-T100k-analysis]] — T=100K results
- [[qa-mab-hardware-analysis]] — D-Wave capability
- [[qa-mab-algorithm-deep-dive]] — Code-level analysis

## Tags
#thesis #quantum #conclusions #pivotal #honest-assessment
