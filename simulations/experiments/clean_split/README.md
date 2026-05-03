# Clean Split QUBO Experiments

## התוצאות

| Metric | Phase A (oracle I) | Phase B (learned I) | Ratio |
|--------|-------------------|---------------------|-------|
| SW ratio | 0.1149 | 0.0952 | **82.9%** |

## הקבצים

```
clean_split_final.py   ← האלגוריתם הסופי (V17 best config)
clean_split_qubo.py    ← QUBO formulation המקורי
archive/               ← כל 22 הגרסאות הקודמות
```

## האלגוריתם הסופי

```python
# UCB-Greedy — BEST Phase B algorithm
# Key: B_init=0.7 (not 0.5)

B_hat = np.full((N, M), 0.7)  # Optimistic init
visits = np.zeros((N, M))

for step in range(T):
    # Route selection: UCB-Greedy only (no I_hat in routing)
    for i in range(N):
        scores = B_hat[i] + 0.5 / sqrt(visits[i])
        route[i] = argmax(scores)
    
    # B_hat update (from observed throughput)
    for i in range(N):
        B_hat[i, route[i]] += 0.12 * (tp[i] - B_hat[i, route[i]])
        visits[i, route[i]] += 1
```

## מה למדנו

1. **SA+QUBO with learned I_hat fails** — the identifiability problem creates a feedback loop
2. **B_init=0.7 is critical** — optimistic initialization prevents premature convergence
3. **I_hat in off-diagonal is always destructive** — even when small
4. **The 18% gap is structural** — cannot be closed without direct interference measurement
5. **UCB-Greedy works because it ignores I_hat** — simpler is better here

## References

- Vault: `qa-mab-multi-agents/knowledge/clean-split-qubo-results.md`
- Results: `results/clean_split/clean_split_final_results.json`
- All variants: `archive/`