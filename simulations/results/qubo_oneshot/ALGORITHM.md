# Algorithm Reference — DIAMOND-QUBO

Deep dive into the centralized one-shot QUBO+SA solver used in this experiment, the QUBO formulation, and the simulated-annealing inner loop. The companion `nb3r_paper.py` is documented in the appendix.

---

## 1. The optimization problem

The environment (`NetworkEnvironment` in `simulations/legacy/simulation_core.py`) is the abstract multi-flow routing model of the DIAMOND paper, parameterised by:

- `N` agents (flows), each with `m` candidate routes (`m = 4` in this experiment),
- `B ∈ ℝ^{N×m}` — per-flow per-route base utility (e.g. throughput when there is no interference),
- `I ∈ ℝ^{N×m×N×m}` — interference tensor; `I[i,k,j,l]` is the utility i loses when i is on route k and j is on route l. The diagonal `I[i,k,i,l] = 0`.

Each agent picks one route. Let `k_i ∈ {0,…,m−1}` be agent i's choice. Per-flow throughput:

```math
U_i(k_1,\dots,k_N) \;=\; B[i, k_i] \;-\; \sum_{j \ne i} I[i, k_i, j, k_j]
```

Social welfare:

```math
\mathrm{SW}(\mathbf{k}) \;=\; \sum_{i=1}^{N} U_i(\mathbf{k}) \;=\; \sum_{i=1}^{N} B[i,k_i] \;-\; \sum_{i \ne j} I[i,k_i,j,k_j]
```

The objective is

```math
\mathbf{k}^{*} \;=\; \arg\max_{\mathbf{k} \in \{0,\dots,m-1\}^N} \mathrm{SW}(\mathbf{k}).
```

This is a combinatorial optimization over `m^N` joint assignments — NP-hard in general (a generalised assignment problem with quadratic interaction).

For reference, at `N=8, m=4` the search space has `4^8 = 65 536` points.

---

## 2. Binary encoding

Encode the choice with one binary variable per (agent, route) pair:

```math
x_{i,k} \in \{0,1\}, \qquad x_{i,k} = 1 \iff \text{agent } i \text{ uses route } k.
```

For this to represent a valid assignment we require the **one-hot constraint**

```math
\sum_{k=0}^{m-1} x_{i,k} = 1 \quad \forall i \in \{1,\dots,N\}.
```

The vector `x ∈ {0,1}^{N·m}` has dimension `Nm`. We index it as `ik := i·m + k`.

Under this encoding, SW rewrites as

```math
\mathrm{SW}(x) \;=\; \sum_{i,k} B[i,k]\, x_{i,k} \;-\; \sum_{i \ne j} \sum_{k,l} I[i,k,j,l]\, x_{i,k}\, x_{j,l}.
```

---

## 3. QUBO formulation

A QUBO ("Quadratic Unconstrained Binary Optimization") is an energy function of the form

```math
E(x) \;=\; x^\top Q\, x \;=\; \sum_{a} Q_{a,a}\, x_a \;+\; \sum_{a \ne b} Q_{a,b}\, x_a\, x_b,
```

where `x ∈ {0,1}^d` and we minimise `E`. The trick is to encode both the objective and the constraints into `Q`.

We want to **minimise** energy, but **maximise** SW. So we negate utility. We have three pieces:

### 3.1 Base utility (linear, diagonal of Q)

Maximising `sum_{i,k} B[i,k] x_{i,k}` ⇔ minimising `-sum B[i,k] x_{i,k}`. This is purely diagonal:

```math
Q_{ik,ik} \mathrel{+}= -B[i,k]
```

### 3.2 Interference (cross-agent, off-diagonal)

Minimising `-(-sum_{i≠j,k,l} I[i,k,j,l] x_{ik} x_{jl})` = minimising `+sum I[i,k,j,l] x_{ik} x_{jl}`:

```math
Q_{ik,\,jl} \mathrel{+}= I[i,k,j,l] \qquad (i \ne j)
```

Note this is intentionally **asymmetric** — we put each `I[i,k,j,l]` once at position `(ik, jl)`, not split between `(ik,jl)` and `(jl,ik)`. The off-diagonal product `x_{ik} x_{jl}` is counted by `Q_{ik,jl} + Q_{jl,ik}` (both fire whenever both bits are 1), so the natural pairing is to place the asymmetric `I[i,k,j,l]` on one side and `I[j,l,i,k]` on the other. They are typically different (interference is not symmetric), and both copies are needed.

In code we just loop over all `(i,k,j,l)` with `i ≠ j`:
```python
for i,k,j,l: Q[ik, jl] = I[i,k,j,l]    # j != i
```

### 3.3 One-hot penalty (same-agent, off-diagonal)

We have an *unconstrained* binary optimiser; the one-hot constraint needs to be encoded as an energy penalty. The standard way:

```math
P_i(x) \;=\; \lambda\,\bigg(\sum_k x_{i,k} - 1\bigg)^2 \;=\; \lambda\,\bigg[\Big(\sum_k x_{i,k}\Big)^2 \;-\; 2\sum_k x_{i,k} \;+\; 1\bigg].
```

Expanding (since `x_{i,k}^2 = x_{i,k}`):

```math
\left(\sum_k x_{i,k}\right)^2 \;=\; \sum_k x_{i,k} \;+\; 2\sum_{k<l} x_{i,k}\,x_{i,l}.
```

So

```math
P_i(x) \;=\; \lambda\,\bigg[\sum_k x_{i,k} + 2\sum_{k<l} x_{i,k} x_{i,l} - 2\sum_k x_{i,k} + 1\bigg] \;=\; \lambda\,\bigg[-\sum_k x_{i,k} + 2\sum_{k<l} x_{i,k} x_{i,l} + 1\bigg].
```

Dropping the constant `+λ`:

```math
Q_{ik,ik} \mathrel{+}= -\lambda \qquad \text{and} \qquad Q_{ik,il} \mathrel{+}= +\lambda \quad (k<l)
```

In the code we split the penalty symmetrically between `Q[ik,il]` and `Q[il,ik]`, each getting `λ/2`, and put `-λ/2` on the diagonal (because the `-λ` term distributes across both directions). The actual code at `code/diamond_qubo.py:23-39` is:

```python
Q[ik, ik] = -B[i, k] - lambda_ / 2.0                # base utility + one-hot diag
for l in range(k + 1, m):
    Q[ik, il] = lambda_ / 2.0                        # one-hot off-diag (upper)
    Q[il, ik] = lambda_ / 2.0                        # one-hot off-diag (lower)
```

The factor of `1/2` is because both `Q[ik,il]` and `Q[il,ik]` contribute to the product `x_{ik}·x_{il}` in `x^T Q x`, so each must carry half the coefficient.

### 3.4 Choosing λ

`λ` must be large enough that **no valid assignment ever benefits from a one-hot violation**. The maximum gain from picking two routes at once (rather than one) at agent i is bounded by the largest single-route net utility, ≈ 1 in our env (B ≤ 1, I ≥ 0). We use `λ = 0.5` empirically; it turns out to be enough on every config because the route-flip SA proposal (next section) never proposes two-active-bits states in the first place — the proposal preserves the one-hot constraint by construction. So `λ` is only used in the decoded energy and doesn't affect the dynamics.

### 3.5 Full QUBO

The final Q built by `build_oracle_qubo(B, I, lambda_=0.5, tau=1.0)`:

| position | term | reason |
|---|---|---|
| `Q[ik, ik]` | `-B[i,k] - λ/2` | objective + one-hot diagonal |
| `Q[ik, il]`, `Q[il, ik]` for `k ≠ l` | `+ λ/2` | one-hot off-diagonal |
| `Q[ik, jl]` for `i ≠ j` | `+ I[i,k,j,l]` | interference (asymmetric) |

`tau` is an overall energy scale; we leave it at 1.

At `N=8, m=4` the matrix is `32 × 32 = 1024` entries.

---

## 4. Simulated annealing on the QUBO

We never enumerate `m^N` joint assignments. We minimise `E(x) = x^T Q x` heuristically using simulated annealing with **route-flip proposals** that preserve the one-hot constraint by construction.

### 4.1 State representation

Instead of carrying the full bit vector `x ∈ {0,1}^{Nm}`, we carry the compact assignment `active ∈ ℤ^N`, where `active[i] = i·m + k_i` is the index of the bit that is set to 1 for agent i. All other bits are implicitly 0. The bit vector is recovered by `x[active] = 1`.

### 4.2 Proposal: pick an agent, flip its route

At each step:

1. Sample `i` uniformly from `{0, …, N−1}`.
2. Let `k_old = active[i] − i·m` (the current route).
3. Sample a new route `k_new ∈ {0,…,m−1} \ {k_old}` uniformly.
4. The proposed move is `active[i] := i·m + k_new`.

This always keeps exactly one bit active per agent ⇒ the one-hot constraint is preserved automatically. The reachable state space is exactly `{0,…,m−1}^N`, the K^N joint assignments. Mixing is good because every state has `m−1` neighbours per agent.

### 4.3 Energy delta — O(1) per step

Naively, recomputing `E(x) = x^T Q x` after every proposal costs `O((Nm)^2)`. Instead we maintain two auxiliary arrays:

```math
\mathrm{Q\_row\_sum}[k] \;=\; \sum_{a \in \mathrm{active}} Q[k, a], \qquad \mathrm{Q\_col\_sum}[k] \;=\; \sum_{a \in \mathrm{active}} Q[a, k].
```

These let us compute the energy change of flipping `old_idx → new_idx` (for the same agent) in O(1):

```math
\Delta E \;=\; \big(Q_{\mathrm{diag}}[\mathrm{new}] - Q_{\mathrm{diag}}[\mathrm{old}]\big) \;+\; \big(\mathrm{Q\_row\_sum}[\mathrm{new}] - Q[\mathrm{new},\mathrm{old}]\big) - \big(\mathrm{Q\_row\_sum}[\mathrm{old}] - Q[\mathrm{old},\mathrm{old}]\big) \;+\; \big(\mathrm{Q\_col\_sum}[\mathrm{new}] - Q[\mathrm{old},\mathrm{new}]\big) - \big(\mathrm{Q\_col\_sum}[\mathrm{old}] - Q[\mathrm{old},\mathrm{old}]\big).
```

The `−Q[new,old]` and `−Q[old,old]` corrections subtract out the rows/cols that involve the bit being flipped (it can't interact with itself). On accept, we update the sums in O(Nm):

```python
Q_row_sum += Q[:, new_idx] - Q[:, old_idx]
Q_col_sum += Q[new_idx, :] - Q[old_idx, :]
```

This is the speedup that makes 60 restarts × 2 000 iters = 120 k flips run in ~0.5 s at N=8.

### 4.4 Metropolis accept rule

Standard SA:

- If `ΔE < 0`: accept (the new state has lower energy).
- Else: accept with probability `exp(−ΔE / T)`, where `T` is the current temperature.

```python
if delta < 0 or (T > 1e-10 and pyrng.random() < math.exp(-delta / T)):
    # accept
else:
    # revert
```

### 4.5 Cooling schedule

We use geometric cooling within each restart:

```math
T_{t+1} \;=\; T_t \cdot \text{decay}, \qquad T_0 = T_{\text{start}} \cdot (1 + 0.3\,r),
```

with `T_start = 2.0` and `decay = 0.999`. Each restart `r` slightly raises the initial temperature for diversity. We chose `decay = 0.999` (slow) over the more common `0.95` (fast) so that the 2 000-step run actually explores rather than freezing immediately — slow geometric decay is a practical approximation to log cooling on these timescales.

### 4.6 Restarts and warm start

A single SA run is a local heuristic. We do `n_restarts = 60` independent runs and keep the best (lowest energy) state seen across all of them.

- **Restart 0** initialises greedily: `k_i := argmax_k B[i, k]` for every agent (ignore interference). This is the "no-interference" solution and is typically not bad.
- **Restarts 1..n_restarts−1** start from the greedy assignment but then apply a small random perturbation: pick `n_flips ∈ {1, …, N/3}` random agents and flip each to a random other route. This diversifies starting points without throwing away the structure of the warm start.

### 4.7 Decoding

After SA finishes we read out `assignment[i] = active[i] − i·m`. We then call `env.social_welfare(assignment)` to compute the true SW (which doesn't depend on `λ` since the one-hot is satisfied).

### 4.8 Why this works for our problem

Three reasons SA crushes NB3R here:

1. **Warm start from greedy-B is informative.** SA on restart 0 begins at a state that is already a local max under no-interference. The optimum usually requires only a few flips from there.
2. **Delta updates make per-step cost trivial.** 120 k flips × O(1) ≈ µs each.
3. **Restarts spread the random walk.** Even when one restart gets stuck in a local maximum, another typically finds the global.

---

## 5. Pseudo-code summary

```
function build_oracle_qubo(B, I, λ=0.5):
    Q ← zeros((Nm, Nm))
    for i in 0..N-1:
        for k in 0..m-1:
            ik ← i*m + k
            Q[ik, ik] += -B[i,k] - λ/2
            for l in k+1..m-1:
                Q[ik, il] += λ/2
                Q[il, ik] += λ/2
    for i, k, j, l with i ≠ j:
        Q[i*m+k, j*m+l] += I[i,k,j,l]
    return Q

function sa_solve(Q, N, m, n_restarts, n_iters, T₀=2.0, decay=0.999, B_warm):
    best ← +∞
    for r in 0..n_restarts-1:
        active[i] ← i*m + argmax_k B_warm[i,k]
        if r > 0: perturb active by 1..⌊N/3⌋ random flips
        Q_row_sum ← Q[:, active].sum(axis=1)
        Q_col_sum ← Q[active, :].sum(axis=0)
        energy ← Q_col_sum[active].sum()
        if energy < best: best, best_active ← energy, active
        T ← T₀ · (1 + 0.3·r)
        for step in 0..n_iters-1:
            T ← T · decay
            i ← random agent
            k_new ← random route ≠ current(i)
            ΔE ← O(1) using Q_row_sum, Q_col_sum, Q diagonal
            if ΔE < 0 or rand() < exp(-ΔE/T):
                accept: update active[i], energy, Q_row_sum, Q_col_sum
                if energy < best: best, best_active ← energy, active
            else: revert
    return decode(best_active)

function solve_diamond_qubo(env, n_restarts=60, n_iters=2000):
    Q ← build_oracle_qubo(env.B, env.I)
    assignment ← sa_solve(Q, env.N, env.m, n_restarts, n_iters, B_warm=env.B)
    return assignment, env.social_welfare(assignment)
```

---

## 6. Hyperparameter rationale

| Param | Value | Why |
|---|---|---|
| `λ` (one-hot weight) | 0.5 | Not needed by the dynamics (proposal preserves one-hot), used only in decoded energy. Any positive value works. |
| `T_start` | 2.0 | Initial accept ratio ≈ exp(−1/2) ≈ 0.6 for typical SW gaps ~1 — moderate exploration. |
| `decay` | 0.999 | Over 2 000 steps, T drops from 2.0 to ~0.27; final accept ratio is much lower. Slow enough to mix. |
| `n_iters` | 2 000 | At N=8 with 32 binary vars, the diameter of the state graph is small; 2 000 is comfortably more than needed. |
| `n_restarts` | 60 | Empirical: 18/18 hard configs (N∈{6,8}, B=skewed, I∈{low,moderate,high}) hit brute-force optimum at this setting. |

We did not tune `λ` (any positive value works), `T_start`, or `decay`. The relevant knob is `n_restarts × n_iters`; sub-linear gains beyond `60 × 2 000`.

---

## Appendix A. Why NB3R loses (algorithm comparison)

The companion algorithm is paper-faithful NB3R (DIAMOND arXiv:2303.15544 §III-C, Algorithm 3 + eq (10) + Corollary 1). It works on the same problem but distributes the optimisation:

```
for t = 1, 2, …, T_rounds:
    n ← random agent
    for k in 0..m-1:
        trial ← σ with σ_n := k
        U[k] ← Σ_i U_i(trial)            # collaborative utility = full SW
    sample σ_n from Boltzmann P(k) ∝ exp(ν(t) · U[k])
    where ν(t) = log(t+1) / Δ,  Δ = N
```

Implementation: `code/nb3r_paper.py`.

Theorem 1 of the paper proves: the stationary distribution of σ concentrates on σ* as ν → ∞. Corollary 1 picks the cooling schedule that guarantees this concentration.

The result of running this on our problem (see `result.json`):

- At `t = 10 000, N = 8`: `ν(t)/Δ = log(10001)/8 ≈ 1.15`. The Boltzmann probability ratio between the best arm and a typical sub-optimal arm with SW gap ~0.3 is `exp(0.3 × 1.15) ≈ 1.4` — essentially uniform sampling.
- Per round, NB3R updates exactly one agent. To cover all `m^N = 65 536` joint states, you'd need an enormous number of rounds. SA on QUBO covers ~120 000 proposed states in 0.5 s with O(1) δ-energy and warm-starts.

NB3R is correct *asymptotically*; SA on QUBO is fast *now* given known B and I.

---

## Appendix B. Source files

- `code/diamond_qubo.py` — `build_oracle_qubo`, `sa_solve`, `solve_diamond_qubo`, `brute_force`, `social_welfare`.
- `code/nb3r_paper.py` — `run_nb3r`, `_boltzmann_sample`, `stable_tail`.
- `code/run_benchmark.py` — driver: sweep N × topology × seed, dump JSON.
- `code/aggregate_and_plot.py` — post-processing.

The four helpers in `diamond_qubo.py` (`build_oracle_qubo`, `sa_solve`, `brute_force`, `social_welfare`) are copied verbatim from `simulations/experiments/convergence_test/02_sa_quality_sweep.py` because that filename starts with a digit and isn't directly importable.
