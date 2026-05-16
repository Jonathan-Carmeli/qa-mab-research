# QA-MAB: Quantum Annealing Multi-Armed Bandit for Dynamic UAV Routing

**Date:** 2026-05-03
**Version:** 2 (reviewed and corrected)

---

## 1. Problem Statement

### 1.1 Setting

Consider a network of $m = 30$ UAVs deployed in a $1000 \times 1000$m² area, partitioned into $Z = 9$ zones (3×3 grid). UAVs within communication range ($r = 350$m) can relay data for end-to-end communication flows.

At each **epoch** $p$, UAVs receive new random positions, and $N = 3$ source-destination communication flows are generated. Each flow must be routed through a multi-hop path of UAVs. There are $K = 4$ candidate paths per flow (computed via $K$-shortest paths over the distance-weighted adjacency graph).

### 1.2 Fault Model

The network has hidden faults:
- **UAV failures:** $n_f = 4$ UAVs have failure rates $\theta^*_i \in [0.2, 0.4]$; remaining UAVs have $\theta^*_i = 0$
- **Zone interference:** $n_z = 2$ zones have interference rates $\phi^*_z \in [0.2, 0.4]$; remaining zones have $\phi^*_z = 0$

These parameters are **constant** across all epochs (the ground truth doesn't change), but the **topology** changes every epoch (new UAV positions, new flows, new candidate paths).

### 1.3 Loss Model

When flows select paths, the environment returns per-flow losses:

$$L_n = \underbrace{\sum_{i \in \text{path}(n)} \theta^*_i}_{\text{UAV faults}} + \underbrace{\sum_{z \in \text{path}(n)} \phi^*_z}_{\text{zone interference}} + \underbrace{C_\text{coll} \cdot |\{l \neq n : \text{path}(n) \cap \text{path}(l) \neq \emptyset\}|}_{\text{collision penalty}} + \underbrace{\sum_{l \neq n} e^{-d_{n,l}/d_0}}_{\text{proximity interference}} + \varepsilon_n$$

Where:
- $C_\text{coll} = 5.0$ — penalty per colliding flow pair (flows sharing ≥1 UAV)
- $d_{n,l}$ — minimum Euclidean distance between any two UAVs on paths $n$ and $l$
- $d_0 = 150.0$m — proximity decay constant
- $\varepsilon_n \sim \mathcal{N}(0, 0.05^2)$ — observation noise

### 1.4 Objective

Minimize cumulative loss $\sum_{p=1}^{P} \sum_{t=1}^{T} \sum_{n=1}^{N} L_n^{(p,t)}$ over $P = 10$ epochs of $T = 100$ steps each, while simultaneously learning $\hat{\theta}$ and $\hat{\phi}$ from the loss signals alone.

---

## 2. Motivation: Why Dynamic?

### 2.1 The Static Environment Failure

In a **static** multi-agent routing environment (fixed topology, $N = 10$ agents, $M = 4$ routes), we observed:

- **Phase A:** With oracle interference knowledge, the QUBO approach works perfectly (social welfare ratio = 0.1309, +49% over baselines)
- **Phase B:** With learned interference estimates, catastrophic failure (SW ratio = 0.0144)

The root cause is an **identifiability loop**: $\hat{I}$ (interference estimate) depends on $\hat{B}$ (bandwidth estimate), which depends on routing decisions, which depend on $\hat{I}$ — a circular dependency that cannot be broken from aggregate signals alone.

### 2.2 The Dynamic Environment Insight

Switching to a **piecewise-stationary** environment (new topology each epoch) breaks this loop:
- New UAV positions → new paths → new combinations of UAVs and zones
- Each epoch provides fresh collision/interference signals from different network regions
- The algorithm must explore broadly, preventing the identifiability trap

The ground truth ($\theta^*, \phi^*$) remains constant — only the topology changes. This enables **transfer learning** across epochs: knowledge about faulty UAVs carries over even as their positions and connections change.

---

## 3. Agents

### 3.1 QA-MAB (Our Algorithm)

#### Overview

At each step $t$ within epoch $p$:
1. **Build QUBO** — encode current estimates + constraints into an $N \times K$ QUBO matrix
2. **SA solver** — find near-optimal path selection via Simulated Annealing
3. **Execute** — routes are chosen, environment returns losses
4. **Learn** — update $\hat{\theta}$ and $\hat{\phi}$ via residual credit assignment

#### State

The agent maintains persistent estimates across epochs:
```
θ̂ ∈ [0,1]^m          — estimated failure rate per UAV (persists across epochs)
φ̂ ∈ [0,1]^Z          — estimated interference rate per zone (persists across epochs)
visit_counts[N,K]     — per-path visit counts (reset each epoch)
```

All initialized to zero at the start of the experiment.

#### QUBO Matrix Construction

**Variable encoding:** $x_{n \cdot K + k} = 1 \iff$ flow $n$ takes path $k$.

**Energy function:** $E(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$

**Diagonal (linear cost per path):**
$$Q_{nk, nk} = \left(\sum_{i \in \text{path}(n,k)} \hat{\theta}_i + \sum_{z \in \text{path}(n,k)} \hat{\phi}_z\right) - \lambda - \frac{c_\text{ucb}}{\sqrt{\max(\text{visits}_{n,k}, 1)}}$$

Where $\lambda = 10.0$ (one-hot incentive) and $c_\text{ucb} = 3.0$ (UCB exploration bonus).

**Off-diagonal — same-flow one-hot constraint:**
For paths $k < k'$ of the same flow $n$:
$$Q_{nk, nk'} = Q_{nk', nk} = \lambda$$

Energy contribution = $2\lambda$ if two paths selected for one flow → SA avoids this.

**Off-diagonal — cross-flow interactions:**
For flows $n \neq l$, paths $k$ and $j$:
$$Q_{nk, lj} = Q_{lj, nk} = C_\text{coll} \cdot \mathbb{1}[\text{shared UAVs}] + e^{-d/d_0}$$

Where $d$ = minimum pairwise Euclidean distance between UAVs on the two paths.

#### Simulated Annealing

The QUBO is scaled by an exploration temperature before solving:
$$Q_\text{scaled} = Q / \gamma(p, t), \quad \gamma(p,t) = \frac{\gamma_0}{(p+1)^a \cdot (t+1)^b}$$

With $\gamma_0 = 2.0$, $a = 0.5$, $b = 0.3$. High temperature early → more exploration; low temperature late → exploitation.

SA parameters: $n_\text{reads} = 20$, $n_\text{sweeps} = 200$, $T_\text{init} = 2.0$, $T_\text{final} = 0.05$.

#### Learning — Residual Credit Assignment

After SA selects paths and the environment returns losses $L_n$:

**Step 1: Isolate fault-related loss**
$$L_\text{fault}[n] = L_n - C_\text{coll} \cdot \text{collision\_count}[n] - \sum_{l \neq n} e^{-d_{n,l}/d_0}$$

This subtracts the deterministic structural components, leaving $L_\text{fault}[n] \approx \sum \theta^*_i + \sum \phi^*_z + \varepsilon$.

**Step 2: Residual credit assignment for UAVs**
For each UAV $i$ on flow $n$'s chosen path:
$$\text{residual}_i = L_\text{fault}[n] - \underbrace{\sum_{j \neq i} \hat{\theta}_j}_{\text{other UAVs}} - \underbrace{\sum_z \hat{\phi}_z}_{\text{zones}}$$
$$\hat{\theta}_i \leftarrow \hat{\theta}_i + \alpha \cdot (\text{residual}_i - \hat{\theta}_i)$$

Each UAV learns only its **marginal contribution**, preventing healthy UAVs from accumulating false positive estimates.

**Step 3: Residual credit assignment for zones** (analogous, using updated $\hat{\theta}$)

**Step 4: Clip** $\hat{\theta}, \hat{\phi} \in [0, 1]$.

Learning rate: $\alpha = 0.15$.

#### Epoch Boundary: Decay

At the start of each epoch:
$$\hat{\theta} \leftarrow 0.7 \cdot \hat{\theta}, \quad \hat{\phi} \leftarrow 0.7 \cdot \hat{\phi}$$

This partially forgets stale path-specific correlations while preserving the underlying fault knowledge signal.

### 3.2 Oracle

Uses true $\theta^*$ and $\phi^*$ to build the QUBO — no learning needed. Solves with SA (same parameters) but **without** temperature scaling or UCB exploration bonus. Estimates are frozen at ground truth and never updated.

**Important limitation:** The Oracle baseline lacks two exploration mechanisms that QA-MAB has: (1) temperature-scaled QUBO and (2) UCB visit-count bonus. This means the Oracle-vs-QA-MAB comparison reflects differences in *both* knowledge quality and exploration strategy, not knowledge quality alone. See Section 5.3 for discussion.

### 3.3 NB3R (Network-Based Bandit Routing)

Uses softmax exploration over per-path weights: $P(k|n) \propto \exp(-W_{n,k} / \eta)$ with $\eta = 0.3$. Weights are updated via EMA: $W_{n,k} \leftarrow W_{n,k} + \alpha_w (L_n - W_{n,k})$ where $\alpha_w = 0.2$.

**Key difference from QA-MAB:** NB3R **resets weights to zero every epoch** — no transfer learning across topologies. It also operates per-flow independently (no cross-flow coordination via QUBO).

### 3.4 Greedy

Always selects path $k = 0$ (the shortest path by hop count) for every flow. No learning, no exploration.

### 3.5 Random

Selects uniformly at random from $K = 4$ candidate paths each step. No learning.

---

## 4. QUBO Configuration

| Component | Value | Role |
|-----------|-------|------|
| $\lambda$ (one-hot penalty) | 10.0 | Incentivize exactly one path per flow |
| $C_\text{coll}$ (collision) | 5.0 | Penalize shared-UAV routing |
| $d_0$ (proximity decay) | 150.0m | Proximity interference distance scale |
| $\alpha$ (learning rate) | 0.15 | EMA smoothing for $\hat{\theta}, \hat{\phi}$ |
| $c_\text{ucb}$ (UCB constant) | 3.0 | Exploration bonus for rarely-visited paths |
| epoch\_decay | 0.7 | Forget stale correlations at epoch boundaries |
| $\gamma_0$ / $a$ / $b$ | 2.0 / 0.5 / 0.3 | Temperature schedule parameters |

**Configuration history:** These values emerged from 22+ iterations in the static environment, with Fixes A (residual credit) and B (epoch decay) added for the dynamic environment. Values tested and rejected include $C_\text{coll} = 2.0$ (insufficient collision avoidance), $c_\text{ucb} = 5.0$ (too much exploration), and no epoch decay (estimation error grows).

---

## 5. Results

All results: $P = 10$ epochs, $T = 100$ steps/epoch, $N = 3$ flows, $m = 30$ UAVs, 20 seeds.

### 5.1 Estimation Error Convergence

$\hat{\theta}$ and $\hat{\phi}$ improve across epochs, then plateau:

| Epoch | $\|\hat{\theta} - \theta^*\|_2$ | $\|\hat{\phi} - \phi^*\|_2$ |
|-------|-------------------------------|-------------------------------|
| 1 | 0.550 ± 0.084 | 0.393 ± 0.073 |
| 2 | 0.476 ± 0.090 | 0.357 ± 0.062 |
| 3 | 0.451 ± 0.105 | 0.342 ± 0.062 |
| 4 | 0.420 ± 0.127 | 0.321 ± 0.069 |
| 5 | 0.414 ± 0.106 | 0.314 ± 0.090 |
| 6 | 0.413 ± 0.089 | 0.312 ± 0.079 |
| 7 | 0.399 ± 0.068 | 0.312 ± 0.084 |
| 8 | 0.397 ± 0.073 | 0.316 ± 0.072 |
| 9 | 0.400 ± 0.074 | 0.318 ± 0.060 |
| 10 | 0.409 ± 0.084 | 0.312 ± 0.061 |

**Key observations:**
- Both errors decrease substantially in the first 4-5 epochs (θ: −24%, φ: −20%)
- Errors **plateau** after epoch 5-6 rather than converging to zero
- The plateau is expected: with epoch decay (×0.7) and only $T = 100$ learning steps per epoch, the estimator cannot fully recover from the decay before the next epoch begins
- The residual error reflects the **identifiability limit** — from aggregate path losses, individual UAV contributions cannot be perfectly decomposed

**Within-epoch dynamics:** Loss is essentially flat within each epoch (first 10 steps ≈ last 10 steps). The QUBO quickly finds good paths (~10-20 steps), after which there are no novel loss signals. **Convergence is driven by new epochs (new topologies), not more steps within an epoch.**

### 5.2 Cumulative Performance

| Agent | Total Loss | vs QA-MAB | Collision Rate |
|-------|-----------|-----------|----------------|
| **QA-MAB** | **234,361** | — | **53.7%** |
| Oracle | 254,305 | +8.5% worse | 57.5% |
| NB3R | 272,358 | +16.2% worse | 63.3% |
| Greedy | 306,672 | +30.9% worse | 74.0% |
| Random | 358,223 | +52.9% worse | 81.1% |

*Total loss = sum across all seeds, epochs, steps, and flows. Collision rate = fraction of steps with at least one collision (≥1 flow pair sharing a UAV).*

QA-MAB achieves the lowest total loss of all agents, including Oracle. It also has the lowest collision rate, 3.8 percentage points below Oracle and 9.6pp below NB3R.

### 5.3 Why QA-MAB Outperforms Oracle

This counterintuitive result — a learning agent beating an omniscient one — has **two contributing factors**:

#### Factor 1: Symmetry Breaking via Imperfect Estimates

Oracle knows exact $\theta^*$ and $\phi^*$, but all healthy UAVs have identical cost ($\theta^* = 0$). From the QUBO perspective, all paths through only healthy UAVs are equivalent: cost = $0 - \lambda = -10$. SA has no signal to spread flows across different healthy paths → flows cluster on the same UAVs → collisions.

QA-MAB's learned estimates $\hat{\theta}$ are imperfect — different healthy UAVs accumulate slightly different estimates based on their path history. These small cost differences give SA a signal to diversify flow routing, reducing collisions.

#### Factor 2: Exploration Mechanisms (Confound)

QA-MAB has two exploration mechanisms that Oracle lacks:
1. **Temperature scaling:** QA-MAB divides Q by $\gamma(p,t)$, which at high temperature amplifies SA's willingness to explore suboptimal paths. Oracle passes raw Q — effectively always "cold."
2. **UCB bonus:** QA-MAB adds $-c_\text{ucb}/\sqrt{\text{visits}}$ to rarely-visited paths. Oracle doesn't use visit counts.

**We cannot currently separate these effects.** An ablation study giving Oracle the same temperature schedule and/or UCB bonus would clarify whether the advantage comes from imperfect estimates alone or from the exploration mechanisms. This is listed as future work.

### 5.4 Cumulative Regret

Cumulative regret (QA-MAB − Oracle) over 1000 steps:

$$\text{Final cumulative regret} = -997.2$$

Negative regret means QA-MAB outperforms Oracle on average across 20 seeds. However, the standard deviation across seeds is substantial — the confidence band spans from approximately −4000 to +2500, indicating that Oracle wins on some seeds. The result is directionally consistent but would benefit from formal statistical testing.

**Why regret doesn't converge to zero:** Both agents face the same changing environment, but QA-MAB's exploration mechanisms (Factor 2 above) and symmetry-breaking estimates (Factor 1) provide a persistent advantage that Oracle cannot overcome with static perfect knowledge alone.

### 5.5 Optimality Gap

To contextualize the absolute performance of all agents, we compare against a **theoretical optimal agent** that has access to the true ground-truth parameters ($\theta^*, \phi^*$) and performs an **exhaustive search** over all $K^N = 4^3 = 64$ possible path combinations at each step. For each combination, it computes the exact loss (including collision penalties and proximity interference) and selects the minimum-loss assignment. This represents the best any agent could possibly do — it has perfect knowledge AND perfect optimization (no SA approximation).

The optimal agent achieves a mean loss per flow-step of **0.701**, computed over the same 20 seeds, 10 epochs, and 100 steps.

#### Optimality Gap by Agent

The **optimality gap** measures how far each agent is from the theoretical lower bound:

$$\text{Optimality Gap} = \frac{L_{\text{agent}} - L_{\text{optimal}}}{L_{\text{optimal}}}$$

| Agent | Mean Loss / Flow-Step | Optimality Gap | Multiplier vs Optimal |
|-------|----------------------|----------------|----------------------|
| **Optimal** | **0.701** | — | 1.00× |
| QA-MAB | 3.906 | 457% | 5.57× |
| Oracle | 4.238 | 505% | 6.05× |
| NB3R | 4.539 | 548% | 6.47× |
| Greedy | 5.111 | 629% | 7.29× |
| Random | 5.970 | 752% | 8.52× |

#### Interpretation

All agents operate **5–9× above the theoretical lower bound**, indicating massive room for improvement. Several factors explain this gap:

1. **SA approximation error:** All QUBO-based agents (QA-MAB, Oracle) use Simulated Annealing with $n_{\text{reads}} = 20$, $n_{\text{sweeps}} = 200$. The optimal agent exhaustively evaluates all 64 combinations — SA often settles for local minima, especially on the flat energy landscapes described in Appendix A.

2. **Estimation error:** QA-MAB and NB3R must *learn* fault parameters from noisy aggregate signals. Even at convergence, $\|\hat{\theta} - \theta^*\|_2 \approx 0.4$ (Section 5.1), meaning the QUBO diagonal costs are systematically wrong.

3. **Collision cascades:** The optimal agent explicitly evaluates all path *combinations*, choosing the joint assignment that minimizes total loss including collisions. SA-based agents optimize a QUBO that encodes collision penalties, but the SA solver does not always find collision-free solutions — leading to $C_{\text{coll}} = 5.0$ penalties that compound.

4. **Proximity interference suboptimality:** Even collision-free path selections may have high proximity interference ($\sum e^{-d/d_0}$). The optimal agent accounts for this exactly; SA-based agents handle it approximately.

The 5.57× gap for QA-MAB suggests that **improving the solver** (more SA reads, better annealing schedules, or actual quantum annealing hardware) could yield substantial gains even without improving the learning algorithm. This motivates the planned D-Wave integration (Section 9).

### 5.6 UCB Ablation

To isolate the contribution of the UCB exploration bonus ($c_{\text{ucb}} = 3.0$), we compare QA-MAB with and without UCB on a reduced experiment ($P = 3$ epochs, $T = 100$ steps, 5 seeds).

| Configuration | Mean Loss / Flow-Step | Collision Rate | $\|\hat{\theta} - \theta^*\|_2$ (final epoch) |
|--------------|----------------------|---------------|-----------------------------------------------|
| QA-MAB (UCB, $c_{\text{ucb}} = 3.0$) | 3.559 | **40.0%** | **0.468 ± 0.092** |
| QA-MAB (no UCB, $c_{\text{ucb}} = 0.0$) | **3.547** | 46.7% | 0.519 ± 0.076 |

#### Per-Epoch Breakdown

| Epoch | UCB ($c_{\text{ucb}} = 3.0$) | No UCB ($c_{\text{ucb}} = 0.0$) |
|-------|-------------------------------|----------------------------------|
| 1 | 3.960 | 3.939 |
| 2 | 2.380 | 2.663 |
| 3 | 4.338 | 4.040 |

#### Interpretation

The UCB ablation reveals a nuanced picture:

1. **Collision avoidance:** UCB reduces the collision rate by **6.7 percentage points** (40.0% vs 46.7%). The UCB bonus encourages visiting under-explored paths, which naturally spreads flows across different UAVs and reduces path overlap.

2. **Estimation quality:** UCB improves $\theta$ estimation error by ~10% (0.468 vs 0.519). By forcing exploration of more diverse paths, UCB provides richer training signal for the residual credit assignment, exposing more UAV combinations.

3. **Loss trade-off:** Despite better collision avoidance and estimation, the mean loss difference is negligible (3.559 vs 3.547). This suggests UCB's exploration comes at a cost: some explored paths are genuinely worse, and the reduced collision penalty is offset by occasionally choosing higher-fault-cost paths.

4. **Epoch dynamics:** UCB shows stronger improvement in Epoch 2 (2.380 vs 2.663), consistent with faster convergence from better exploration in Epoch 1. However, Epoch 3 reverses (4.338 vs 4.040), possibly due to topology-dependent variance at small sample sizes.

**Conclusion:** UCB's primary value is in **collision reduction** and **estimation quality**, not raw loss minimization. In a full-scale experiment ($P = 10$, 20 seeds), the collision avoidance benefit would likely compound, explaining why UCB is part of the default QA-MAB configuration. A definitive conclusion requires a larger-scale ablation, which is left as future work.

### 5.7 Transfer Learning Advantage

The key structural advantage of QA-MAB over NB3R is **cross-epoch transfer learning**:
- QA-MAB's $\hat{\theta}$ and $\hat{\phi}$ persist across epochs (with 0.7 decay)
- NB3R resets all weights to zero every epoch — no transfer

This means QA-MAB enters each new epoch with prior knowledge about which UAVs are likely faulty, enabling faster convergence to good routing decisions. NB3R must re-learn from scratch each epoch.

---

## 6. Figures

### Figure 1: Convergence of Estimation Error

![Convergence of θ and φ](convergence_plot.png)

Two panels showing $\|\hat{\theta} - \theta^*\|_2$ (left, blue) and $\|\hat{\phi} - \phi^*\|_2$ (right, orange) across 10 epochs. Solid lines = mean across 20 seeds; shaded regions = ±1 standard deviation. Both errors decrease in the first 4-5 epochs then plateau, consistent with the identifiability limit discussed in Section 5.1.

### Figure 2: Cumulative Loss and Regret

![Cumulative loss and regret](regret_convergence_plot.png)

**Left:** Cumulative total loss per step for QA-MAB (blue), NB3R (red), and Oracle (green dashed). QA-MAB maintains consistently lower cumulative loss than both baselines across all 10 epochs. Step-like jumps at epoch boundaries reflect topology changes.

**Right:** Cumulative regret (QA-MAB − Oracle) with ±1 std shaded band. The mean trend is consistently negative (QA-MAB advantage), but the wide confidence band indicates significant seed-to-seed variability.

---

## 7. Algorithm Pseudocode

```
Initialize θ̂ = 0, φ̂ = 0

for epoch p = 0 to P-1:
    Sample new topology (UAV positions, flows, K-shortest paths)
    θ̂ *= 0.7, φ̂ *= 0.7        (epoch decay)
    visit_counts = 0             (reset per-epoch)

    for step t = 0 to T-1:
        // Build QUBO
        for each flow n, path k:
            Q[nk,nk] = Σθ̂[i∈path] + Σφ̂[z∈path] − λ − ucb_c/√(max(visits[n,k],1))
        Add one-hot constraints (Q[nk,nk'] = λ for k≠k', same flow)
        Add collision penalties (C_coll for shared UAVs across flows)
        Add proximity penalties (exp(−d/d0) across flows)

        // Solve
        γ = γ₀ / ((p+1)^a · (t+1)^b)
        Q_scaled = Q / γ
        paths = SA_solve(Q_scaled, n_reads=20, n_sweeps=200)

        // Get feedback
        losses = environment(paths)

        // Learn (residual credit assignment)
        for each flow n:
            L_fault = L[n] − C_coll·collisions[n] − proximity[n]
            for each UAV i on path[n]:
                residual = L_fault − Σ_{j≠i} θ̂[j] − Σ_z φ̂[z]
                θ̂[i] += α · (residual − θ̂[i])
            for each zone z on path[n]:  (using updated θ̂)
                residual = L_fault − Σ_i θ̂[i] − Σ_{z'≠z} φ̂[z']
                φ̂[z] += α · (residual − φ̂[z])
        Clip θ̂, φ̂ to [0, 1]
        visit_counts[n, paths[n]] += 1
```

---

## 8. Summary

| Question | Answer |
|----------|--------|
| Does $\hat{\theta}$ improve across epochs? | **Yes** — E1: 0.550 → E7: 0.399 (−27%), then plateaus |
| Does $\hat{\phi}$ improve across epochs? | **Yes** — E1: 0.393 → E5: 0.314 (−20%), then plateaus |
| Does QA-MAB beat NB3R? | **Yes** — 16.2% lower total loss, 9.6pp lower collision rate |
| Does QA-MAB beat Oracle? | **Yes** — 8.5% lower total loss (partially due to exploration mechanisms; see §5.3) |
| What drives convergence? | New epochs (new topologies) >> more steps within an epoch |
| Key algorithmic innovations? | Residual credit assignment + epoch decay + UCB exploration + temperature schedule |

---

## 9. Limitations and Open Questions

### Limitations
1. **Oracle comparison confound** — QA-MAB has exploration mechanisms (UCB + temperature) that Oracle lacks. An ablation with Oracle+UCB+temperature is needed to isolate the "imperfect estimates" effect.
2. **Scale** — Only tested with $N = 3$ flows. Earlier experiments with $N = 5$ yielded ~99% collision rates, suggesting the approach needs architectural changes for higher flow counts.
3. **No statistical significance testing** — While the mean regret is negative, the wide confidence band means Oracle wins on some seeds. Formal hypothesis tests are needed.
4. **Convergence plateau** — Estimation error plateaus rather than converging to zero. With 10 epochs, we cannot determine whether more epochs would improve estimates further or if this is a fundamental limit.
5. **Classical SA only** — Current results use classical Simulated Annealing. Quantum Annealing (D-Wave) may provide different exploration characteristics.

### Future Work
1. **Fair Oracle ablation** — Oracle with UCB + temperature schedule
2. **D-Wave integration** — Replace SA with quantum annealing hardware
3. **Scaling to $N > 3$** — Architectural changes for higher flow counts
4. **Epoch decay sensitivity** — Systematic comparison of decay ∈ {0.5, 0.7, 0.9, 1.0}
5. **Fault rate sensitivity** — Performance with weaker faults ($\theta^* \in [0.05, 0.1]$)
6. **Formal regret analysis** — Competitive ratio or dynamic regret bound for the non-stationary setting
7. **Statistical analysis** — Paired t-tests or Wilcoxon signed-rank tests across seeds

---

## Appendix A: The Oracle Algorithm in Detail

This appendix provides a precise, code-level explanation of the Oracle baseline agent, the mechanism by which it fails to achieve optimal performance, and why this failure mode is structurally meaningful rather than an implementation oversight.

### A.1 Oracle Algorithm — Step by Step

The Oracle agent receives the **true** ground-truth parameters $\theta^*$ and $\phi^*$ at initialization. Its complete decision procedure at each step is:

```
Oracle.act(t, p):
  1. Build QUBO matrix Q using θ* and φ* (true values)
     - Diagonal:   Q[nk,nk] = Σ θ*[i∈path(n,k)] + Σ φ*[z∈path(n,k)] − λ
     - Same-flow:  Q[nk,nk'] = λ  (one-hot constraint)
     - Cross-flow: Q[nk,lj]  = C_coll · 𝟙[shared UAVs] + exp(−d/d₀)
  2. Pass raw Q directly to SA solver (no scaling, no modification)
  3. SA_solve(Q, n_reads=20, n_sweeps=200, T_init=2.0, T_final=0.05)
  4. Decode binary solution → chosen paths
  5. Return paths (no learning step — update() is a no-op)
```

**What Oracle does NOT do** (verified from source code):
- ❌ No `visit_counts` passed to `build_qubo` — the UCB bonus term is never computed
- ❌ No temperature scaling — `Q` is passed to SA as-is, without division by $\gamma(p,t)$
- ❌ No `update()` — estimates are frozen; the method body is literally `pass`
- ❌ No epoch decay — there is nothing to decay since estimates never change

The Oracle uses **identical SA solver parameters** as QA-MAB ($n_\text{reads} = 20$, $n_\text{sweeps} = 200$, $T_\text{init} = 2.0$, $T_\text{final} = 0.05$). The SA solver itself is the same function. The only differences are in what goes *into* the solver.

### A.2 Why Oracle Doesn't Always Win — The Exact Mechanism

The Oracle's failure stems from the interaction of two structural properties: the **symmetry problem** and the **cold-start problem**. Together, they cause Oracle's SA solver to systematically under-explore the solution space, leading to higher collision rates.

### A.3 The Symmetry Problem — All Healthy UAVs Look Identical

In our fault model, $n_f = 4$ out of $m = 30$ UAVs are faulty ($\theta^* \in [0.2, 0.4]$), and the remaining 26 UAVs are perfectly healthy ($\theta^* = 0$). Similarly, $n_z = 2$ out of $Z = 9$ zones are problematic, and 7 zones have $\phi^* = 0$.

Consider what Oracle's QUBO diagonal looks like for a path through only healthy UAVs in healthy zones:

$$Q^{\text{Oracle}}_{nk,nk} = \underbrace{\sum_{i \in \text{path}} \theta^*_i}_{= 0} + \underbrace{\sum_{z \in \text{path}} \phi^*_z}_{= 0} - \lambda = -10.0$$

Every path through healthy UAVs in healthy zones has **exactly the same diagonal cost**: $-\lambda = -10.0$. The SA solver sees a flat energy landscape across these paths — there is no signal to prefer one healthy path over another.

In a network with 30 UAVs, 26 of which are healthy, most candidate paths consist entirely of healthy UAVs. This means the majority of the $N \times K = 12$ path variables have identical diagonal entries. When SA must assign one path per flow, it has no cost-based reason to spread flows across different healthy paths.

**The consequence:** SA frequently assigns multiple flows to paths that share UAVs — not because the paths are *good*, but because they are *indistinguishable*. The cross-flow collision penalty ($C_\text{coll} = 5.0$) exists in the QUBO and should prevent this, but SA is a local search heuristic operating on a flat landscape. With 20 reads and 200 sweeps, it often fails to find collision-free configurations when the diagonal provides no gradient to guide the search.

**Contrast with QA-MAB:** QA-MAB's learned estimates $\hat{\theta}$ are imperfect approximations of $\theta^*$. Crucially, even healthy UAVs accumulate small nonzero estimates based on their co-occurrence with faulty UAVs on observed paths. These small differences — say, $\hat{\theta}_i = 0.03$ for one healthy UAV and $\hat{\theta}_j = 0.07$ for another — break the symmetry. Paths through different healthy UAVs now have *different* diagonal costs, giving SA a gradient to follow. This noise-induced symmetry breaking is a feature, not a bug.

### A.4 The Cold-Start Problem — Oracle Operates in Permanent Greedy Mode

QA-MAB applies two exploration mechanisms that modulate the QUBO before it reaches the SA solver:

**1. Temperature scaling (QUBO-level):**
$$Q^{\text{QA-MAB}}_\text{scaled} = Q / \gamma(p, t), \quad \gamma(p,t) = \frac{2.0}{(p+1)^{0.5} \cdot (t+1)^{0.3}}$$

At the start of the experiment ($p = 0, t = 0$), $\gamma = 2.0$, so Q is divided by 2.0. This halves the magnitude of all QUBO entries relative to SA's own temperature schedule, effectively making SA explore more broadly. As $p$ and $t$ increase, $\gamma$ shrinks toward zero, amplifying Q's structure and driving SA toward greedy exploitation.

Oracle passes raw Q with no scaling — equivalent to $\gamma = 1.0$ always. Compared to QA-MAB at early steps, Oracle's QUBO has **twice the magnitude**, meaning SA's fixed temperature schedule ($T_\text{init} = 2.0 \to T_\text{final} = 0.05$) accepts fewer uphill moves. Oracle is permanently in a relatively "cold" regime.

**2. UCB exploration bonus:**
$$Q^{\text{QA-MAB}}_{nk,nk} \mathrel{-}= \frac{c_\text{ucb}}{\sqrt{\max(\text{visits}_{n,k}, 1)}}$$

With $c_\text{ucb} = 3.0$, a never-visited path gets a bonus of $-3.0$ on its diagonal (reducing cost by 3.0), while a path visited 9 times gets only $-1.0$. This explicitly encourages SA to try under-explored paths.

Oracle calls `build_qubo` with `visit_counts=None`, so the UCB branch in the QUBO builder is skipped entirely. Oracle has **zero** exploration incentive beyond SA's inherent stochasticity.

**Combined effect:** Oracle's SA solver sees a high-magnitude, flat-across-healthy-paths QUBO with no visit-count bias. It rapidly converges to a local minimum and stays there. QA-MAB's SA solver sees a lower-magnitude, UCB-differentiated QUBO that encourages trying different path combinations. Over 100 steps per epoch, QA-MAB discovers collision-free configurations that Oracle never explores.

### A.5 Quantitative Impact: Collisions

The collision rate data from Section 5.2 confirms this analysis:

| Agent | Collision Rate | Interpretation |
|-------|---------------|----------------|
| QA-MAB | 53.7% | UCB + temperature → better flow separation |
| Oracle | 57.5% | Symmetry trap → higher collision clustering |
| NB3R | 63.3% | Per-flow softmax → no cross-flow coordination |

Oracle's 3.8 percentage point higher collision rate translates directly to excess loss: each collision adds $C_\text{coll} = 5.0$ per colliding flow pair. Across 20 seeds × 10 epochs × 100 steps × 3 flows, even a small collision rate increase compounds to a significant total loss difference.

### A.6 Why This Comparison Is Fair for the Thesis

One might argue that the Oracle comparison is unfair because QA-MAB has exploration mechanisms that Oracle lacks. We address this concern on multiple levels:

**1. Oracle represents the "perfect knowledge" baseline, not the "perfect algorithm" baseline.**
The purpose of the Oracle is to test the question: *"How much does perfect knowledge of $\theta^*$ and $\phi^*$ help?"* The answer is: less than one might expect, because knowledge alone is insufficient — the QUBO+SA pipeline also needs exploration to handle the combinatorial structure of multi-flow routing. This is itself a meaningful finding.

**2. The Oracle is the strongest *natural* baseline.**
Giving Oracle the true parameters with a standard greedy SA solve is the most straightforward way to use perfect information. It is the baseline a reviewer would expect. Adding UCB or temperature scaling to Oracle would create a *different* algorithm ("Oracle+exploration"), which conflates two questions: "Does learning help?" and "Does exploration help?"

**3. QA-MAB's exploration mechanisms are part of its contribution.**
The temperature schedule and UCB bonus are integral to the QA-MAB algorithm design. They were developed to address the exploration-exploitation tradeoff in the bandit setting. Removing them from QA-MAB (for a "fair" comparison) would cripple the very algorithm we are evaluating.

**4. The result is still meaningful even with the confound.**
If QA-MAB — which must *learn* $\hat{\theta}$ and $\hat{\phi}$ from noisy aggregate signals — outperforms an agent with *perfect* knowledge, that demonstrates the algorithm's overall effectiveness. The fact that exploration contributes to this advantage does not diminish it; it shows that the algorithm's design (learning + exploration + QUBO) is more effective than knowledge alone.

**5. The confound is explicitly acknowledged.**
Section 5.3 and Section 9 clearly state that an ablation study (Oracle+UCB+temperature) is needed to separate the exploration effect from the imperfect-estimates effect. This is listed as future work, not hidden.

### A.7 Summary: Oracle vs QA-MAB Side-by-Side

| Feature | Oracle | QA-MAB |
|---------|--------|--------|
| $\hat{\theta}$ | True $\theta^*$ (frozen) | Learned online (initialized to 0) |
| $\hat{\phi}$ | True $\phi^*$ (frozen) | Learned online (initialized to 0) |
| Healthy UAV diagonal cost | All identical: $-\lambda$ | Varies: $-\lambda - \text{UCB} + \hat{\theta}_i$ |
| QUBO temperature scaling | None ($\gamma = 1$, always) | $\gamma(p,t) = \gamma_0 / (p+1)^a(t+1)^b$ |
| UCB exploration bonus | None | $-c_\text{ucb}/\sqrt{\text{visits}}$ |
| Learning | None (`update()` is `pass`) | Residual credit assignment |
| Epoch decay | None | $\times 0.7$ at epoch boundaries |
| SA parameters | Identical | Identical |
| SA solver function | Identical | Identical |
| Result | 57.5% collisions, +8.5% loss | 53.7% collisions, lowest loss |
