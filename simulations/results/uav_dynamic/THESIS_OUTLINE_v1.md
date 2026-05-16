# Thesis Outline v1: Quantum-Assisted Multi-Armed Bandits for Dynamic UAV Routing with Faulty Agents

**Author:** Yehonatan Carmeli  
**Program:** M.Sc. Electrical & Computer Engineering, Ben-Gurion University  
**Version:** 1.0 — May 2026  
**Status:** Structural blueprint — content to be filled per section

---

## Overview

This thesis investigates the QA-MAB (Quantum-Assisted Multi-Armed Bandit) framework for dynamic UAV flow routing in networks with hidden agent failures. The central discovery is a paradox: a theoretically optimal solver (SA with Oracle knowledge) collapses due to symmetry in the QUBO cost landscape, while an imperfect learner (MAB with noisy parameter estimates) breaks that symmetry and achieves superior routing performance. This finding motivates quantum annealing as a principled solution — a solver that can tunnel through symmetric energy barriers rather than relying on accidental noise.

The thesis is structured in four chapters, each building on the previous:

1. **Model & SA Limits** — Formulate the problem, prove SA solves the QUBO correctly, then show why it fails in practice.
2. **Learning Without Decay** — Analyze the MAB learning algorithm and prove convergence of parameter estimates.
3. **Theory vs Reality** — Formalize the symmetry-breaking paradox and explain why imperfect knowledge outperforms perfect knowledge.
4. **The Quantum Advantage** — Argue that quantum annealing is architecturally suited to resolve the symmetry problem.

---

# Chapter 1: Model Building and SA Limits

**Subtitle:** *"When Simulated Annealing Solves the QUBO — and When It Collapses"*

**Chapter goal:** Establish the problem, derive the QUBO formulation, demonstrate that SA is a competent QUBO solver, and then reveal the symmetry failure that makes Oracle-SA unusable in practice.

---

## 1.1 Problem Definition: Dynamic UAV Routing with Hidden Failures

**What goes here:** Formal definition of the network routing environment. A directed graph G = (V, E) with K UAV agents that must route flows from source s to sink t across P epochs. Each epoch draws a new random topology. A subset of edges have hidden failure probabilities θ*[i] (edge reliability) and φ*[z] (zone reliability) that are unknown to the controller. The controller observes only end-to-end flow outcomes (success/failure), not which specific edge or zone caused a failure.

**Key definitions to include:**
- Network graph G = (V, E), flow paths P = {p₁, ..., p_M}, K agents
- Hidden parameters: θ* ∈ [0,1]^|E| (edge failure rates), φ* ∈ [0,1]^|Z| (zone failure rates)
- Epoch structure: each epoch e ∈ {1, ..., P} samples a new topology G_e
- Observation model: binary outcome per flow (success = all edges on path healthy)
- Objective: maximize total successful deliveries across all epochs

**Connects to:** This section sets up the notation used throughout. The "hidden failures" aspect is what makes the problem a bandit — the agent cannot directly observe failure causes.

---

## 1.2 Temporal Decoupling: Epoch Length as a Design Parameter

**What goes here:** A critical practical simplification in our model. The real UAV routing problem is continuous — UAVs move, communicate, and make decisions asynchronously. We decouple time into discrete decision epochs of length T, where T is set by two competing factors: (a) the average time for a flow to traverse the topology (traversal time), and (b) the time required to collect loss observations and solve the QUBO (sensing + computation time).

**Key insight:** T is not arbitrary — it is a design parameter that reflects the sensing-to-solver latency of the system. If T is too small, the QUBO is solved before sufficient observations are gathered → poor estimates. If T is too large, the system is sluggish and can't adapt to topology changes.

**Key equations to include:**
```
T = max(t_traverse, t_sense + t_qubo_solve)
t_qubo_solve ≈ O(n_reads × n_sweeps × K^N)
```

**Simplifications justified:** In our simulations, T is treated as a fixed parameter per epoch. In reality, T varies with network conditions, QUBO solve time, and communication delays. This simplification converts a continuous real-time problem into a batch-optimization problem, which is standard in the UAV routing literature (cite). We acknowledge this as a modeling choice, not a fundamental limitation.

**Connection to Chapter 2:** The choice of T directly affects how many observations are available per epoch for credit assignment. Larger T → more observations → better θ̂ updates → faster convergence.

---

## 1.3 QUBO Formulation

**What goes here:** Derivation of the Quadratic Unconstrained Binary Optimization formulation that maps the UAV routing decision to a form solvable by SA or quantum annealing. Each binary variable x_{nk} ∈ {0,1} represents "agent k is assigned to path n." The QUBO matrix Q encodes both the estimated path quality (diagonal) and collision penalties (off-diagonal).

**Key equations:**

- **Decision variables:** x_{nk} = 1 if agent k routes through path n
- **QUBO diagonal (path quality):** `Q(nk, nk) = −[Σ_{i∈path_n} θ̂[i] + Σ_{z∈zones(path_n)} φ̂[z]] + λ`
  - Negative because QUBO minimizes; better paths have lower diagonal cost
  - λ is the Lagrange multiplier enforcing one-path-per-agent constraints
- **QUBO off-diagonal (collision penalty):** `Q(nk, n'k') = +γ · overlap(path_n, path_n')` for k ≠ k'
  - Penalizes assigning multiple agents to overlapping paths
  - γ controls collision avoidance strength
- **Constraint terms:** Penalty for assigning agent k to more than one path, penalty for leaving an agent unassigned

**Connects to:** This is the "correct" QUBO — validated by the SA optimality test in §1.4. The diagonal structure is critical: when θ̂ = θ* (Oracle), healthy paths have identical diagonal costs, which is the root cause of the symmetry problem (§1.5–1.6).

**Finding anchored here:** The QUBO formulation itself is sound. The problem is not in the encoding but in the solver's response to symmetric cost landscapes.

---

## 1.4 SA as QUBO Solver: The Optimality Test

**What goes here:** Empirical validation that Simulated Annealing correctly solves the QUBO as formulated. Present the 30-seed experiment: for each of 30 random topologies, run SA on the QUBO with known (Oracle) parameters and compare the SA solution to the true optimum (found by brute-force enumeration for small instances or branch-and-bound for larger ones).

**Key results to present:**
- SA achieves **90% optimality rate** across 30 seeds (finds the global minimum in 27/30 cases)
- Average **optimality gap = 0.0014** (SA solution value vs true minimum)
- SA runtime and convergence characteristics (cooling schedule, acceptance rates)

**What this proves:** SA is a competent solver for this QUBO size. The failures we observe in practice (Chapter 3) are NOT due to SA failing to solve the optimization — they arise from the structure of the cost landscape itself.

**Methodology notes:** Document the SA hyperparameters (initial temperature, cooling rate, number of sweeps), the brute-force verification procedure, and the gap metric definition: `gap = (E_SA − E_opt) / |E_opt|`.

**Finding anchored here:** "SA solves the QUBO correctly" — this is the baseline that makes the symmetry failure surprising.

---

## 1.5 The Symmetry Problem: Why Oracle with True θ* Fails

**What goes here:** The central negative result of Chapter 1. When the controller has perfect knowledge (Oracle: θ̂ = θ*, φ̂ = φ*), all healthy paths have identical or near-identical diagonal costs in the QUBO. SA finds a valid minimum, but because many configurations are equally optimal, the chosen solution is essentially arbitrary. With K agents making independent arbitrary choices among equivalent optima, they **cluster on the same paths**, causing collisions and catastrophic throughput loss.

**Key argument structure:**
1. With true parameters, the diagonal cost of path n is `−[Σ_{i∈path_n} θ*[i] + Σ_{z∈zones(n)} φ*[z]] + λ`
2. For healthy paths (all edges reliable, θ*[i] ≈ 0), these sums are nearly identical
3. The QUBO has exponentially many degenerate global minima (any permutation of agents across equally-good paths)
4. SA breaks ties arbitrarily (by thermal noise at final temperature) → agents cluster
5. Clustering → edge contention → collisions → throughput collapse

**Include:** A concrete numerical example showing two paths with identical diagonal cost, and the resulting degenerate solution space. Visualize the "flat valley" in the energy landscape.

**Finding anchored here:** "SA fails in practice due to symmetry" — Oracle performance is paradoxically worse than the learner.

---

## 1.5 Formalizing the Symmetry Failure: Degenerate Optima and Equal Diagonal Costs

**What goes here:** Mathematical formalization of the symmetry problem identified in §1.4. Define the degeneracy formally: count the number of equivalent global minima as a function of the number of healthy paths M_h and agents K.

**Key formal content:**

- **Definition (QUBO Degeneracy):** A QUBO instance has degeneracy D if there exist D distinct binary vectors x that all achieve the global minimum energy E_opt.
- **Theorem (Oracle Degeneracy):** Under Oracle parameters with M_h healthy paths of equal reliability and K ≤ M_h agents, the QUBO has degeneracy `D = M_h! / (M_h − K)!` — the number of injective assignments of K agents to M_h equivalent paths.
- **Corollary:** As M_h grows, D grows super-exponentially, and the probability that SA independently finds a collision-free assignment approaches `K! / M_h^K` → 0.

**Key equation:** Diagonal cost equality condition:
```
∀ paths n, n' ∈ healthy set:  Q(nk, nk) = Q(n'k, n'k)  ⟺  Σ_{i∈n} θ*[i] = Σ_{i∈n'} θ*[i]
```

**Include:** Proof sketch for the degeneracy count. Discussion of how off-diagonal collision penalties partially break symmetry but are insufficient when the number of paths exceeds agents.

---

## 1.6 Conditions Under Which SA Works vs Fails

**What goes here:** A taxonomy of problem instances classified by whether SA produces good routing solutions. This section synthesizes §1.4–1.5 into actionable criteria.

**Classification:**
| Condition | SA Outcome | Reason |
|-----------|-----------|--------|
| Few paths, many faulty → clear best path | ✅ Works | Diagonal costs well-separated |
| Many healthy paths, K agents | ❌ Fails | Degenerate optima, clustering |
| Noisy θ̂ (imperfect estimates) | ✅ Works (paradoxically) | Noise breaks degeneracy |
| High collision penalty γ | ⚠️ Partial | Helps but doesn't fully resolve |

**Key insight to develop:** SA's failure is not a bug in SA — it is a structural property of the QUBO when the cost landscape is symmetric. This motivates two responses: (a) inject asymmetry artificially (Chapter 3), or (b) use a solver that naturally handles degenerate minima (Chapter 4).

**Connects to:** Chapters 3 and 4 directly. This section is the "bridge" that motivates the rest of the thesis.

---

# Chapter 2: Learning Without Decay

**Subtitle:** *"MAB Convergence with Persistent Parameter Estimates in Piecewise-Stationary Environments"*

**Chapter goal:** Establish the learning algorithm (residual credit assignment), prove that parameter estimates converge to true values when epoch_decay = 1.0, show empirical convergence, and analyze why any decay < 1.0 destroys convergence.

---

## 2.1 Environment Model: Piecewise-Stationary Bandits

**What goes here:** Formal definition of the environment as a piecewise-stationary bandit. Each epoch e draws a new topology G_e (non-stationary structure), but the underlying failure parameters θ* and φ* remain constant across all epochs (stationary parameters). This is the key modeling assumption: the network's physical reliability doesn't change — only the available paths change.

**Key definitions:**
- **Epoch:** A period during which the topology G_e is fixed and T_e rounds of routing occur
- **Piecewise-stationarity:** G_e changes between epochs; θ*, φ* are constant ∀e
- **Arm:** A path assignment (which path each agent takes); arm set changes each epoch
- **Reward:** Number of successful deliveries in a round

**Distinction from standard MAB:** In classical non-stationary bandits, the reward distribution changes. Here, the *arm set* changes (new topology) but the *underlying parameters generating rewards* are fixed. This is why persistent memory (no decay) works — there's something stable to learn.

**Connects to:** §2.3–2.4 depend critically on this stationarity assumption for θ*, φ*.

---

## 2.2 Residual Credit Assignment: The Learning Algorithm

**What goes here:** Detailed description of the algorithm that updates edge-level estimates θ̂[i] and zone-level estimates φ̂[z] from end-to-end flow observations. Since the controller only observes whether an entire flow succeeded or failed (not which edge caused the failure), credit must be assigned to individual components.

**Algorithm pseudocode to include:**
```
After each round t in epoch e:
  For each flow f assigned to path p:
    outcome = observe(f)  ∈ {0, 1}
    predicted = Π_{i∈p} (1 − θ̂[i]) · Π_{z∈zones(p)} (1 − φ̂[z])
    residual = outcome − predicted
    For each edge i ∈ p:
      θ̂[i] ← θ̂[i] − α · residual · ∂predicted/∂θ̂[i]
    For each zone z ∈ zones(p):
      φ̂[z] ← φ̂[z] − α · residual · ∂predicted/∂φ̂[z]
```

**Key aspects to discuss:**
- The gradient computation through the product of (1 − θ̂[i]) terms
- Learning rate α and its interaction with epoch_decay
- Initialization of θ̂, φ̂ (uniform priors, e.g., 0.5)
- Clipping/projection to keep estimates in [0,1]

**Connects to:** This is the algorithm whose convergence is analyzed in §2.4. The gradient structure through the product is what makes convergence possible despite partial observability.

---

## 2.3 Why Epoch Decay Destroys Learning

**What goes here:** Mathematical analysis of what happens when epoch_decay < 1.0. Between epochs, the update rule applies: `θ̂[i] ← epoch_decay · θ̂[i] + (1 − epoch_decay) · θ̂_prior`. With decay < 1.0, this pulls estimates back toward the prior after every epoch, erasing learned information.

**Key analysis:**

- **With decay = 0.8:** After P epochs, the effective memory of observations from epoch 1 is scaled by 0.8^(P−1). For P=50, this is 0.8^49 ≈ 1.4×10⁻⁵ — effectively zero. The learner has amnesia.
- **With decay = 1.0:** No forgetting. All observations from all epochs contribute equally to the current estimate. Information accumulates monotonically.
- **The intuition:** Epoch decay was designed for non-stationary *parameters* (tracking a moving target). But θ* is stationary — decay solves a problem that doesn't exist and creates one that does.

**Formal statement:** 

> **Proposition (Decay–Convergence Tradeoff):** For epoch_decay = d < 1, the steady-state estimation error satisfies `E[||θ̂ − θ*||²] ≥ C · (1−d)²` for a constant C > 0 depending on the prior distance. Only d = 1 allows E[||θ̂ − θ*||²] → 0.

**Finding anchored here:** "epoch_decay=1.0 is optimal" — this is the theoretical explanation for why.

---

## 2.4 Convergence Proof Sketch: θ̂ → θ* as P → ∞ with Decay = 1.0

**What goes here:** The main theoretical result of Chapter 2. A convergence proof (or rigorous proof sketch) showing that with epoch_decay = 1.0 and sufficient topological diversity across epochs, the parameter estimates converge to the true values.

**Proof structure:**
1. **Sufficient excitation:** Each edge i must appear in at least one path in infinitely many epochs (topological diversity ensures this w.h.p. for random graph generation)
2. **Gradient consistency:** Show that E[residual · ∂predicted/∂θ̂[i]] has the correct sign — it pushes θ̂[i] toward θ*[i]
3. **Robbins-Monro conditions:** With appropriate learning rate schedule (or constant α with averaging), the stochastic approximation converges
4. **Product structure:** Handle the coupling between θ̂ and φ̂ estimates through the product in the predicted probability

**Theorem statement (to be proven rigorously):**

> **Theorem (Parameter Convergence):** Under Assumptions A1 (stationarity of θ*, φ*), A2 (sufficient topological diversity), and A3 (learning rate satisfying Σα_t = ∞, Σα_t² < ∞), with epoch_decay = 1.0:
> `θ̂[i] →_{a.s.} θ*[i]  and  φ̂[z] →_{a.s.} φ*[z]  as P → ∞`

**Connects to:** This convergence result is what makes Chapter 3's paradox sharp — the learner converges to Oracle, yet performs *better* during convergence than at convergence.

---

## 2.5 Empirical Convergence: The P=50 Result

**What goes here:** Present the empirical convergence data from our simulations. This section validates the theoretical prediction of §2.4 with concrete numbers.

**Key results:**
- **θ̂ error trajectory:** Starts at 0.532 (random initialization far from truth), decreases monotonically to 0.118 over P=50 epochs
- **φ̂ error trajectory:** Starts at 0.411, decreases to 0.082 over P=50 epochs
- **Convergence rate:** Approximately O(1/√P) — consistent with stochastic approximation theory
- **Epoch-by-epoch plots:** Error vs epoch number, showing smooth decrease with some variance

**Include:**
- Convergence plots (θ_err vs epoch, φ_err vs epoch) averaged over 30 seeds with confidence bands
- Comparison across different epoch_decay values (d=0.5, 0.8, 0.9, 1.0) showing only d=1.0 achieves sustained convergence
- Table of final errors at P=50 for each decay value

**Finding anchored here:** "θ̂ converges from 0.532 → 0.118 over P=50 epochs" — concrete validation of theory.

---

## 2.6 The Identifiability Question: Can We Reach θ̂ = 0 Exactly?

**What goes here:** A theoretical analysis of the fundamental limits of learning in this partial-observation model. Even with infinite data, can the controller perfectly identify which edges are faulty and which are healthy?

**Key issues:**
- **Aliasing:** If two edges always appear together in every path, their individual failure rates are not separately identifiable — only their combined effect is observable
- **Product ambiguity:** Observing P(success on path n) = Π(1 − θ*[i]) determines the product but not individual factors without sufficient path diversity
- **Zone-edge confounding:** Zone failures and edge failures on the same path create additional identification challenges

**Formal question:** Under what topological conditions is (θ*, φ*) uniquely identifiable from flow observations?

> **Conjecture (Identifiability):** (θ*, φ*) is identifiable if and only if the binary matrix A (where A_{ni} = 1 iff edge i is in path n) has full column rank over the reals — equivalently, every edge appears in a unique combination of paths.

**Why this matters for the thesis:** Even if θ̂ cannot reach θ* exactly, the *residual error* is what breaks symmetry (Chapter 3). So identifiability limits may actually be beneficial for routing performance — a deeper instance of the paradox.

---

# Chapter 3: Theory vs Reality

**Subtitle:** *"Perfect Solver vs Useful Error — The Symmetry Breaking Discovery"*

**Chapter goal:** Formalize the central paradox: a perfect solver with perfect knowledge produces worse routing than an imperfect solver with noisy estimates. Explain the mechanism (symmetry breaking through estimation error) and explore its implications.

---

## 3.1 Theoretical Framework: Would Perfect SA + Perfect Knowledge Be Optimal?

**What goes here:** Set up the theoretical baseline. If we had a perfect QUBO solver (always finds global minimum) and perfect parameter knowledge (θ̂ = θ*), would QA-MAB achieve optimal routing? The answer, surprisingly, is no — and this section explains why by connecting Chapter 1's symmetry analysis to routing performance.

**Formal argument:**
1. With θ̂ = θ*, the QUBO has degenerate minima (from §1.5)
2. A perfect solver returns *some* global minimum, but cannot distinguish between D equivalent solutions
3. Without a tie-breaking mechanism, the returned solution has no preference for collision-free assignments
4. Expected number of collisions scales as K²/M_h (birthday paradox analogue)
5. Therefore, perfect solver + perfect knowledge ≠ optimal routing

**Key definition:**

> **Definition (Routing Optimality Gap):** Let R_opt be the maximum expected throughput (achieved by a centralized scheduler with full knowledge and collision avoidance). The routing gap is `Δ_R = R_opt − R_actual`.

**Show that:** Oracle-SA has Δ_R > 0 due to collisions, even though its QUBO gap ≈ 0.

---

## 3.2 The Symmetry Breaking Mechanism

**What goes here:** The core explanatory contribution of the thesis. When θ̂ ≠ θ* (imperfect estimates), the diagonal costs of healthy paths are no longer identical. Each path gets a slightly different cost based on the estimation noise:

```
Q(nk, nk) = −[Σ_{i∈n} θ̂[i] + Σ_{z∈zones(n)} φ̂[z]] + λ
```

Because θ̂[i] varies randomly around θ*[i], different paths accumulate different total errors, creating a non-degenerate cost landscape that SA can navigate meaningfully.

**Key mechanism to formalize:**
1. **Error as signal:** θ̂[i] = θ*[i] + ε[i], where ε[i] is the estimation error
2. **Diagonal cost differentiation:** Path n's cost becomes `−Σ_{i∈n}(θ*[i] + ε[i]) − Σ_z(φ*[z] + δ[z]) + λ`
3. **The ε terms break degeneracy:** Different paths have different Σε[i], creating a unique ranking
4. **SA finds the "least noisy" path:** The path with smallest accumulated error gets lowest cost → preferred
5. **Diversity emerges:** Because different agents have different SA runs with different thermal noise, they land on different non-degenerate minima → reduced clustering

**Analogy to include:** This is analogous to symmetry breaking in physics — a perfectly symmetric potential has no preferred direction, but any perturbation (even infinitesimal) selects a specific ground state.

**Finding anchored here:** "Imperfect θ̂ creates useful asymmetry."

---

## 3.3 The Paradox Formally: Better Knowledge → Worse Performance

**What goes here:** State and prove the main paradox as a formal result. This is the thesis's headline theorem.

> **Theorem (Knowledge-Performance Paradox):** Consider the QA-MAB system with K ≥ 2 agents and M_h ≥ K healthy paths of equal true reliability. Let R(θ̂) denote expected throughput when using estimates θ̂ in the QUBO. Then:
> `R(θ̂ = θ* + ε) > R(θ̂ = θ*)  for suitable ||ε|| > 0`
> That is, imperfect knowledge strictly outperforms perfect knowledge.

**Proof approach:**
1. R(θ*) suffers from collision rate C_oracle ∝ K²/M_h (from §3.1)
2. R(θ* + ε) has collision rate C_noisy < C_oracle because noise diversifies SA solutions
3. The throughput loss from suboptimal path selection (choosing slightly worse paths due to noise) is second-order in ||ε||
4. The throughput gain from collision avoidance is first-order in the collision reduction
5. For small enough ||ε||, gain dominates loss → R(θ* + ε) > R(θ*)

**Include:** The critical ||ε|| range — too much noise selects genuinely bad paths; too little noise doesn't break symmetry sufficiently. There exists an optimal noise level.

**Finding anchored here:** "The paradox: learning error outperforms perfect knowledge."

---

## 3.4 Noise as Feature: How Learning Noise Prevents Paralysis

**What goes here:** Deeper exploration of the noise mechanism from §3.2, connecting it to well-known concepts in optimization and machine learning.

**Connections to draw:**
- **Simulated annealing itself:** SA adds thermal noise to escape local minima. Here, the *problem input* has noise that prevents the solver from getting trapped in degenerate regions.
- **Dropout / regularization:** In deep learning, noise during training prevents overfitting. Here, noise in parameter estimates prevents "overfitting" to the symmetric solution.
- **Exploration-exploitation in MAB:** The estimation error provides a natural exploration bonus — paths with high uncertainty get varied costs across runs, ensuring diverse assignments.
- **Thompson Sampling analogy:** Thompson Sampling samples from the posterior over arm parameters, naturally creating the kind of noise that breaks ties. Our system achieves a similar effect through learning error rather than deliberate posterior sampling.

**Key insight to formalize:** The estimation error acts as an implicit exploration mechanism. As θ̂ → θ* (convergence), this exploration signal vanishes, and performance should eventually degrade — creating a tension between learning (wants convergence) and routing (wants noise). Discuss whether this is observed in our P=50 data.

---

## 3.5 Ablation Results: Effect of Epoch Decay Values

**What goes here:** Systematic experimental comparison across epoch_decay values {0.5, 0.8, 0.9, 0.95, 1.0}. For each value, report: (a) final θ̂ error, (b) convergence trajectory, (c) routing throughput, (d) collision rate.

**Expected table structure:**

| decay | θ_err (final) | φ_err (final) | Throughput | Collisions | Interpretation |
|-------|--------------|--------------|------------|------------|----------------|
| 0.5   | high         | high         | medium     | medium     | Too much noise, poor paths |
| 0.8   | medium       | medium       | medium     | medium     | Partial learning, partial noise |
| 1.0   | 0.118        | 0.082        | highest    | lowest     | Full convergence, optimal noise trajectory |

**Key analysis:** With decay=1.0, the system traverses the optimal noise trajectory — starting with high noise (good for symmetry breaking) and gradually reducing noise (good for path quality) as estimates converge. Decay < 1.0 prevents convergence, locking the system at a fixed noise level that's suboptimal for later epochs.

**Finding anchored here:** "epoch_decay=1.0 (no decay) is optimal" — explained by the noise trajectory argument.

---

## 3.6 Open Question: Designing a Solver That Breaks Symmetry Without Error

**What goes here:** Forward-looking theoretical discussion. If noise is beneficial but convergence eliminates it, can we design a system that maintains symmetry-breaking capability even with perfect knowledge?

**Approaches to discuss:**
1. **Explicit tie-breaking:** Add a small random perturbation to the QUBO diagonal when degeneracy is detected
2. **Anti-collision regularization:** Modify the QUBO to include a diversity term that penalizes solutions too similar to recently-chosen ones
3. **Multi-start SA:** Run SA from multiple random initializations and select diverse solutions
4. **Quantum annealing (preview of Chapter 4):** QA naturally samples from the degenerate ground state manifold with quantum fluctuations providing the symmetry breaking

**Connects to:** Chapter 4, where quantum annealing is proposed as the principled solution. This section frames QA as answering a well-defined theoretical need (not just "use quantum because it's cool").

---

# Chapter 4: The Quantum Advantage

**Subtitle:** *"Why Quantum Annealing Is Purpose-Built for the Symmetry Problem"*

**Chapter goal:** Argue that quantum annealing addresses the specific failure mode identified in Chapters 1 and 3. QA's ability to tunnel through energy barriers and sample from degenerate ground states makes it architecturally suited to the symmetry problem. Present the integration plan and expected improvements.

---

## 4.1 SA's Fundamental Limit: Thermal vs Quantum Exploration

**What goes here:** Analysis of why classical SA struggles with degenerate minima from a computational physics perspective. SA explores the energy landscape via thermal fluctuations (Boltzmann acceptance: P(ΔE) = exp(−ΔE/T)). When the landscape is flat (degenerate minima), thermal fluctuations provide no directional guidance — all moves in the degenerate valley are equally likely.

**Key concepts:**
- **Thermal exploration:** Random walk in the energy landscape, biased by gradient. In flat regions, the walk is unbiased → diffusive → slow mixing between equivalent solutions.
- **Energy barriers between degenerate minima:** Even when multiple optima have equal energy, they may be separated by high-energy barriers. SA must thermally climb over these barriers, which becomes exponentially unlikely at low temperature.
- **The freezing problem:** As SA cools, it commits to whichever degenerate minimum it happens to be near — with no mechanism to prefer one over another based on routing quality (which is not encoded in the QUBO energy).

**Formal statement:**

> **Proposition (SA Mixing Time in Degenerate Landscapes):** For a QUBO with D degenerate global minima separated by energy barriers of height B, the mixing time of SA at temperature T satisfies `τ_mix ≥ D · exp(B/T)`. At the final SA temperature, τ_mix is exponentially large.

---

## 4.2 Quantum Tunneling Through Energy Barriers

**What goes here:** Introduction of quantum annealing and its key advantage: quantum tunneling. Unlike thermal exploration, which must climb over energy barriers, QA can tunnel through them by maintaining quantum superposition during the annealing process.

**Key physics:**
- **Transverse field:** QA applies a transverse magnetic field that creates quantum fluctuations, enabling the system to exist in superposition of multiple classical states simultaneously
- **Tunneling rate:** The tunneling probability through a barrier of width w and height B scales as `P_tunnel ∝ exp(−w√B)` — depends on width, not just height (unlike thermal: exp(−B/T))
- **Thin barriers in QUBO:** The barriers between degenerate minima in our QUBO are typically *thin* (differ by flipping a few bits = reassigning a few agents) even if *tall* (violating constraints temporarily costs high energy). QA exploits thin barriers.

**Why this matters for our problem:** The degenerate minima in the Oracle QUBO correspond to different permutations of agents across equivalent paths. These are separated by thin barriers (swap two agents = flip O(1) bits) with potentially tall constraint-violation peaks. QA tunnels through these efficiently; SA cannot.

---

## 4.3 The Hybrid Approach: QA for Global Structure, SA for Fine-Tuning

**What goes here:** Propose a hybrid quantum-classical algorithm. Use quantum annealing to find diverse solutions in the degenerate landscape (global structure), then use classical SA to refine each candidate solution (local optimization).

**Hybrid algorithm outline:**
1. **QA phase:** Submit QUBO to D-Wave. Collect N_reads samples from the quantum annealer. Due to quantum fluctuations, these samples naturally spread across the degenerate manifold.
2. **Diversity extraction:** From N_reads samples, select K diverse solutions (one per agent) using a maximum-diversity selection algorithm.
3. **Classical refinement:** For each selected solution, run short SA to optimize within its local basin.
4. **Assignment:** Each agent adopts its corresponding refined solution as its routing decision.

**Why hybrid beats pure QA:** Pure QA on current hardware (D-Wave) has limited precision and connectivity. By using QA for the hard part (symmetry breaking / diverse sampling) and SA for the easy part (local refinement), we get the best of both worlds.

**Include:** Complexity comparison: classical SA alone vs hybrid QA+SA, in terms of solution diversity and routing throughput.

---

## 4.4 D-Wave Integration Plan

**What goes here:** Practical details of connecting our QA-MAB framework to D-Wave quantum hardware. Reference the existing `dwave_setup.py` code in our repository.

**Implementation details:**
- **D-Wave API:** Ocean SDK, `DWaveSampler` and `EmbeddingComposite` for automatic minor embedding
- **QUBO submission:** Convert our Q matrix to D-Wave's BQM (Binary Quadratic Model) format
- **Embedding challenges:** Our QUBO has all-to-all connectivity (collision penalties between any pair of agents); D-Wave's Pegasus topology requires embedding. Discuss chain length and chain break frequency.
- **Hyperparameters:** `num_reads` (number of samples), `annealing_time` (microseconds), `chain_strength`
- **Hybrid solvers:** D-Wave's `LeapHybridSampler` for problems too large for direct embedding

**Code reference:** `dwave_setup.py` contains the boilerplate for connecting to D-Wave, embedding the QUBO, and retrieving samples. Document its structure and how it interfaces with the main QA-MAB loop.

**Practical considerations:** D-Wave access costs, queue times, problem size limits (current Advantage system: ~5000 qubits), and how our QUBO size scales with K agents and M paths.

---

## 4.5 Expected Improvement: Quantum Tunneling and the Optimality Gap

**What goes here:** Predictions for how much quantum annealing can improve routing performance, based on the theoretical analysis of Chapters 1–3.

**Analysis framework:**
1. **Collision reduction:** QA samples diverse solutions from the degenerate manifold → fewer collisions than SA-Oracle. Quantify expected collision rate under uniform sampling from degenerate minima.
2. **Comparison baseline:** QA-MAB with QA solver vs QA-MAB with SA solver (our current system). The SA solver already benefits from noise (Chapter 3), so QA must improve on noise-assisted SA, not just on Oracle-SA.
3. **The key question:** Does QA provide *better* symmetry breaking than estimation noise? Arguments for: QA's sampling is informed by the energy landscape structure; noise is random. Arguments against: current QA hardware has limited precision; noise is "free" while QA requires hardware.

**Expected results (to be validated):**
- QA should close the remaining optimality gap (currently ~10% from §1.3) by providing structured diversity
- The improvement should be most pronounced in high-symmetry instances (many healthy paths, many agents)
- QA should maintain good performance even as θ̂ → θ* (where noise-based symmetry breaking vanishes)

**Include:** A simulation-based prediction using the "QA simulator" (sample uniformly from degenerate minima) as a proxy for actual QA.

---

## 4.6 Limitations and Risks: When Quantum Doesn't Help

**What goes here:** Honest assessment of the limitations of the quantum annealing approach. No thesis should oversell its solution.

**Limitations to discuss:**

1. **Hardware noise vs useful quantum effects:** Current D-Wave systems have thermal noise, control errors, and decoherence. The "quantum" samples may not be better than classical random restarts in practice.
2. **Embedding overhead:** All-to-all QUBO connectivity requires long chains in the Pegasus graph. Chain breaks introduce errors that may negate quantum advantage.
3. **Problem scale:** For small K and M, the QUBO is small enough that classical SA with random restarts may solve it well. Quantum advantage, if it exists, likely manifests only at scale.
4. **The noise paradox revisited:** If estimation noise already provides good symmetry breaking (Chapter 3), the marginal value of QA's tunneling may be small. QA is most valuable when estimates converge (θ̂ → θ*, noise vanishes).
5. **Cost and access:** D-Wave QPU time is a limited resource. Is the improvement worth the cost compared to classical alternatives (multi-start SA, Thompson Sampling)?

**Risks:**
- QA may not outperform clever classical heuristics (e.g., explicit random tie-breaking in SA)
- The QUBO may need reformulation for efficient embedding on quantum hardware
- Results on D-Wave may not be reproducible due to hardware variability

**Mitigation:** Frame the thesis contribution as the *theoretical insight* (symmetry breaking, the paradox, the noise mechanism) with QA as *one promising solver*, not the only one.

---

# Open Questions

**What still needs to be done to complete this thesis:**

## Theoretical

1. **Rigorous convergence proof (§2.4):** The proof sketch needs to be formalized. Handle the product structure in the gradient and the coupling between θ̂ and φ̂. Determine if existing stochastic approximation theorems (Robbins-Monro, ODE method) apply directly or need extension.

2. **Identifiability conditions (§2.6):** Prove or disprove the conjecture that full column rank of the path-edge incidence matrix is necessary and sufficient for identifiability. Characterize what happens when the matrix is rank-deficient.

3. **Optimal noise level (§3.3):** The paradox theorem shows that *some* noise helps, but what is the optimal ||ε||? Can we characterize the noise level that maximizes throughput as a function of K, M_h, and network structure?

4. **Convergence-performance tradeoff (§3.4):** As P → ∞ and θ̂ → θ*, does routing performance eventually degrade? Is there a critical epoch P* after which convergence hurts more than it helps? This would formalize the tension between learning and routing.

## Experimental

5. **D-Wave experiments (§4.4–4.5):** Run the QUBO on actual D-Wave hardware. Compare QA samples to SA samples in terms of diversity and routing throughput. This is the central experimental validation of Chapter 4.

6. **Larger-scale simulations:** Our current results are for specific K and M values. Test scalability: how does the paradox behave as the network grows? Does the advantage of noise increase or decrease?

7. **Alternative symmetry-breaking methods (§3.6):** Implement and compare explicit tie-breaking, anti-collision regularization, and multi-start SA against noise-based symmetry breaking. If a classical method matches QA, the quantum advantage argument weakens.

8. **Thompson Sampling comparison:** Implement a Thompson Sampling variant of QA-MAB (sample θ from posterior rather than using point estimate θ̂). This is a natural classical alternative that provides deliberate noise for symmetry breaking.

## Writing

9. **Related work survey:** Position our paradox finding relative to existing literature on: (a) symmetry breaking in combinatorial optimization, (b) noise benefits in optimization (stochastic gradient descent, simulated annealing), (c) quantum advantage for combinatorial problems.

10. **Notation unification:** Ensure consistent notation across chapters. Currently θ, φ, Q, x are overloaded in different contexts.

11. **Figure plan:** Design the key figures — convergence plots (Ch. 2), energy landscape visualization (Ch. 1 & 3), throughput comparison (Ch. 3), QA vs SA sample diversity (Ch. 4).

---

# Appendices (Planned)

- **Appendix A:** Full QUBO derivation with constraint encoding details
- **Appendix B:** SA hyperparameter sensitivity analysis (30-seed study details)
- **Appendix C:** Convergence proof — full technical version
- **Appendix D:** D-Wave embedding details and code documentation
- **Appendix E:** Simulation codebase overview and reproducibility instructions

---

*This outline is a living document. Update section descriptions as results solidify and new insights emerge.*
