# DIAMOND Convergence Conditions — Under What Conditions Does NB3R Converge?

## The Core Theorem (Corollary 1 in DIAMOND)

> **DIAMOND proves:** Under Boltzmann policy with logarithmic cooling schedule ν(t) = log(t)/Δ, NB3R converges to the globally optimal routing strategy with probability 1 as t → ∞.

## The Three Conditions for Convergence

### Condition 1: Collaborative Utility Definition

Each agent n computes a **collaborative utility**:

$$U_n(\sigma) = u_n(\sigma) + \sum_{m \in \mathcal{N}_n} u_m(\sigma)$$

Where:
- $u_n(\sigma)$ = the actual throughput agent n receives under assignment σ
- $\mathcal{N}_n$ = the set of **neighbors** of agent n (defined by the network topology)
- **Key:** Each agent sums its OWN utility plus the utilities of its neighbors

**This is LOCAL communication** — each agent only talks to its neighbors, not everyone.

### Condition 2: Connected Neighborhood Graph

The proof requires that the **neighborhood graph** (where nodes = agents, edges = neighbor relationships) is **connected**.

If the graph has disconnected components → agents in one component never receive information from agents in another → no convergence to global optimum.

**In practice:** If you have clusters that don't communicate with each other, NB3R won't coordinate across clusters.

### Condition 3: Logarithmic Cooling Schedule

The temperature must decrease slowly enough:

$$\nu(t) = \frac{\log(t)}{\Delta}, \quad \Delta = N \cdot u_{\max}$$

Where:
- N = number of agents
- $u_{\max}$ = maximum possible utility per agent
- This is the same logarithmic schedule proven by Hajek (1988) for simulated annealing

**In practice:** Fast cooling (linear or geometric schedules) may converge faster but is not guaranteed to find the global optimum.

## Why This Matters for Our Work

### What DIAMOND Proves About Local Communication

**Yes, DIAMOND proves convergence with LOCAL communication** — each agent only broadcasts to its neighbors (Condition 1).

The key insight is that $u_n(\sigma)$ already contains **global information**:

$$u_n(\sigma) = B_n - \sum_{j} I_{n,j} \cdot x_j$$

The interference term $\sum_j I_{n,j} \cdot x_j$ sums over **all** agents j, not just neighbors. So when agent n broadcasts $U_n$, it carries information about the entire network's state — just communicated locally through neighbor chains.

### What DIAMOND Does NOT Cover

1. **Convergence rate** — only proves eventual convergence, not how fast
2. **Partial observability** — assumes each agent observes its true $u_n$
3. **Asynchronous updates** — the proof is for synchronous updates
4. **Quantized/bandwidth-limited communication** — assumes real-valued broadcasts
5. **Non-convergent topologies** — disconnected graphs aren't covered

## The Topology Assumption

**Connected graph assumption** is critical.

In DIAMOND's model:
- Agents are connected if they're within communication range
- The centralized unit (CU) in 5G can reach all agents
- The neighborhood graph in their simulations is always connected

**In sparse cluster networks:**
- If clusters don't have any cross-cluster communication links
- Then the neighborhood graph is **disconnected**
- → DIAMOND's convergence guarantee **does not apply**

This is why in our simulation_v2, SparseNB3R ≈ Random when clusters are isolated.

## Practical Implications

### When NB3R Converges (DIAMOND Guarantees):
- Fully connected network (CU can reach all agents)
- Mesh network with sufficient connectivity
- Any network where the communication graph is connected

### When NB3R May NOT Converge (No Guarantee):
- Isolated clusters with no cross-cluster links
- Very sparse networks (nodes may not propagate information far enough)
- Networks with dynamic topology changes

### For Real Wireless Networks:
- WiFi access points in the same building → likely connected via backbone
- Cellular base stations → always connected via carrier infrastructure
- Ad-hoc mesh networks → depends on density

## Citation

From DIAMOND (arXiv:2303.15544):
- Theorem 1: Existence of optimal policy
- Corollary 1: Logarithmic cooling schedule guarantees convergence
- The distributed stage (NB3R) communication is explicitly local (neighbors only)

## Tags
#thesis #quantum #diamond #convergence #nb3r #local-communication
