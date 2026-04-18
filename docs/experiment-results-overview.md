# QA-MAB Simulation Results

## Summary
Built and ran a complete simulation suite comparing **QA-MAB** (centralized, QUBO-based) vs **NB3R** (distributed, softmax-based) for multi-agent routing with interference.

Extension of [[DIAMOND-paper-notes]] (arXiv:2303.15544).

## Key Findings
- **Short horizon (T≤500):** NB3R wins — learns faster without explicit model
- **Long horizon (T≥10,000):** QA-MAB wins — accurate QUBO enables better optimization
- **Scaling (N↑):** QA-MAB advantage grows — more agents = more local minima = QA tunneling advantage
- **SA is weak proxy:** Simulated Annealing gets stuck in local minima that real QA would tunnel through
- **QAOA validated:** Qiskit QAOA simulation confirms quantum framework works (limited to ~10 qubits classically)

## Detailed Results
Full results, parameters, and analysis: [[RESULTS_SUMMARY]]

## Code Location
`simulations/qa_mab_extension/`

## Open Questions
- [ ] T=100,000 results (running)
- [ ] QAOA at N=5 (need quantum hardware or better simulator)
- [ ] D-Wave comparison (theoretical — would need hardware access)
- [ ] Ablation studies (I_scale, B_scale, λ, τ)

## Tags
#thesis #quantum #simulation #qa-mab #diamond
