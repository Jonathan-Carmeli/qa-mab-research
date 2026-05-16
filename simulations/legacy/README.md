# Legacy Simulation Files

All files from the **original CMAB model** (pre-physical upgrade).

## Structure
- `*.py` — simulation scripts (old model)
- `results/` — results from old model validation runs

## Key Files
- `qa_mab.py` — original QA-MAB agent (CMAB model)
- `nb3r.py` — NB3R baseline agent
- `simulation_core.py` — core simulation loop
- `simulation_v2.py` — sparse network variant
- `stochastic_noise_experiment.py` — noise robustness
- `dwave_setup.py` — D-Wave integration setup

## Validation (old model)
- `validate_cat2_qubo_optimality.py` → Cat 2 results
- `validate_cat3_sa_solver_accuracy.py` → Cat 3 results
- `validate_cat4_learning_dynamics.py` → Cat 4 results
- `validate_cat7_sqa_comparison.py` → Cat 7 results (SA vs SQA)
- `validate_cat8_log_scaled.py` → Cat 8 results (incomplete)

**⚠️** Do not use these files for new experiments. Use `../physical_validation/` instead.