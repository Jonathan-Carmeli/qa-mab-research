# QA-MAB Research Loop — Autonomous Iteration

## Mission
Answer the core question: **Does QA-MAB's learning (θ̂/φ̂) actually work, and can we improve it?**

## The Loop
```
Question → Simulation → Answer → Next Question → Simulation → ...
```

## Question Queue

1. **CURRENT:** Does the learning (θ̂/φ̂) actually matter? or is it all QUBO structure?
   - **Test:** QA-MAB vs NoLearn (θ̂=φ̂=0) vs Oracle over P=15, same topology
   - **Running in:** quick-shore session
   - **Expected answer:** If QA-MAB >> NoLearn → learning matters; If ≈ → QUBO does the work

2. **If learning matters:** Does epoch_decay=0.7 lose too much info?
   - **Test:** decay sweep {1.0, 0.99, 0.95, 0.9, 0.7} × P=20
   - **Running in:** marine-fjord session
   - **Expected answer:** optimal decay for transfer learning

3. **If decay=0.7 is too strong:** What decay lets θ̂ converge as P→∞?
   - **Test:** decay ∈ {0.95, 0.99, 1.0} with P=50, track theta_err
   - **Status:** pending

4. **If decay=1.0 works:** Can we close the optimality gap with enough epochs?
   - **Test:** decay=1.0, P=50, compare loss to Optimal agent
   - **Status:** pending

5. **If P doesn't help:** Can D-Wave close the 5.57× optimality gap?
   - **Test:** SA vs D-Wave on same QUBO
   - **Status:** pending (needs D-Wave token)

## Status at 2026-05-03 16:55
- [ ] Question 1 (NoLearn): running (quick-shore)
- [ ] Question 2 (decay sweep): running (marine-fjord)  
- [ ] Question 3: pending
- [ ] Question 4: pending
- [ ] Question 5: pending

## How to Monitor
```bash
# Check what's running
ps aux | grep -E "python3.*uav" | grep -v grep

# Check latest results
python3 -c "
import pickle
with open('/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/learning_gap_results.pkl','rb') as f:
    r = pickle.load(f)
for k,v in r.items():
    print(f'decay={k}: LearnGain={v[\"learning_gain\"].mean():+.3f} OracleGap={v[\"oracle_gap\"].mean():+.3f}')
"
```

## Stop Signal
```bash
touch /tmp/qa_mab_loop_stop
```
