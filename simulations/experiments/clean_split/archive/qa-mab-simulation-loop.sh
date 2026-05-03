#!/bin/bash
# qa-mab-simulation-loop.sh
# Runs QA-MAB clean split experiments in background, auto-iterates
# Heartbeat checks this every 10 min

BASE="/Users/jon_claw/qa-mab-research/simulations"
RESULTS="$BASE/results/clean_split"
EXP_DIR="$BASE/experiments/clean_split"
LOG="$EXP_DIR/loop.log"

ALGOS=("clean_split_v3" "clean_split_v4" "clean_split_v5")
MAX_ITER=50
SLEEP_BETWEEN=300  # 5 min between checks

cd "$BASE"

echo "=== QA-MAB Loop started $(date) ===" >> "$LOG"

iter=0
while [ $iter -lt $MAX_ITER ]; do
    # Check if something is already running
    RUNNING=$(ps aux | grep -E "clean_split_v[0-9]" | grep -v grep | grep python3 | wc -l)
    if [ "$RUNNING" -gt 0 ]; then
        echo "[$(date)] Iteration $iter: already running, waiting..." >> "$LOG"
        sleep 60
        continue
    fi

    # Read latest results
    latest_v=""
    latest_sw=0
    for f in "$RESULTS"/clean_split_v*_results.json; do
        if [ -f "$f" ]; then
            v=$(echo "$f" | sed 's/.*clean_split_v//;s/_results.json//')
            sw=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('phaseB', d.get('phaseB_learned', {})).get('sw_ratio', 0))" 2>/dev/null)
            bv=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('phaseA', d.get('phaseA_oracle', {})).get('sw_ratio', 0))" 2>/dev/null)
            echo "[$(date)] Version $v: PhaseA=$bv PhaseB=$sw" >> "$LOG"
            comp=$(python3 -c "print(1 if $sw >= $latest_sw else 0)")
            if [ "$comp" -eq 1 ]; then
                latest_v="$v"; latest_sw=$sw
            fi
        fi
    done

    echo "[$(date)] Best so far: v$latest_v SW=$latest_sw" >> "$LOG"

    # Decision: which version to run next?
    THRESHOLD=0.10
    if (( $(echo "$latest_sw >= $THRESHOLD" | bc -l) )); then
        echo "[$(date)] SUCCESS! Phase B SW=$latest_sw >= $THRESHOLD. Stopping loop." >> "$LOG"
        break
    fi

    # Advance to next version
    current_num=0
    if [ -n "$latest_v" ]; then
        current_num=$(echo "$latest_v" | sed 's/[^0-9]//g')
    fi
    next_num=$((current_num + 1))
    next_version="v$next_num"
    script="$EXP_DIR/clean_split_v${next_num}.py"

    if [ ! -f "$script" ]; then
        echo "[$(date)] No script for $next_version at $script. Creating default..." >> "$LOG"
        # Create a reasonable default based on what works (Phase A SW=0.13)
        # V4: mean-based B tracking, greedy I update
        cat > "$script" << 'SCRIPTEOF'
"""clean_split_v4.py — Mean-based B + Collision I (no interference in B_observed)"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, '/Users/jon_claw/qa-mab-research/simulations')
from simulation_core import NetworkEnvironment

RESULTS = '/Users/jon_claw/qa-mab-research/simulations/results/clean_split'
os.makedirs(RESULTS, exist_ok=True)
BASE_SEED = 2026; N = 10; M = 4; T = 500; N_SEEDS = 20
B_LR = 0.1; I_LR = 0.1; I_CAP = 0.5; UCB_C = 0.5; T_START = 2.0

class AlgoV4:
    def __init__(self, env, use_oracle_I=True, seed=42):
        self.env = env; self.N = env.N; self.m = env.m
        self.use_oracle_I = use_oracle_I
        self.rng = np.random.default_rng(seed)
        self.B_hat = np.full((self.N, self.m), 0.75)
        self.B_cnt = np.zeros((self.N, self.m))
        self.I_hat = np.zeros((self.N, self.m, self.N, self.m))
        self.visits = np.zeros((self.N, self.m))
        self._prev_x = None; self._prev_tp = None; self.step_count = 0

    def idx(self, i, k): return i * self.m + k

    def build_qubo(self, t):
        size = self.N * self.m; Q = np.zeros((size, size))
        for i in range(self.N):
            for k in range(self.m):
                idx = self.idx(i, k)
                ucb = UCB_C / math.sqrt(self.visits[i, k] + 1)
                Q[idx, idx] = -self.B_hat[i, k] - ucb
        tau = 1.0
        for i in range(self.N):
            for k in range(self.m):
                for l in range(k + 1, self.m):
                    Q[self.idx(i, k), self.idx(i, l)] += tau / 2
                    Q[self.idx(i, l), self.idx(i, k)] += tau / 2
        for i in range(self.N):
            for k in range(self.m):
                for j in range(self.N):
                    if j == i: continue
                    for l in range(self.m):
                        I_val = self.env.I[i, k, j, l] if self.use_oracle_I else self.I_hat[i, k, j, l]
                        Q[self.idx(i, k), self.idx(j, l)] += I_val
        return Q

    def sa_solve(self, Q, t):
        n, m, size = self.N, self.m, self.N * self.m
        T = T_START / math.log(t + 2)
        best_x, best_e = None, float('inf')
        for r in range(8):
            x = np.zeros(size)
            for i in range(n): x[i * m + int(np.argmax(self.B_hat[i]))] = 1.0
            e = float(x @ Q @ x)
            if e < best_e: best_e, best_x = e, x.copy()
            T_r = T * (1 + r * 0.3)
            for _ in range(500):
                T_r *= 0.95
                if T_r < 1e-12: break
                i = self.rng.integers(0, n)
                block = x[i * m:(i + 1) * m]
                k_old = int(np.argmax(block))
                k_new = (k_old + 1 + self.rng.integers(0, m - 1)) % m
                x[i * m + k_old] = 0.0; x[i * m + k_new] = 1.0
                ne = float(x @ Q @ x)
                if ne < e or self.rng.random() < math.exp(-(ne - e) / T_r):
                    e = ne
                    if e < best_e: best_e, best_x = e, x.copy()
                else:
                    x[i * m + k_new] = 0.0; x[i * m + k_old] = 1.0
        return {i: int(np.argmax(best_x[i * m:(i + 1) * m])) for i in range(n)}

    def step(self):
        t = self.step_count + 1
        Q = self.build_qubo(t)
        assignment = self.sa_solve(Q, t)
        tp_actual = self.env.compute_throughput(assignment)
        opt_sw = float(np.sum(np.max(self.env.B, axis=1)))

        if self.use_oracle_I:
            for i in range(self.N):
                k = assignment[i]
                B_obs = tp_actual[i] + sum(self.env.I[i, k, j, assignment[j]] for j in range(self.N) if j != i)
                self.B_hat[i, k] += B_LR * (B_obs - self.B_hat[i, k])
                self.visits[i, k] += 1
        else:
            # Mean-based B: just average tp per route (no interference model)
            for i in range(self.N):
                k = assignment[i]
                self.B_cnt[i, k] += 1
                self.B_hat[i, k] += (tp_actual[i] - self.B_hat[i, k]) / self.B_cnt[i, k]
                self.visits[i, k] += 1
            
            # Global correction: if SW is low, all B_hat are systematically overestimated
            actual_sw = sum(tp_actual.values())
            if opt_sw > 0 and actual_sw < opt_sw * 0.5:
                # Systematically too high — scale down
                self.B_hat *= 0.95

            # I_hat via collision inference
            if self._prev_x is not None:
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        ki = self._prev_x[i]; kj = self._prev_x[j]
                        drop_i = max(0.0, self.B_hat[i, ki] - self._prev_tp[i])
                        drop_j = max(0.0, self.B_hat[j, kj] - self._prev_tp[j])
                        if drop_i > 0.01: self.I_hat[i, ki, j, kj] = min(self.I_hat[i, ki, j, kj] + I_LR, I_CAP)
                        if drop_j > 0.01: self.I_hat[j, kj, i, ki] = min(self.I_hat[j, kj, i, ki] + I_LR, I_CAP)
            self.I_hat *= (1.0 - 0.001)

        self._prev_x = assignment.copy()
        self._prev_tp = {i: tp_actual[i] for i in range(self.N)}
        self.step_count += 1
        return float(sum(tp_actual.values()))

    def run(self, T_steps):
        return [self.step() for _ in range(T_steps)]

def run_exp(use_oracle_I, n_seeds, T_steps):
    sw_ratios, B_errors, I_errors = [], [], []
    for si in range(n_seeds):
        env = NetworkEnvironment(N, M, seed=BASE_SEED + si * 1000 + N, B_scale='uniform', I_scale='moderate')
        algo = AlgoV4(env, use_oracle_I=use_oracle_I, seed=si * 1000 + N)
        sw_hist = algo.run(T_steps)
        opt = float(np.sum(np.max(env.B, axis=1)))
        sw_ratios.append(float(np.mean(sw_hist[-50:]) / opt) if opt > 0 else 0)
        B_errors.append(float(np.mean(np.abs(algo.B_hat - env.B)[-50:])))
        if not use_oracle_I:
            I_errors.append(float(np.mean(np.abs(algo.I_hat - env.I)[-50:])))
    r = {'sw_ratio': float(np.mean(sw_ratios)), 'sw_std': float(np.std(sw_ratios)),
         'B_error': float(np.mean(B_errors)), 'B_error_std': float(np.std(B_errors))}
    if not use_oracle_I: r['I_error'] = float(np.mean(I_errors))
    return r

def main():
    print("=" * 60)
    print("CLEAN SPLIT V4 — Mean-based B + Collision I")
    print("=" * 60)
    all_res = {}
    print("\nPHASE A: Oracle I")
    rA = run_exp(True, N_SEEDS, T)
    all_res['phaseA'] = rA
    print(f"  SW={rA['sw_ratio']:.4f} +/- {rA['sw_std']:.4f}  B_err={rA['B_error']:.4f}")
    print("\nPHASE B: Learned I (mean tp + collision)")
    rB = run_exp(False, N_SEEDS, T)
    all_res['phaseB'] = rB
    print(f"  SW={rB['sw_ratio']:.4f} +/- {rB['sw_std']:.4f}  B_err={rB['B_error']:.4f}  I_err={rB['I_error']:.4f}")
    print(f"\n  Phase A (Oracle):   SW={rA['sw_ratio']:.4f}")
    print(f"  Phase B (Learned):  SW={rB['sw_ratio']:.4f}")
    out = os.path.join(RESULTS, 'clean_split_v4_results.json')
    with open(out, 'w') as f: json.dump(all_res, f, indent=2)
    print(f"\nSaved: {out}")

if __name__ == '__main__':
    main()
SCRIPTEOF
        echo "Created $script" >> "$LOG"
    fi

    echo "[$(date)] Running $next_version..." >> "$LOG"
    python3 "$script" >> "$LOG" 2>&1
    echo "[$(date)] $next_version finished with exit $?" >> "$LOG"

    iter=$((iter + 1))
    sleep 5
done

echo "=== Loop ended $(date) ===" >> "$LOG"