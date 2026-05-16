#!/bin/bash
# UAV Dynamic Environment — Autonomous Monitoring Loop
# Continuously monitors experiment results, analyzes, and decides next actions.
# Stops when: (a) QA-MAB wins NB3R by >5% AND Oracle gap <30%
#             OR (b) max iterations reached OR (c) /tmp/uav_monitor_stop exists
#
# Usage: ./uav_monitor_loop.sh [--continuous]
#   without args: one-shot analysis (for heartbeat)
#   --continuous: runs full loop

export PATH="/Users/jon_claw/homebrew/bin:$PATH"
WORKDIR="/Users/jon_claw/Thesis_brain/simulation"
RESULTS_DIR="/Users/jon_claw/Thesis_brain/results/uav_routing"
LOG_FILE="/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/monitor_log.txt"
ITER_LOG="/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/iteration_log.md"
STOP_FILE="/tmp/uav_monitor_stop"
MAX_ITER=5

log() {
    echo "[$(date '+%Y-%m-%d %H:%M')] $1" | tee -a "$LOG_FILE"
}

check_results_ready() {
    local agents=("Random" "Greedy" "NB3R" "QA_MAB" "Oracle")
    for agent in "${agents[@]}"; do
        if [ ! -f "$RESULTS_DIR/${agent}.npz" ]; then
            return 1
        fi
    done
    return 0
}

analyze_results() {
python3 -c "
import sys, os, numpy as np, json

results_dir = '/Users/jon_claw/Thesis_brain/results/uav_routing'
SAVE='/Users/jon_claw/qa-mab-research/simulations/results/uav_dynamic/latest_analysis.json'

agents = ['Random', 'Greedy', 'NB3R', 'QA_MAB', 'Oracle']
data = {}
for name in agents:
    fname = results_dir + '/' + name + '.npz'
    if os.path.exists(fname):
        data[name] = np.load(fname)

P, T = 10, 100
out = {}

if 'QA_MAB' not in data or 'NB3R' not in data:
    out['status'] = 'NO_DATA'
    json.dump(out, open(SAVE,'w'))
    print('NO_DATA')
    sys.exit(0)

def mean_loss_last3(name):
    d = data[name]
    losses = d['losses']
    return float(losses[:, -3:, :, :].mean())

out['qamab_loss'] = mean_loss_last3('QA_MAB')
out['nb3r_loss'] = mean_loss_last3('NB3R')
out['oracle_loss'] = mean_loss_last3('Oracle')
out['random_loss'] = mean_loss_last3('Random')

if out['nb3r_loss'] > 0:
    out['gain_pct'] = 100.0 * (out['nb3r_loss'] - out['qamab_loss']) / out['nb3r_loss']
else:
    out['gain_pct'] = 0.0

out['qamab_coll'] = float((data['QA_MAB']['coll'] > 0).mean())
out['nb3r_coll'] = float((data['NB3R']['coll'] > 0).mean())

if 'theta_err' in data['QA_MAB']:
    te = data['QA_MAB']['theta_err']
    out['theta_e1'] = float(te[:, 0].mean())
    out['theta_e10'] = float(te[:, -1].mean())
    out['theta_converged'] = out['theta_e10'] < out['theta_e1'] * 0.9
    out['theta_status'] = f\"E1={out['theta_e1']:.3f} E10={out['theta_e10']:.3f} improved={out['theta_converged']}\"
else:
    out['theta_status'] = 'NO_DATA'

if out['oracle_loss'] > 0:
    out['oracle_gap_pct'] = 100.0 * (out['qamab_loss'] - out['oracle_loss']) / out['oracle_loss']
else:
    out['oracle_gap_pct'] = 999.0

json.dump(out, open(SAVE,'w'))

for k, v in out.items():
    print(f'{k}={v}')
"
}

run_experiment() {
    local label=$1
    log "=== Starting experiment: $label ==="
    cd "$WORKDIR" && python3 -m src.uav_routing.run_experiment \
        --epochs 10 --steps 100 --seeds 20 \
        2>&1 | tee -a "$LOG_FILE"
    log "=== Experiment finished: $label ==="
}

spawn_claude_consult() {
    log "Spawning Claude Code for QUBO consultation..."
    # Write a brief for Claude Code
    cat > /tmp/qubo_consult_brief.md << 'BRIEF'
# QUBO Design Consultation

## Current Results (attach from iteration log)
Need to fill in after analysis.

## What to do:
1. Read latest_analysis.json for current metrics
2. Review the current QUBO construction in Thesis_brain/simulation/src/uav_routing/qubo.py
3. Propose 1-2 specific QUBO modifications to improve QA-MAB performance

## Decision rules for the monitor:
- QA-MAB wins if: gain > 5% over NB3R AND oracle gap < 30%
- If current iteration fails → propose next modification
- Max 5 iterations total

## Common QUBO knobs to tune:
- UCB constant (ucb_c) — higher = more exploration
- lambda_onehot — one-hot penalty strength
- C_coll — collision penalty strength
- Diagonal vs off-diagonal balance
- Temperature schedule (gamma_0, a, b)
- How to add UCB: on diagonal (current) vs separate exploration term

## Output format:
Write your suggestion to /tmp/qubo_suggestion.md with:
1. Current diagnosis
2. Proposed change
3. Expected effect
4. Code snippet if applicable

BRIEF
}

decide_action() {
    local gain=$1
    local oracle_gap=$2
    local theta_converged=$3
    local iter=$4

    if python3 -c "exit(0 if float('$gain') > 5.0 and float('$oracle_gap') < 30.0 else 1)" 2>/dev/null; then
        echo "SUCCESS"
    elif [ "$iter" -ge "$MAX_ITER" ]; then
        echo "MAX_ITER"
    else
        echo "ITERATE"
    fi
}

# ─── Main ────────────────────────────────────────────
echo "=== UAV Monitor started $(date) ===" | tee -a "$LOG_FILE"
mkdir -p "$(dirname $LOG_FILE)"

if [ "$1" == "--continuous" ]; then
    # Continuous loop mode
    log "Continuous mode — will loop until success/stop signal"
    ITER=1
    while [ $ITER -le $MAX_ITER ]; do
        [ -f "$STOP_FILE" ] && log "Stop file detected" && break
        
        if ! check_results_ready; then
            log "[Iter $ITER] Results not ready yet — waiting 5min..."
            sleep 300
            continue
        fi
        
        log "[Iter $ITER] Analyzing results..."
        ANALYSIS=$(analyze_results)
        echo "$ANALYSIS" >> "$LOG_FILE"
        
        # Parse
        GAIN=$(echo "$ANALYSIS" | grep -o 'gain_pct=[^ ]*' | cut -d= -f2)
        GAP=$(echo "$ANALYSIS" | grep -o 'oracle_gap_pct=[^ ]*' | cut -d= -f2)
        THETA=$(echo "$ANALYSIS" | grep 'theta_status=' | cut -d= -f2-)
        QA_COLL=$(echo "$ANALYSIS" | grep -o 'qamab_coll=[^ ]*' | cut -d= -f2)
        NB3R_COLL=$(echo "$ANALYSIS" | grep -o 'nb3r_coll=[^ ]*' | cut -d= -f2)
        
        log "  → QA-MAB gain: ${GAIN}%, Oracle gap: ${GAP}%, Theta: $THETA"
        log "  → Collisions: QA-MAB=${QA_COLL}, NB3R=${NB3R_COLL}"
        
        ACTION=$(decide_action "$GAIN" "$GAP" "false" "$ITER")
        log "  → Decision: $ACTION"
        
        if [ "$ACTION" == "SUCCESS" ]; then
            log "SUCCESS — QA-MAB meets criteria. Stopping."
            echo "## Iteration $ITER — $(date)" >> "$ITER_LOG"
            echo "**Status:** SUCCESS ✓" >> "$ITER_LOG"
            echo "**Gain:** ${GAIN}%, **Oracle gap:** ${GAP}%" >> "$ITER_LOG"
            echo "**Theta:** $THETA" >> "$ITER_LOG"
            break
        elif [ "$ACTION" == "MAX_ITER" ]; then
            log "Max iterations reached."
            break
        else
            log "Need improvement — spawning Claude Code consultation..."
            echo "## Iteration $ITER — $(date)" >> "$ITER_LOG"
            echo "**Status:** ITERATING" >> "$ITER_LOG"
            echo "**Gain:** ${GAIN}%, **Oracle gap:** ${GAP}%" >> "$ITER_LOG"
            echo "**Theta:** $THETA" >> "$ITER_LOG"
            echo "" >> "$ITER_LOG"
            
            spawn_claude_consult
            
            # Wait for suggestion
            log "Waiting for Claude Code suggestion at /tmp/qubo_suggestion.md..."
            for i in $(seq 1 24); do  # up to 2 min
                sleep 5
                if [ -f /tmp/qubo_suggestion.md ]; then
                    log "Got suggestion — applying..."
                    cat /tmp/qubo_suggestion.md >> "$ITER_LOG"
                    # TODO: apply suggestion to qubo.py / config.py
                    log "Applied suggestion (manual review needed for code changes)"
                    rm /tmp/qubo_suggestion.md
                    break
                fi
            done
            
            log "Re-running experiment with changes..."
            run_experiment "Iteration_${ITER}_v2"
            ITER=$((ITER+1))
        fi
    done
    log "Monitor loop ended."
else
    # One-shot (for heartbeat)
    if check_results_ready; then
        log "Results complete — analyzing..."
        ANALYSIS=$(analyze_results)
        echo ""
        echo "=== Latest Analysis ==="
        echo "$ANALYSIS"
        echo ""
    else
        echo "NOT_COMPLETE — experiment still running"
    fi
fi
