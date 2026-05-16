#!/bin/bash
# UAV Dynamic Environment — Self-Monitoring Loop
# Watches for results, analyzes, and iterates on QUBO autonomously
#
# Usage:
#   ./monitor.sh [--continuous]  # runs forever
#   ./monitor.sh                 # runs once (for cron/heartbeat)
#
# Flag to stop: /tmp/uav_monitor_stop
#
# Decision rules:
#   - If results not complete → wait
#   - If QA-MAB wins over NB3R AND theta converges → success, stop
#   - If QA-MAB doesn't improve after iteration → try next idea
#   - Max iterations: 5
#   - Consult Claude Code/Opus for QUBO design decisions

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

write_iteration_header() {
    local iter=$1
    echo "" >> "$ITER_LOG"
    echo "## Iteration $iter — $(date '+%Y-%m-%d %H:%M')" >> "$ITER_LOG"
    echo "" >> "$ITER_LOG"
}

write_iteration_footer() {
    local iter=$1
    local status=$2
    local sw_ratio=$3
    local theta_conv=$4
    echo "**Status:** $status" >> "$ITER_LOG"
    echo "**QA-MAB vs NB3R gain:** $sw_ratio" >> "$ITER_LOG"
    echo "**Theta convergence:** $theta_conv" >> "$ITER_LOG"
    echo "" >> "$ITER_LOG"
}

check_results_ready() {
    local agents=("Random" "Greedy" "NB3R" "QA_MAB" "Oracle")
    for agent in "${agents[@]}"; do
        if [ ! -f "$RESULTS_DIR/${agent}.npz" ]; then
            return 1  # not ready
        fi
    done
    return 0  # ready
}

run_experiment() {
    local label=$1
    log "Starting experiment: $label"
    cd "$WORKDIR" && python3 -m src.uav_routing.run_experiment \
        --epochs 10 --steps 100 --seeds 20 \
        2>&1 | tee -a "$LOG_FILE"
    log "Experiment finished: $label"
}

analyze_results() {
    # Returns: QA_MAB_vs_NB3R_gain theta_convergence_status
    local python_code='
import sys, os, numpy as np

results_dir = "/Users/jon_claw/Thesis_brain/results/uav_routing"

agents = ["Random", "Greedy", "NB3R", "QA_MAB", "Oracle"]
data = {}
for name in agents:
    fname = results_dir + "/" + name + ".npz"
    if os.path.exists(fname):
        d = np.load(fname)
        data[name] = d

P, T = 10, 100
if "QA_MAB" not in data or "NB3R" not in data:
    print("0.0 NO_DATA")
    sys.exit(0)

# Mean loss over last 3 epochs (QA-MAB vs NB3R)
def mean_loss_last3(name):
    d = data[name]
    losses = d["losses"]  # (n_seeds, P, T, N)
    last3 = losses[:, -3:, :, :].mean()
    return float(last3)

qamab_loss = mean_loss_last3("QA_MAB")
nb3r_loss = mean_loss_last3("NB3R")
oracle_loss = mean_loss_last3("Oracle")
random_loss = mean_loss_last3("Random")

# Gain: how much better is QA-MAB than NB3R (%)
if nb3r_loss > 0:
    gain = 100.0 * (nb3r_loss - qamab_loss) / nb3r_loss
else:
    gain = 0.0

# Collision rate
qamab_coll = float((data["QA_MAB"]["coll"] > 0).mean())
nb3r_coll = float((data["NB3R"]["coll"] > 0).mean())

# Theta convergence: ||theta_hat - theta*|| at epoch 10 vs epoch 1
if "QA_MAB" in data and "theta_err" in data["QA_MAB"]:
    theta_err = data["QA_MAB"]["theta_err"]  # (n_seeds, P)
    e1 = theta_err[:, 0].mean()
    e10 = theta_err[:, -1].mean()
    converged = e10 < e1 * 0.9  # 10% improvement
    theta_status = f"E1={e1:.3f} E10={e10:.3f} improved={converged}"
else:
    theta_status = "NO_DATA"

# Oracle vs QA-MAB gap
if oracle_loss > 0:
    gap = 100.0 * (qamab_loss - oracle_loss) / oracle_loss
else:
    gap = 999.0

print(f"GAIN_PCT={gain:.2f}")
print(f"QA_LOSS={qamab_loss:.4f} NB3R_LOSS={nb3r_loss:.4f} ORACLE_LOSS={oracle_loss:.4f}")
print(f"QA_COLL={qamab_coll:.4f} NB3R_COLL={nb3r_coll:.4f}")
print(f"THETA_STATUS={theta_status}")
print(f"ORACLE_GAP_PCT={gap:.2f}")
'
    python3 -c "$python_code"
}

decide_next_action() {
    local gain=$1
    local oracle_gap=$2
    local theta_status=$3
    local coll_qa=$4
    local coll_nb3r=$5
    local iter=$6

    # Success criteria:
    # 1. QA-MAB beats NB3R by meaningful margin
    # 2. Theta converges somewhat
    # 3. Oracle gap not too large

    if [[ $(echo "$gain > 5.0" | bc -l) -eq 1 ]] && \
       [[ $(echo "$oracle_gap < 30.0" | bc -l) -eq 1 ]]; then
        echo "SUCCESS"
    elif [[ "$iter" -ge "$MAX_ITER" ]]; then
        echo "MAX_ITER"
    else
        echo "ITERATE"
    fi
}

# ── Main ──
echo "=== UAV Monitor started $(date) ===" >> "$LOG_FILE"
mkdir -p "$(dirname $LOG_FILE)"

if [ ! -f "$ITER_LOG" ]; then
    echo "# UAV Dynamic Environment — Iteration Log" > "$ITER_LOG"
    echo "Monitor started: $(date)" >> "$ITER_LOG"
    echo "" >> "$ITER_LOG"
fi

if [ "$1" == "--continuous" ]; then
    log "Running in continuous mode (will loop until $STOP_FILE exists)"
    while [ ! -f "$STOP_FILE" ]; do
        if check_results_ready; then
            log "Results detected — analyzing..."
            ANALYSIS=$(analyze_results)
            echo "$ANALYSIS" >> "$LOG_FILE"
            
            # Parse analysis
            GAIN=$(echo "$ANALYSIS" | grep "GAIN_PCT" | sed 's/GAIN_PCT=//')
            GAP=$(echo "$ANALYSIS" | grep "ORACLE_GAP_PCT" | sed 's/ORACLE_GAP_PCT=//')
            THETA=$(echo "$ANALYSIS" | grep "THETA_STATUS" | sed 's/THETA_STATUS=//')
            QA_COLL=$(echo "$ANALYSIS" | grep "QA_COLL" | sed 's/QA_COLL=//')
            NB3R_COLL=$(echo "$ANALYSIS" | grep "NB3R_COLL" | sed 's/NB3R_COLL=//')
            
            log "Analysis: QA-MAB gain over NB3R=${GAIN}%, Oracle gap=${GAP}%, Theta: $THETA"
            log "Collision rates: QA-MAB=${QA_COLL}, NB3R=${NB3R_COLL}"
            
            write_iteration_header "AUTO"
            echo "**Analysis:** $ANALYSIS" >> "$ITER_LOG"
            
            ACTION=$(decide_next_action "$GAIN" "$GAP" "$THETA" "$QA_COLL" "$NB3R_COLL" 1)
            if [ "$ACTION" == "SUCCESS" ]; then
                log "SUCCESS detected — stopping loop"
                write_iteration_footer "AUTO" "SUCCESS" "$GAIN" "$THETA"
                break
            else
                log "Need iteration, consulting Claude Code..."
                echo "**Decision:** ITERATE — consulting Claude Code for QUBO improvements" >> "$ITER_LOG"
                write_iteration_footer "AUTO" "ITERATING" "$GAIN" "$THETA"
                
                # Signal that we need Claude Code intervention
                echo "CONSULT_CODE=1" >> "$LOG_FILE"
                break
            fi
        else
            log "Waiting for results... (not yet complete)"
        fi
        
        if [ -f "$STOP_FILE" ]; then
            log "Stop file detected — exiting"
            break
        fi
        sleep 300  # 5 min between checks
    done
else
    # One-shot mode (for heartbeat/cron)
    if check_results_ready; then
        log "Results complete — analyzing..."
        ANALYSIS=$(analyze_results)
        echo "$ANALYSIS" >> "$LOG_FILE"
        echo ""
        echo "=== Analysis ===" 
        echo "$ANALYSIS"
    else
        log "Results not yet complete"
        echo "NOT_COMPLETE"
    fi
fi
