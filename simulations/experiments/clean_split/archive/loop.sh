#!/bin/bash
RESULTS="/Users/jon_claw/qa-mab-research/simulations/results/clean_split"
EXP_DIR="/Users/jon_claw/qa-mab-research/simulations/experiments/clean_split"
LOG="$EXP_DIR/loop.log"
SLEEP=120
MAX_ITER=100

echo "=== Loop started $(date) ===" >> "$LOG"

iter=0
while [ $iter -lt $MAX_ITER ]; do
    # Find latest completed version
    latest_v=0; latest_sw=0; phase_b_sw=0
    for f in "$RESULTS"/clean_split_v*_results.json; do
        [ -f "$f" ] || continue
        v=$(echo "$f" | sed 's/.*clean_split_v//;s/_results.json//')
        pb=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('phaseB', d.get('phaseB_learned', {})).get('sw_ratio', -999))" 2>/dev/null)
        echo "[$(date)] V$v: PhaseB=$pb" >> "$LOG"
        comp=$(python3 -c "print(1 if $pb > $latest_sw else 0)" 2>/dev/null)
        [ "$comp" = "1" ] && latest_v=$v && latest_sw=$pb && phase_b_sw=$pb
    done
    echo "[$(date)] Best: V$latest_v SW=$latest_sw" >> "$LOG"

    # Success condition
    if (( $(echo "$phase_b_sw > 0.10" | bc -l) )); then
        echo "SUCCESS V$latest_v SW=$latest_sw" >> "$LOG"
        break
    fi

    # Next version
    next=$((latest_v + 1))
    script="$EXP_DIR/clean_split_v${next}.py"

    if [ ! -f "$script" ]; then
        echo "[$(date)] V$next not found. Loop ending." >> "$LOG"
        break
    fi

    # Kill any stale
    pkill -f "clean_split_v[0-9].py" 2>/dev/null
    sleep 2

    echo "[$(date)] Running V$next..." >> "$LOG"
    python3 "$script" >> "$LOG" 2>&1
    echo "[$(date)] V$next done (exit $?)" >> "$LOG"
    iter=$((iter + 1))
    sleep $SLEEP
done

echo "=== Loop done $(date) ===" >> "$LOG"
