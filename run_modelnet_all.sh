#!/bin/bash

METHODS=(SinkhornCPD RPOT FilterReg CPD FGR TEASER++ Sparse-ICP RobOT LSG-CPD)
MASTER=/tmp/modelnet_all.log

echo "=== ModelNet40 full run started at $(date) ===" > "$MASTER"

for m in "${METHODS[@]}"; do
    SAFE=$(echo "$m" | tr '+/' '_')
    LOG=/tmp/modelnet_${SAFE}.log
    echo "[$(date +%H:%M:%S)] Starting $m -> $LOG" | tee -a "$MASTER"
    python -u -m experiments.modelnet --method "$m" > "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] Done $m: $(tail -1 $LOG)" | tee -a "$MASTER"
done

echo "=== All done at $(date) ===" | tee -a "$MASTER"
