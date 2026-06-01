#!/bin/bash

METHODS="CPD FilterReg FGR RPOT RobOT TEASER++ Sparse-ICP LSG-CPD"
AXES="noise outlier overlap rotation"

for method in $METHODS; do
    for axis in $AXES; do
        echo ""
        echo "========================================"
        echo "  $(date): $method / $axis"
        echo "========================================"
        python -m experiments.bunny --axis $axis --method "$method"
    done
done

echo ""
echo "$(date): ALL DONE"
