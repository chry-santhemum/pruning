#!/bin/bash
# Script to run SNIP set difference experiments for P=Q diagonal sweep
# P and Q range from 0.01 to 0.10 (1% to 10%)

set -e  # Exit on error

echo "=========================================="
echo "SNIP Set Difference - Diagonal Sweep"
echo "P=Q from 0.01 to 0.10 (1% to 10%)"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
METHOD="wandg_set_difference"
TYPE="unstructured"
SPARSITY_RATIO=0.5
PRUNE_DATA="align"  # Safety dataset

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Using Python: $(which python)"
echo ""

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

# Check if scores exist
if [ ! -d "out/experiments/diagonal_sweep/scores/utility" ] || [ ! -d "out/experiments/diagonal_sweep/scores/safety" ]; then
    echo "ERROR: SNIP scores not found!"
    echo "Please run 'bash experiments/dump_scores.sh' first"
    exit 1
fi

echo "Starting diagonal sweep experiments..."
echo ""

# Loop through P=Q values from 0.01 to 0.10
for i in {1..10}; do
    # Calculate P and Q value (e.g., 1 -> 0.01, 2 -> 0.02, etc.)
    PQ=$(printf "0.%02d" $i)

    # Format for directory name (e.g., 0.01 -> 001, 0.10 -> 010)
    PQ_DIR=$(printf "%03d" $i)

    SAVE_DIR="out/experiments/diagonal_sweep/p_${PQ}_q_${PQ}"

    echo "=========================================="
    echo "Experiment $i/10: P=$PQ, Q=$PQ"
    echo "Save directory: $SAVE_DIR"
    echo "=========================================="

    # Run pruning, zero-shot eval, and ASR evaluation
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --p $PQ \
        --q $PQ \
        --sparsity_type $TYPE \
        --save $SAVE_DIR \
        --eval_zero_shot \
        --eval_attack \
        --save_attack_res

    echo ""
    echo "✓ Experiment $i/10 completed: P=$PQ, Q=$PQ"
    echo ""

    # Small delay between experiments
    sleep 2
done

echo "=========================================="
echo "✓ All 10 experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: out/experiments/diagonal_sweep/"
echo ""
echo "Next step: Parse results"
echo "  python experiments/collect_diagonal_results.py"
echo ""
