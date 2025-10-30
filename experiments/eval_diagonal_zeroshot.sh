#!/bin/bash
# Script to evaluate zero-shot accuracy for diagonal sweep experiments
# Re-prunes models and runs zero-shot evaluation only (skips attack eval)

set -e  # Exit on error

echo "=========================================="
echo "Zero-Shot Evaluation - Diagonal Sweep"
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
if [ ! -d "out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety" ] || [ ! -d "out/llama2-7b-chat-hf/unstructured/wandg/align" ]; then
    echo "ERROR: SNIP scores not found!"
    echo "Please run 'bash experiments/dump_scores.sh' first"
    exit 1
fi

echo "Starting zero-shot evaluation for diagonal sweep..."
echo ""

# Loop through P=Q values from 0.01 to 0.10
for i in {1..10}; do
    # Calculate P and Q value (e.g., 1 -> 0.01, 2 -> 0.02, etc.)
    PQ=$(printf "0.%02d" $i)

    SAVE_DIR="out/experiments/diagonal_sweep/p_${PQ}_q_${PQ}"
    ZEROSHOT_LOG="${SAVE_DIR}/log_zeroshot.txt"

    echo "=========================================="
    echo "Experiment $i/10: P=$PQ, Q=$PQ"
    echo "Save directory: $SAVE_DIR"
    echo "=========================================="

    # Skip if zero-shot results already exist
    if [ -f "$ZEROSHOT_LOG" ]; then
        echo "  Zero-shot results already exist, skipping..."
        echo ""
        continue
    fi

    # Run pruning and zero-shot evaluation only (no attack eval)
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --p $PQ \
        --q $PQ \
        --sparsity_type $TYPE \
        --save $SAVE_DIR \
        --eval_zero_shot

    echo ""
    echo "✓ Experiment $i/10 completed: P=$PQ, Q=$PQ"
    echo ""

    # Small delay between experiments
    sleep 2
done

echo "=========================================="
echo "✓ All 10 zero-shot evaluations completed!"
echo "=========================================="
echo ""
echo "Results saved to: out/experiments/diagonal_sweep/p_*/log_zeroshot.txt"
echo ""
echo "Next step: Collect all results"
echo "  python experiments/collect_diagonal_results.py"
echo ""
