#!/bin/bash
# Batch experiment runner for ActSVD grid search
# Reads rank_pairs.json and runs all 30 experiments sequentially

set -e  # Exit on error

echo "=========================================="
echo "ActSVD Grid Search - Batch Experiment Runner"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
RANK_PAIRS_FILE="experiments/actsvd/rank_pairs.json"

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Using Python: $(which python)"
echo ""

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

# Check if rank pairs file exists
if [ ! -f "$RANK_PAIRS_FILE" ]; then
    echo "ERROR: Rank pairs file not found: $RANK_PAIRS_FILE"
    echo "Please run: python experiments/actsvd/generate_rank_pairs.py"
    exit 1
fi

echo "Loading rank pairs from: $RANK_PAIRS_FILE"
echo ""

# Parse rank pairs and count total
TOTAL=$(python3 -c "
import json
with open('$RANK_PAIRS_FILE') as f:
    pairs = json.load(f)
print(len(pairs))
")

echo "Total experiments to run: $TOTAL"
echo ""

# Check CUDA availability
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"

echo "Starting ActSVD sweep experiments..."
echo ""

# Counter for progress
COUNTER=1
FAILED=0

# Read and process each pair
python3 -c "
import json
import sys

with open('$RANK_PAIRS_FILE') as f:
    pairs = json.load(f)

for pair in pairs:
    print(f\"{pair['ru']},{pair['rs']}\")
" | while IFS=',' read -r RU RS; do

    SAVE_DIR="out/experiments/actsvd_sweep/ru_$(printf '%04d' $RU)_rs_$(printf '%04d' $RS)"
    LOG_FILE="${SAVE_DIR}/log.txt"

    echo "=========================================="
    echo "Experiment $COUNTER/$TOTAL: r_u=$RU, r_s=$RS"
    echo "Save directory: $SAVE_DIR"
    echo "=========================================="

    # Skip if results already exist
    if [ -f "$LOG_FILE" ]; then
        echo "  Results already exist, skipping..."
        echo ""
        COUNTER=$((COUNTER + 1))
        continue
    fi

    # Run experiment
    python experiments/actsvd/run_single_actsvd.py \
        --ru $RU \
        --rs $RS \
        --model $MODEL \
        --eval_zero_shot \
        --eval_attack

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "✗ Experiment $COUNTER/$TOTAL failed: r_u=$RU, r_s=$RS"
        echo ""
        FAILED=$((FAILED + 1))
    else
        echo ""
        echo "✓ Experiment $COUNTER/$TOTAL completed: r_u=$RU, r_s=$RS"
        echo ""
    fi

    # Small delay between experiments
    sleep 2

    COUNTER=$((COUNTER + 1))
done

echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All $TOTAL ActSVD experiments completed successfully!"
else
    echo "⚠ Completed with $FAILED failures out of $TOTAL experiments"
fi
echo "=========================================="
echo ""
echo "Results saved to: out/experiments/actsvd_sweep/"
echo ""
echo "Next steps:"
echo "  1. Collect results: python experiments/actsvd/collect_actsvd_results.py"
echo "  2. Generate plots: python experiments/actsvd/plot_actsvd_results.py"
echo ""
