#!/bin/bash
# Script to run SNIP set difference experiments for custom P,Q pairs
# Custom pairs: (1,1), (2,1), (4,2), (7,3), (3,2), (4,4), (5,5), (6,5), (6,6), (9,8)

set -e  # Exit on error

echo "=========================================="
echo "SNIP Set Difference - Custom P,Q Sweep"
echo "10 custom (P,Q) pairs for Llama-2-7B"
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

echo "Starting custom P,Q sweep experiments..."
echo ""

# Define custom (P,Q) pairs as percentages
declare -a PQ_PAIRS=(
    "1 1"
    "2 1"
    "4 2"
    "7 3"
    "3 2"
    "4 4"
    "5 5"
    "6 5"
    "6 6"
    "9 8"
)

# Total number of experiments
TOTAL=${#PQ_PAIRS[@]}
COUNTER=1

# Loop through custom P,Q pairs
for pair in "${PQ_PAIRS[@]}"; do
    # Parse P and Q
    read -r P_PCT Q_PCT <<< "$pair"

    # Convert percentages to decimals (e.g., 1 -> 0.01, 10 -> 0.10)
    P=$(printf "0.%02d" $P_PCT)
    Q=$(printf "0.%02d" $Q_PCT)

    SAVE_DIR="out/experiments/custom_pq_sweep/p_${P}_q_${Q}"
    LOG_FILE="${SAVE_DIR}/log_wandg_set_difference.txt"

    echo "=========================================="
    echo "Experiment $COUNTER/$TOTAL: P=$P ($P_PCT%), Q=$Q ($Q_PCT%)"
    echo "Save directory: $SAVE_DIR"
    echo "=========================================="

    # Skip if results already exist
    if [ -f "$LOG_FILE" ]; then
        echo "  Results already exist, skipping..."
        echo ""
        COUNTER=$((COUNTER + 1))
        continue
    fi

    # Run pruning, zero-shot eval, and ASR evaluation
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --p $P \
        --q $Q \
        --sparsity_type $TYPE \
        --save $SAVE_DIR \
        --eval_zero_shot \
        --eval_attack \
        --save_attack_res

    echo ""
    echo "✓ Experiment $COUNTER/$TOTAL completed: P=$P, Q=$Q"
    echo ""

    # Small delay between experiments
    sleep 2

    COUNTER=$((COUNTER + 1))
done

echo "=========================================="
echo "✓ All $TOTAL custom P,Q experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: out/experiments/custom_pq_sweep/"
echo ""
echo "Next step: Collect results"
echo "  python experiments/collect_custom_pq_results.py"
echo ""
