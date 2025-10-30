#!/bin/bash
# Script to dump SNIP scores for both utility and safety datasets
# These scores are required for wandg_set_difference pruning method

set -e  # Exit on error

echo "=========================================="
echo "SNIP Score Dumping for Set Difference"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
METHOD="wandg"  # SNIP method in the codebase
TYPE="unstructured"
SPARSITY_RATIO=0.5

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Using Python: $(which python)"
echo ""

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "[1/2] Dumping SNIP scores for UTILITY dataset (alpaca_cleaned_no_safety)..."
echo "----------------------------------------------------------------------"
python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data alpaca_cleaned_no_safety \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $TYPE \
    --save out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/ \
    --dump_wanda_score

echo ""
echo "[2/2] Dumping SNIP scores for SAFETY dataset (align)..."
echo "----------------------------------------------------------------------"
python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data align \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $TYPE \
    --save out/llama2-7b-chat-hf/unstructured/wandg/align/ \
    --dump_wanda_score

echo ""
echo "=========================================="
echo "✓ Score dumping completed successfully!"
echo "=========================================="
echo ""
echo "Scores saved to:"
echo "  - out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/"
echo "  - out/llama2-7b-chat-hf/unstructured/wandg/align/"
echo ""
echo "Next step: Run diagonal sweep experiments"
echo "  bash experiments/run_diagonal_sweep.sh"
echo ""
