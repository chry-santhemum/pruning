#!/bin/bash
# Quick test script to verify ActSVD pipeline works
# Runs one lightweight experiment (r_u=100, r_s=100) and tests full pipeline

set -e  # Exit on error

echo "=========================================="
echo "ActSVD Pipeline Test Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Generate rank pairs (just creates JSON)"
echo "  2. Run ONE test experiment: r_u=100, r_s=100"
echo "  3. Collect results (1 experiment)"
echo "  4. Generate plots (1 data point)"
echo ""
echo "Estimated runtime: ~15-20 minutes"
echo ""
echo "Starting test..."
echo ""

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Using Python: $(which python)"
echo ""

# Disable hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=0

# Step 1: Generate rank pairs
echo "=========================================="
echo "[STEP 1/4] Generating rank pairs..."
echo "=========================================="
echo ""

python experiments/actsvd/generate_rank_pairs.py

echo ""

# Step 2: Run single test experiment
echo "=========================================="
echo "[STEP 2/4] Running test experiment: r_u=100, r_s=100"
echo "=========================================="
echo ""

python experiments/actsvd/run_single_actsvd.py \
    --ru 100 \
    --rs 100 \
    --model llama2-7b-chat-hf \
    --eval_zero_shot \
    --eval_attack \
    --force

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ Test experiment failed!"
    echo "Check logs above for errors."
    exit $EXIT_CODE
fi

echo ""
echo "✓ Test experiment completed"
echo ""

# Step 3: Collect results
echo "=========================================="
echo "[STEP 3/4] Collecting results..."
echo "=========================================="
echo ""

python experiments/actsvd/collect_actsvd_results.py

echo ""

# Step 4: Generate plots
echo "=========================================="
echo "[STEP 4/4] Generating plots..."
echo "=========================================="
echo ""

python experiments/actsvd/plot_actsvd_results.py

echo ""
echo "=========================================="
echo "✓ PIPELINE TEST COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Test results:"
echo "  - Experiment: out/experiments/actsvd_sweep/ru_0100_rs_0100/"
echo "  - CSV: out/experiments/actsvd_sweep/results_actsvd.csv"
echo "  - Plots: out/experiments/actsvd_sweep/plot_*.png"
echo ""
echo "Pipeline is working! You can now run the full sweep:"
echo "  bash experiments/run_actsvd_pipeline.sh"
echo ""
