#!/bin/bash
# Master script to run the complete ActSVD grid search pipeline in tmux
# Runs: generate pairs → sweep experiments → collect results → generate plots
# All in a detached tmux session for background execution

TMUX_SESSION="actsvd_experiments"
LOG_FILE="experiments/actsvd_pipeline.log"
PROJECT_DIR="/workspace/projects/2881r-mini-project"

echo "=========================================="
echo "ActSVD Grid Search Pipeline Launcher"
echo "=========================================="
echo ""

# Check if tmux session already exists
if tmux has-session -t $TMUX_SESSION 2>/dev/null; then
    echo "ERROR: Tmux session '$TMUX_SESSION' already exists!"
    echo ""
    echo "Options:"
    echo "  1. Attach to existing session:"
    echo "     tmux attach -t $TMUX_SESSION"
    echo ""
    echo "  2. Kill existing session and start new:"
    echo "     tmux kill-session -t $TMUX_SESSION"
    echo "     bash experiments/run_actsvd_pipeline.sh"
    echo ""
    exit 1
fi

echo "Creating detached tmux session: $TMUX_SESSION"
echo "Log file: $LOG_FILE"
echo ""
echo "This will run 30 ActSVD experiments (~10-15 hours total)"
echo ""

# Create detached tmux session and run the pipeline
tmux new-session -d -s $TMUX_SESSION -c $PROJECT_DIR "
    set -e

    # Redirect all output to log file
    exec > >(tee $LOG_FILE) 2>&1

    echo '=========================================='
    echo 'ActSVD Grid Search Pipeline'
    echo 'Started: '\$(date)
    echo '=========================================='
    echo ''

    # Step 1: Generate rank pairs
    echo '[STEP 1/4] Generating rank pairs...'
    echo '------------------------------------------------'
    source .venv/bin/activate
    python experiments/actsvd/generate_rank_pairs.py
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Rank pair generation failed with exit code '\$EXIT_CODE
        echo 'Check log file for details: $LOG_FILE'
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Rank pairs generated'
    echo ''

    # Step 2: Run all experiments
    echo '[STEP 2/4] Running ActSVD sweep experiments (30 total)...'
    echo '------------------------------------------------'
    bash experiments/actsvd/run_actsvd_sweep.sh
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Experiment sweep failed with exit code '\$EXIT_CODE
        echo 'Check log file for details: $LOG_FILE'
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ All experiments completed'
    echo ''

    # Step 3: Collect results
    echo '[STEP 3/4] Collecting and parsing results...'
    echo '------------------------------------------------'
    python experiments/actsvd/collect_actsvd_results.py
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Result collection failed with exit code '\$EXIT_CODE
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Results collected'
    echo ''

    # Step 4: Generate plots
    echo '[STEP 4/4] Generating plots...'
    echo '------------------------------------------------'
    python experiments/actsvd/plot_actsvd_results.py
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Plot generation failed with exit code '\$EXIT_CODE
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Plots generated'
    echo ''

    # Final summary
    echo '=========================================='
    echo '✓ PIPELINE COMPLETED SUCCESSFULLY!'
    echo 'Finished: '\$(date)
    echo '=========================================='
    echo ''
    echo 'Results:'
    echo '  CSV: out/experiments/actsvd_sweep/results_actsvd.csv'
    echo '  Plots: out/experiments/actsvd_sweep/plot_*.png'
    echo ''
    echo 'Log file: $LOG_FILE'
    echo ''
    echo 'Tmux session will remain active for 30 seconds...'
    sleep 30
    echo 'Session ending. Press Enter to exit or Ctrl+B D to detach.'
"

echo "=========================================="
echo "✓ Pipeline launched in tmux session!"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  - Attach to session:  tmux attach -t $TMUX_SESSION"
echo "  - View log file:      tail -f $LOG_FILE"
echo "  - List sessions:      tmux ls"
echo ""
echo "Check progress:"
echo "  - Count completed:    ls -d out/experiments/actsvd_sweep/ru_*/log.txt 2>/dev/null | wc -l"
echo "  - View results:       cat out/experiments/actsvd_sweep/results_actsvd.csv"
echo ""
echo "Kill session if needed:"
echo "  tmux kill-session -t $TMUX_SESSION"
echo ""
