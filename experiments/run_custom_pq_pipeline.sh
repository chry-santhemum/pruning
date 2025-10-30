#!/bin/bash
# Master script to run the complete custom P,Q experiment pipeline in tmux
# Runs: experiments → collect results → generate plots
# All in a detached tmux session for background execution

TMUX_SESSION="custom_pq_experiments"
LOG_FILE="experiments/custom_pq_pipeline.log"
PROJECT_DIR="/workspace/projects/2881r-mini-project"

echo "=========================================="
echo "Custom P,Q Experiment Pipeline Launcher"
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
    echo "     bash experiments/run_custom_pq_pipeline.sh"
    echo ""
    exit 1
fi

echo "Creating detached tmux session: $TMUX_SESSION"
echo "Log file: $LOG_FILE"
echo ""

# Create detached tmux session and run the pipeline
tmux new-session -d -s $TMUX_SESSION -c $PROJECT_DIR "
    set -e

    # Redirect all output to log file
    exec > >(tee $LOG_FILE) 2>&1

    echo '=========================================='
    echo 'Custom P,Q Experiment Pipeline'
    echo 'Started: '\$(date)
    echo '=========================================='
    echo ''

    # Step 1: Run experiments
    echo '[STEP 1/3] Running custom P,Q sweep experiments...'
    echo '------------------------------------------------'
    bash experiments/run_custom_pq_sweep.sh
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Experiment sweep failed with exit code '\$EXIT_CODE
        echo 'Check log file for details: $LOG_FILE'
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Experiments completed successfully!'
    echo ''

    # Step 2: Collect results
    echo '[STEP 2/3] Collecting and parsing results...'
    echo '------------------------------------------------'
    source .venv/bin/activate
    python experiments/collect_custom_pq_results.py
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Result collection failed with exit code '\$EXIT_CODE
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Results collected successfully!'
    echo ''

    # Step 3: Generate plots
    echo '[STEP 3/3] Generating plots...'
    echo '------------------------------------------------'
    python experiments/plot_custom_pq_results.py
    EXIT_CODE=\$?

    if [ \$EXIT_CODE -ne 0 ]; then
        echo ''
        echo 'ERROR: Plot generation failed with exit code '\$EXIT_CODE
        exit \$EXIT_CODE
    fi

    echo ''
    echo '✓ Plots generated successfully!'
    echo ''

    # Final summary
    echo '=========================================='
    echo '✓ PIPELINE COMPLETED SUCCESSFULLY!'
    echo 'Finished: '\$(date)
    echo '=========================================='
    echo ''
    echo 'Results:'
    echo '  CSV: out/experiments/custom_pq_sweep/results_custom_pq.csv'
    echo '  Plots: out/experiments/custom_pq_sweep/plot_*.png'
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
echo "Kill session if needed:"
echo "  tmux kill-session -t $TMUX_SESSION"
echo ""
