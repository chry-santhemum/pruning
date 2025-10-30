#!/usr/bin/env python3
"""
Collect and parse results from ActSVD grid search experiments.
Reads log.txt files from each experiment and creates results_actsvd.csv.

ActSVD log format:
    rank    INST    metric    score
    50_50   1       PPL       10.5
    50_50   inst_   ASR_basic 0.85
"""

import os
import re
import pandas as pd
from pathlib import Path

def parse_actsvd_log(log_path):
    """
    Parse the log.txt file from an ActSVD experiment.

    Format:
        rank\tINST\tmetric\tscore
        50_50\t1\tPPL\t10.5
        50_50\tinst_\tASR_basic\t0.85

    Args:
        log_path: Path to log.txt file

    Returns:
        Dictionary of metric_name -> score
    """
    results = {}

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Skip header if present
    start_idx = 1 if lines and 'rank' in lines[0].lower() else 0

    for line in lines[start_idx:]:
        parts = line.strip().split('\t')
        if len(parts) >= 4:
            rank_str, inst, metric, score = parts[:4]

            # Build metric name
            # inst can be: '1', 'inst_', 'no_inst_', etc.
            if inst and inst != '1':
                metric_name = f"{inst}{metric}"
            else:
                metric_name = metric

            try:
                score_value = float(score.strip())
                results[metric_name] = score_value
            except ValueError:
                print(f"Warning: Could not parse score '{score}' for metric '{metric_name}'")
                continue

    return results

def extract_ranks_from_dirname(dirname):
    """
    Extract r_u and r_s from directory name.

    Expected format: ru_XXXX_rs_YYYY

    Args:
        dirname: Directory name

    Returns:
        Tuple of (r_u, r_s) or (None, None) if parsing fails
    """
    match = re.match(r'ru_(\d+)_rs_(\d+)', dirname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def collect_results(base_dir="out/experiments/actsvd_sweep"):
    """
    Collect results from all ActSVD experiments.

    Args:
        base_dir: Base directory containing experiment subdirectories

    Returns:
        DataFrame with all results
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None

    # List to store all results
    all_results = []

    # Find all experiment directories
    exp_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('ru_')])

    if not exp_dirs:
        print(f"Error: No experiment directories found in {base_dir}")
        return None

    print(f"Found {len(exp_dirs)} experiment directories")
    print()

    for exp_dir in exp_dirs:
        # Extract ranks from directory name
        ru, rs = extract_ranks_from_dirname(exp_dir.name)

        if ru is None or rs is None:
            print(f"Warning: Could not parse directory name: {exp_dir.name}")
            continue

        log_file = exp_dir / "log.txt"

        print(f"Processing r_u={ru:4d}, r_s={rs:4d}...", end=" ")

        # Parse log file
        metrics = parse_actsvd_log(log_file)

        if metrics is None:
            print("✗ (log not found)")
            continue

        if not metrics:
            print("✗ (no metrics found)")
            continue

        # Create result entry
        result = {
            'r_u': ru,
            'r_s': rs,
        }

        # Add all metrics
        result.update(metrics)

        all_results.append(result)

        # Print summary
        vanilla_asr = metrics.get('inst_ASR_basic', 'N/A')
        zeroshot_acc = metrics.get('averaged', 'N/A')
        print(f"✓ (ASR={vanilla_asr}, Acc={zeroshot_acc})")

    if not all_results:
        print()
        print("Error: No results collected!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by r_u, then r_s
    df = df.sort_values(['r_u', 'r_s'])

    return df

def main():
    print("=" * 60)
    print("Collecting ActSVD Grid Search Results")
    print("=" * 60)
    print()

    # Collect results
    df = collect_results()

    if df is None:
        print("Failed to collect results.")
        return

    print()
    print("=" * 60)
    print(f"✓ Collected {len(df)} results")
    print("=" * 60)
    print()

    # Save to CSV
    output_file = "out/experiments/actsvd_sweep/results_actsvd.csv"
    df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print()

    # Display summary
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()

    # Show key columns if available
    display_cols = ['r_u', 'r_s', 'PPL']

    # Add ASR metrics if available
    if 'inst_ASR_basic' in df.columns:
        display_cols.append('inst_ASR_basic')
    if 'ASR_gcg' in df.columns:
        display_cols.append('ASR_gcg')
    if 'averaged' in df.columns:
        display_cols.append('averaged')

    # Display subset of columns
    available_cols = [col for col in display_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

    print()

    # Show Vanilla ASR trend if available
    if 'inst_ASR_basic' in df.columns and 'r_u' in df.columns:
        print("=" * 60)
        print("Vanilla ASR Trend (inst_ASR_basic)")
        print("=" * 60)
        print()
        for _, row in df.head(10).iterrows():
            ru = row['r_u']
            rs = row['r_s']
            asr = row['inst_ASR_basic']
            bar = '█' * int(asr * 50)  # Visual bar
            print(f"r_u={ru:4d}, r_s={rs:4d}: {asr:6.4f} {bar}")

        if len(df) > 10:
            print(f"  ... ({len(df) - 10} more rows)")
        print()

    print("=" * 60)
    print("Next step: Generate plots")
    print("  python experiments/actsvd/plot_actsvd_results.py")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
