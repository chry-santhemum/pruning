#!/usr/bin/env python3
"""
Script to collect and parse results from diagonal sweep experiments (P=Q).
Reads log files from each experiment and creates a summary CSV.
"""

import os
import json
import pandas as pd
import re
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse the log_wandg_set_difference.txt file.

    Expected format:
    method  actual_sparsity  p  q  metric  score
    wandg_set_difference  0.500000  0.01  0.01  inst_ASR_basic  0.5400
    """
    results = {}

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            method, sparsity, p, q, metric, score = parts[:6]

            # Store results by metric type
            metric_name = metric.strip()
            score_value = float(score.strip())
            results[metric_name] = score_value

    return results

def collect_results(base_dir="out/experiments/diagonal_sweep"):
    """
    Collect results from all P=Q experiments.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None

    # List to store all results
    all_results = []

    # P=Q values from 0.01 to 0.10
    pq_values = [i / 100.0 for i in range(1, 11)]

    for pq in pq_values:
        exp_dir = base_path / f"p_{pq:.2f}_q_{pq:.2f}"
        log_file = exp_dir / "log_wandg_set_difference.txt"

        print(f"Processing P=Q={pq:.2f}...")

        if not exp_dir.exists():
            print(f"  Warning: Directory not found: {exp_dir}")
            continue

        # Parse log file
        metrics = parse_log_file(log_file)

        if metrics is None:
            print(f"  Warning: Could not parse results for P=Q={pq:.2f}")
            continue

        # Create result entry
        result = {
            'P': pq,
            'Q': pq,
            'P_Q': pq,
        }

        # Add all metrics found in the log
        result.update(metrics)

        all_results.append(result)

        # Print summary
        vanilla_asr = metrics.get('inst_ASR_basic', metrics.get('no_inst_ASR_basic', 'N/A'))
        print(f"  ✓ P=Q={pq:.2f}: Vanilla ASR = {vanilla_asr}")

    if not all_results:
        print("Error: No results collected!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by P_Q
    df = df.sort_values('P_Q')

    return df

def main():
    print("=" * 60)
    print("Collecting Diagonal Sweep Results (P=Q)")
    print("=" * 60)
    print()

    # Collect results
    df = collect_results()

    if df is None:
        print("Failed to collect results.")
        return

    # Save to CSV
    output_file = "out/experiments/diagonal_sweep/results_diagonal.csv"
    df.to_csv(output_file, index=False)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()
    print(df.to_string(index=False))
    print()
    print(f"Results saved to: {output_file}")
    print()

    # Print vanilla ASR trend if available
    if 'inst_ASR_basic' in df.columns:
        print("=" * 60)
        print("Vanilla ASR Trend (inst_ASR_basic)")
        print("=" * 60)
        print()
        for _, row in df.iterrows():
            pq = row['P_Q']
            asr = row['inst_ASR_basic']
            bar = '█' * int(asr * 50)  # Visual bar
            print(f"P=Q={pq:05.2f}: {asr:6.4f} {bar}")
        print()

if __name__ == "__main__":
    main()
