#!/usr/bin/env python3
"""
Plot ActSVD grid search results: Zero-shot Accuracy vs ASR metrics.
Generates 3 PNG plots for Vanilla ASR, Adv-Suffix ASR, and Adv-Decoding ASR.
Uses same styling as diagonal/custom P,Q plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def create_plot(df, x_col, y_col, ylabel, output_file):
    """
    Create a scatter plot with triangle markers.

    Args:
        df: DataFrame with results
        x_col: Column name for x-axis (zero-shot accuracy)
        y_col: Column name for y-axis (ASR metric)
        ylabel: Y-axis label (with LaTeX subscripts)
        output_file: Output PNG file path
    """
    # Wider and shorter figure (more rectangular)
    plt.figure(figsize=(10, 4))

    # Custom color RGB(25, 197, 132) converted to hex
    color = '#19C584'

    # Create scatter plot with triangle markers, no lines
    plt.scatter(df[x_col], df[y_col], marker='^',
                s=150, color=color, edgecolors='none')

    # Set axis limits with extra space at ends
    plt.xlim(0.33, 0.62)
    plt.ylim(0, 1.0)

    # Set x-axis ticks from 0.35 to 0.6
    plt.xticks(np.arange(0.35, 0.61, 0.05), fontsize=16)
    plt.yticks(fontsize=16)

    # Labels - larger font, no bold, no title
    plt.xlabel('Zero-shot Accuracy', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_file}")

    # Close to free memory
    plt.close()

def main():
    print("=" * 60)
    print("Plotting ActSVD Grid Search Results")
    print("=" * 60)
    print()

    # Load data
    csv_file = "out/experiments/actsvd_sweep/results_actsvd.csv"
    print(f"Loading data from: {csv_file}")

    if not Path(csv_file).exists():
        print(f"ERROR: Results file not found: {csv_file}")
        print("Please run: python experiments/actsvd/collect_actsvd_results.py")
        return

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} data points")
    print()

    # Check required columns
    required_cols = ['averaged', 'inst_ASR_basic', 'ASR_gcg', 'inst_ASR_multiple_nosys']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        print()
        print("Make sure experiments include both --eval_zero_shot and --eval_attack")
        return

    # Output directory
    output_dir = Path("out/experiments/actsvd_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    print()

    # Plot 1: Vanilla ASR
    create_plot(
        df=df,
        x_col='averaged',
        y_col='inst_ASR_basic',
        ylabel='$\\mathrm{ASR_{Vanilla}}$',
        output_file=output_dir / 'plot_vanilla_asr.png'
    )

    # Plot 2: Adv-Suffix ASR (GCG)
    create_plot(
        df=df,
        x_col='averaged',
        y_col='ASR_gcg',
        ylabel='$\\mathrm{ASR_{Adv\\text{-}Suffix}}$',
        output_file=output_dir / 'plot_adv_suffix_asr.png'
    )

    # Plot 3: Adv-Decoding ASR (multiple sampling)
    create_plot(
        df=df,
        x_col='averaged',
        y_col='inst_ASR_multiple_nosys',
        ylabel='$\\mathrm{ASR_{Adv\\text{-}Decoding}}$',
        output_file=output_dir / 'plot_adv_decoding_asr.png'
    )

    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - {output_dir / 'plot_vanilla_asr.png'}")
    print(f"  - {output_dir / 'plot_adv_suffix_asr.png'}")
    print(f"  - {output_dir / 'plot_adv_decoding_asr.png'}")
    print()

if __name__ == "__main__":
    main()
