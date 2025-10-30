#!/usr/bin/env python3
"""
Generate (r_u, r_s) rank pairs for ActSVD grid search experiments.
Creates 30 pairs with log spacing from 50 to 4000.
Saves to rank_pairs.json for use by sweep script.
"""

import json
import numpy as np
from pathlib import Path

def generate_log_spaced_ranks(min_rank=50, max_rank=4000, num_points=30):
    """
    Generate log-spaced rank values.

    Log spacing provides denser sampling at lower ranks where
    changes have more impact on model behavior.

    Args:
        min_rank: Minimum rank value
        max_rank: Maximum rank value
        num_points: Number of points to generate

    Returns:
        List of integer rank values
    """
    ranks = np.logspace(
        np.log10(min_rank),
        np.log10(max_rank),
        num_points
    )
    # Round to integers and ensure uniqueness
    ranks = sorted(set([int(round(r)) for r in ranks]))

    # Ensure we have exactly num_points
    if len(ranks) < num_points:
        # If rounding caused duplicates, fill in with linear spacing
        additional = num_points - len(ranks)
        linear_fill = np.linspace(min_rank, max_rank, additional + 2)[1:-1]
        ranks.extend([int(r) for r in linear_fill])
        ranks = sorted(set(ranks))[:num_points]

    return ranks[:num_points]

def generate_diagonal_pairs(ranks):
    """
    Generate diagonal (r_u, r_s) pairs where r_u = r_s.

    Args:
        ranks: List of rank values

    Returns:
        List of dictionaries with 'ru' and 'rs' keys
    """
    pairs = []
    for rank in ranks:
        pairs.append({
            'ru': rank,
            'rs': rank
        })
    return pairs

def main():
    print("=" * 60)
    print("ActSVD Rank Pair Generator")
    print("=" * 60)
    print()

    # Configuration
    MIN_RANK = 50
    MAX_RANK = 4000
    NUM_POINTS = 30

    print(f"Configuration:")
    print(f"  Min rank: {MIN_RANK}")
    print(f"  Max rank: {MAX_RANK}")
    print(f"  Number of points: {NUM_POINTS}")
    print(f"  Spacing: Log-spaced (denser at lower ranks)")
    print()

    # Generate ranks
    print("Generating log-spaced rank values...")
    ranks = generate_log_spaced_ranks(MIN_RANK, MAX_RANK, NUM_POINTS)

    print(f"✓ Generated {len(ranks)} rank values")
    print()

    # Show rank distribution
    print("Rank distribution:")
    print(f"  First 10: {ranks[:10]}")
    print(f"  Middle 10: {ranks[10:20]}")
    print(f"  Last 10: {ranks[20:]}")
    print()

    # Generate (r_u, r_s) pairs
    print("Generating diagonal (r_u = r_s) pairs...")
    pairs = generate_diagonal_pairs(ranks)

    print(f"✓ Generated {len(pairs)} pairs")
    print()

    # Show first few pairs
    print("First 5 pairs:")
    for i, pair in enumerate(pairs[:5], 1):
        print(f"  {i}. r_u={pair['ru']:4d}, r_s={pair['rs']:4d}")
    print()

    print("Last 5 pairs:")
    for i, pair in enumerate(pairs[-5:], len(pairs)-4):
        print(f"  {i}. r_u={pair['ru']:4d}, r_s={pair['rs']:4d}")
    print()

    # Save to JSON
    output_dir = Path(__file__).parent
    output_file = output_dir / "rank_pairs.json"

    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)

    print("=" * 60)
    print("✓ Rank pairs saved!")
    print("=" * 60)
    print()
    print(f"Output file: {output_file}")
    print(f"Total pairs: {len(pairs)}")
    print()
    print("Next step: Run sweep experiments")
    print("  bash experiments/actsvd/run_actsvd_sweep.sh")
    print()

if __name__ == "__main__":
    main()
