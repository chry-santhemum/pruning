#!/usr/bin/env python3
"""
Single ActSVD experiment wrapper.
Runs one ActSVD (orthogonal projection) experiment for given (r_u, r_s) values.
Wraps main_low_rank_diff.py with standardized directory structure.

Usage:
    python run_single_actsvd.py --ru 100 --rs 100
    python run_single_actsvd.py --ru 500 --rs 1000 --eval_zero_shot --eval_attack
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_actsvd_experiment(ru, rs, model, eval_zero_shot, eval_attack):
    """
    Run a single ActSVD experiment.

    Args:
        ru: Utility rank (r_u)
        rs: Safety rank (r_s)
        model: Model name
        eval_zero_shot: Whether to evaluate zero-shot accuracy
        eval_attack: Whether to evaluate attack success rate
    """
    # Navigate to project root
    project_root = Path(__file__).resolve().parents[2]

    # Format save directory
    save_dir = project_root / f"out/experiments/actsvd_sweep/ru_{ru:04d}_rs_{rs:04d}"

    print("=" * 60)
    print(f"ActSVD Experiment: r_u={ru}, r_s={rs}")
    print("=" * 60)
    print()
    print(f"Configuration:")
    print(f"  Model: {model}")
    print(f"  Utility rank (r_u): {ru}")
    print(f"  Safety rank (r_s): {rs}")
    print(f"  Utility dataset: alpaca_cleaned_no_safety")
    print(f"  Safety dataset: align")
    print(f"  Save directory: {save_dir}")
    print()

    # Check if results already exist
    log_file = save_dir / "log.txt"
    if log_file.exists():
        print(f"✓ Results already exist: {log_file}")
        print("  Skipping experiment (use --force to override)")
        return True

    # Build command
    cmd = [
        "python",
        str(project_root / "main_low_rank_diff.py"),
        "--model", model,
        "--rank_pos", str(ru),
        "--rank_neg", str(rs),
        "--prune_data_pos", "alpaca_cleaned_no_safety",
        "--prune_data_neg", "align",
        "--save", str(save_dir),
    ]

    if eval_zero_shot:
        cmd.append("--eval_zero_shot")

    if eval_attack:
        cmd.append("--eval_attack")

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=True,
            capture_output=False  # Show output in real-time
        )
        print()
        print("=" * 60)
        print(f"✓ Experiment completed: r_u={ru}, r_s={rs}")
        print("=" * 60)
        print()
        return True

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"✗ Experiment failed: r_u={ru}, r_s={rs}")
        print(f"  Exit code: {e.returncode}")
        print("=" * 60)
        print()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run single ActSVD experiment"
    )
    parser.add_argument(
        "--ru",
        type=int,
        required=True,
        help="Utility rank (r_u)"
    )
    parser.add_argument(
        "--rs",
        type=int,
        required=True,
        help="Safety rank (r_s)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama2-7b-chat-hf",
        choices=["llama2-7b-chat-hf", "llama2-13b-chat-hf"],
        help="Model to use"
    )
    parser.add_argument(
        "--eval_zero_shot",
        action="store_true",
        help="Evaluate zero-shot accuracy"
    )
    parser.add_argument(
        "--eval_attack",
        action="store_true",
        help="Evaluate attack success rate"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results exist"
    )

    args = parser.parse_args()

    # Run experiment
    success = run_actsvd_experiment(
        ru=args.ru,
        rs=args.rs,
        model=args.model,
        eval_zero_shot=args.eval_zero_shot,
        eval_attack=args.eval_attack
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
