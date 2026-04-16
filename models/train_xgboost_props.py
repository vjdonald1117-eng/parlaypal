"""
Train XGBoost models for multiple player prop stats.

Usage:
  python models/train_xgboost_props.py
  python models/train_xgboost_props.py --stats pts reb ast stl blk fg3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


DEFAULT_STATS = ["pts", "reb", "ast", "stl", "blk", "fg3"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost prop models for many stats.")
    p.add_argument("--stats", nargs="+", default=DEFAULT_STATS, help="Stats to train.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    train_one = os.path.join(this_dir, "train_xgboost.py")
    repo_dir = os.path.join(this_dir, "..")

    for stat in args.stats:
        cmd = [sys.executable, train_one, "--stat", stat, "--auto-build-data"]
        print(f"\n[train:{stat}] {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=repo_dir)

    print("\n[OK] Finished training requested prop models.")


if __name__ == "__main__":
    main()

