"""
models/train_xgboost.py
========================
Train an XGBoost regressor for a player prop stat.

Supported stats: pts, reb, ast, stl, blk, fg3

Usage
-----
  python models/train_xgboost.py --stat pts
  python models/train_xgboost.py --stat fg3
  python models/train_xgboost.py --stat reb --csv data/xgboost_training_data_reb.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder


VALID_STATS = {"pts", "reb", "ast", "stl", "blk", "fg3"}

_models_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.join(_models_dir, "..")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost for one player prop stat.")
    p.add_argument("--stat", default="pts", choices=sorted(VALID_STATS))
    p.add_argument(
        "--csv",
        default="",
        help="Path to training CSV (default: data/xgboost_training_data_<stat>.csv; pts falls back to legacy data/xgboost_training_data.csv).",
    )
    p.add_argument(
        "--auto-build-data",
        action="store_true",
        help="If CSV missing, run models/build_training_data.py for this stat.",
    )
    return p.parse_args()


def resolve_csv_path(stat: str, csv_arg: str) -> str:
    if csv_arg:
        return os.path.abspath(csv_arg)
    default_stat = os.path.join(_repo_dir, "data", f"xgboost_training_data_{stat}.csv")
    if stat == "pts":
        legacy = os.path.join(_repo_dir, "data", "xgboost_training_data.csv")
        return default_stat if os.path.exists(default_stat) else legacy
    return default_stat


def ensure_csv_exists(stat: str, csv_path: str, auto_build: bool) -> None:
    if os.path.exists(csv_path):
        return
    if not auto_build:
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}\n"
            f"Generate first with: python models/build_training_data.py --stat {stat} --output \"{csv_path}\""
        )
    cmd = [
        sys.executable,
        os.path.join(_models_dir, "build_training_data.py"),
        "--stat",
        stat,
        "--output",
        csv_path,
    ]
    print(f"[build] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=_repo_dir)


def main() -> None:
    args = parse_args()
    stat = args.stat
    csv_path = resolve_csv_path(stat, args.csv)
    model_path = os.path.join(_models_dir, f"xgboost_{stat}_model.json")

    ensure_csv_exists(stat, csv_path, args.auto_build_data)

    print("Loading training data...")
    df = pd.read_csv(csv_path, parse_dates=["game_date"])
    print(f"  stat={stat}  rows={len(df):,}")

    _rot_defaults = {
        "historical_minute_overlap": 22.0,
        "usage_to_sub_ratio": 1.0,
        "coach_id": 0.0,
        "bench_reliance_factor": 0.32,
    }
    for col, val in _rot_defaults.items():
        if col not in df.columns:
            df[col] = val
            print(f"  [warn] CSV missing {col!r} — filled with {val} (re-run build_training_data.py).")

    df = df.sort_values("game_date").reset_index(drop=True)
    print(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

    stat_feat = f"last_5_avg_{stat}"
    target_col = f"target_{stat}"
    NUMERIC_FEATURES = [
        stat_feat,
        "last_5_avg_min",
        "is_home",
        "days_rest",
        "is_potential_blowout",
        "closing_spread",
        "closing_total",
        "historical_minute_overlap",
        "usage_to_sub_ratio",
        "coach_id",
        "bench_reliance_factor",
    ]
    CAT_FEATURE = "opponent_team_id"

    required = NUMERIC_FEATURES + [CAT_FEATURE, target_col]
    df = df.dropna(subset=required)
    if df.empty:
        raise RuntimeError(f"No rows remain after dropping NaNs for stat={stat}.")
    print(f"  Rows after dropping NaNs: {len(df):,}")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    team_ohe = ohe.fit(df[[CAT_FEATURE]])
    ohe_cols = [f"opp_{int(c)}" for c in ohe.categories_[0]]
    team_dummies = pd.DataFrame(
        team_ohe.transform(df[[CAT_FEATURE]]),
        columns=ohe_cols,
        index=df.index,
    )

    X = pd.concat([df[NUMERIC_FEATURES].reset_index(drop=True), team_dummies.reset_index(drop=True)], axis=1)
    y = df[target_col].reset_index(drop=True)
    print(f"  Feature matrix shape: {X.shape}")

    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print("\nTime-series split:")
    print(
        f"  Train: {len(X_train):,} rows  ({df['game_date'].iloc[:split_idx].min().date()} to {df['game_date'].iloc[:split_idx].max().date()})"
    )
    print(
        f"  Test : {len(X_test):,} rows  ({df['game_date'].iloc[split_idx:].min().date()} to {df['game_date'].iloc[split_idx:].max().date()})"
    )

    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n{'='*45}")
    print(f"  [{stat.upper()}] Test MAE : {mae:.4f}")
    print(f"{'='*45}")

    importance = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
    print("\nTop-10 feature importances:")
    for feat, score in importance.items():
        print(f"  {feat:<25} {score:.4f}")

    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
