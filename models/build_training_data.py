"""
models/build_training_data.py
==============================
Phase 1 of the XGBoost ML pipeline: Feature Engineering.

Pulls historical box scores from PostgreSQL, computes rolling/contextual
features per player BEFORE each game, and saves a flat CSV ready for
XGBoost training.

Usage
-----
  python models/build_training_data.py
  python models/build_training_data.py --stat reb
  python models/build_training_data.py --stat ast
  python models/build_training_data.py --stat pts --output data/pts_training.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from models.rotation_coaching_features import apply_rotation_coaching_features

VALID_STATS = {"pts", "reb", "ast", "stl", "blk", "fg3"}
DEFAULT_OUTPUT = os.path.join(str(_REPO_ROOT), "data", "xgboost_training_data.csv")
STAT_SQL_COL = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "fg3": "fg3m",
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build XGBoost training CSV from historical box scores."
    )
    parser.add_argument(
        "--stat",
        default="pts",
        choices=sorted(VALID_STATS),
        help="Target stat column to predict (default: pts)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path (default: data/xgboost_training_data.csv)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit("[ERROR] DATABASE_URL not found in environment / .env file.")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=5,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

RAW_QUERY = text("""
    SELECT
        pbs.player_id,
        pbs.game_id,
        pbs.team_id                     AS player_team_id,
        pbs.minutes_played,
        pbs.days_rest,
        pbs.pts,
        pbs.reb,
        pbs.ast,
        pbs.stl,
        pbs.blk,
        pbs.fg3m,
        g.game_date,
        g.home_team_id,
        g.away_team_id,
        g.closing_spread,
        g.closing_total,
        g.season,
        p.full_name
    FROM player_box_scores_traditional pbs
    JOIN games   g ON g.id  = pbs.game_id
    JOIN players p ON p.id  = pbs.player_id
    WHERE pbs.dnp_status   = FALSE
      AND pbs.minutes_played >= 5
    ORDER BY pbs.player_id, g.game_date
""")


def load_raw(engine) -> pd.DataFrame:
    print("Querying database...")
    with engine.connect() as conn:
        df = pd.read_sql(RAW_QUERY, conn)
    print(f"  Raw rows loaded: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 5


def build_features(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    For each row compute rolling features using only games BEFORE that row.
    Pandas .shift(1) ensures no data leakage (current game excluded).
    """
    stat_col = STAT_SQL_COL[stat]

    # --- Derived contextual columns ---
    df["is_home"] = (df["player_team_id"] == df["home_team_id"]).astype(int)
    df["opponent_team_id"] = df.apply(
        lambda r: r["away_team_id"] if r["is_home"] else r["home_team_id"],
        axis=1,
    )
    df["is_potential_blowout"] = (pd.to_numeric(df["closing_spread"], errors="coerce").abs() > 10.0).astype(int)

    # --- Rolling features per player (sorted by game_date, already ordered) ---
    grp = df.groupby("player_id", sort=False)

    # Shift by 1 so the current game is excluded, then roll over the window.
    # min_periods=1 keeps rows even when a player has fewer than 5 prior games.
    df[f"last_{ROLLING_WINDOW}_avg_{stat}"] = (
        grp[stat_col]
        .apply(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df[f"last_{ROLLING_WINDOW}_avg_min"] = (
        grp["minutes_played"]
        .apply(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # Drop the very first appearance of each player (no prior game to roll over)
    df = df[df[f"last_{ROLLING_WINDOW}_avg_{stat}"].notna()].copy()

    return df


# ---------------------------------------------------------------------------
# Column selection & export
# ---------------------------------------------------------------------------

def select_columns(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    stat_col = STAT_SQL_COL[stat]
    feature_cols = [
        # Rolling / form features
        f"last_{ROLLING_WINDOW}_avg_{stat}",
        f"last_{ROLLING_WINDOW}_avg_min",
        # Contextual features
        "opponent_team_id",
        "is_home",
        "days_rest",
        "is_potential_blowout",
        "closing_spread",
        "closing_total",
        "historical_minute_overlap",
        "usage_to_sub_ratio",
        "coach_id",
        "bench_reliance_factor",
        # Identifiers (kept for debugging / joining, not fed to model directly)
        "player_id",
        "game_id",
        "player_team_id",
        "game_date",
        "season",
        "full_name",
        # Target variable
        stat_col,
    ]
    return df[feature_cols].rename(columns={stat_col: f"target_{stat}"})


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    stat = args.stat

    engine = get_engine()

    raw_df = load_raw(engine)

    print(f"Engineering features (target stat: {stat})...")
    featured_df = build_features(raw_df, stat)

    print("Adding rotation / coaching features (DB scan; may take a minute)...")
    featured_df = apply_rotation_coaching_features(
        featured_df, engine, rolling_window=ROLLING_WINDOW
    )

    final_df = select_columns(featured_df, stat)

    # Sanity-check: no NaNs in critical columns
    critical = [f"last_{ROLLING_WINDOW}_avg_{stat}", f"last_{ROLLING_WINDOW}_avg_min",
                "opponent_team_id", "is_home", f"target_{stat}"]
    null_counts = final_df[critical].isnull().sum()
    if null_counts.any():
        print("[WARN] Null values detected in feature columns:")
        print(null_counts[null_counts > 0].to_string())

    output_path = args.output
    if output_path == DEFAULT_OUTPUT and stat != "pts":
        output_path = os.path.join(str(_REPO_ROOT), "data", f"xgboost_training_data_{stat}.csv")
    save_csv(final_df, output_path)

    print(f"\n[OK] Training data saved -> {output_path}")
    print(f"     Total rows : {len(final_df):,}")
    print(f"     Columns    : {list(final_df.columns)}")
    print(f"     Seasons    : {sorted(final_df['season'].unique())}")
    print(f"     Players    : {final_df['player_id'].nunique():,}")


if __name__ == "__main__":
    main()
