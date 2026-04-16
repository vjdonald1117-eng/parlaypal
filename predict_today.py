"""
Predict home win probabilities for today's NBA games from Supabase.

Loads model from:
  - models/xgboost_game_winner_model.json
  - models/xgboost_game_winner_model.meta.json

Uses features:
  - home_ppg
  - away_ppg
  - rest_diff

Usage:
  python predict_today.py

For season-to-date PPG (recommended after syncing games), run first:
  python scripts/update_team_stats.py
That writes data/team_season_ppg.json, which this script prefers over the
rolling 10-game fallback. Re-train with train_model.py if you switch PPG defs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEAM_SEASON_PPG_PATH = os.path.join(DATA_DIR, "team_season_ppg.json")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_game_winner_model.json")
META_PATH = os.path.join(MODELS_DIR, "xgboost_game_winner_model.meta.json")


def _get_engine():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def _load_meta() -> dict:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_season_ppg_from_file() -> dict[int, float] | None:
    """
    Written by scripts/update_team_stats.py — season-to-date offensive PPG per team.
    """
    if not os.path.exists(TEAM_SEASON_PPG_PATH):
        return None
    try:
        with open(TEAM_SEASON_PPG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("ppg_by_team_id") or {}
        return {int(k): float(v) for k, v in raw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _league_avg_ppg_from_lookup(ppg_lookup: dict[int, float]) -> float:
    vals = [float(v) for v in ppg_lookup.values() if v is not None and pd.notna(v)]
    return float(sum(vals) / len(vals)) if vals else 110.0


def _build_team_ppg_lookup_rolling(engine) -> dict[int, float]:
    """Fallback: prior 10-game rolling PPG (completed games only, aligned with training)."""
    q = text(
        """
        SELECT
            g.id AS game_id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score
        FROM games g
        WHERE g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND g.game_date IS NOT NULL
          AND TRIM(LOWER(COALESCE(g.status, ''))) IN ('final', 'complete')
        ORDER BY g.game_date, g.id
        """
    )
    with engine.connect() as conn:
        games = pd.read_sql_query(q, conn)
    if games.empty:
        return {}

    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    home_rows = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": games["game_date"],
            "team_id": games["home_team_id"],
            "points_scored": games["home_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "game_date": games["game_date"],
            "team_id": games["away_team_id"],
            "points_scored": games["away_score"],
        }
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    team_games["team_ppg_pre"] = (
        team_games.groupby("team_id")["points_scored"]
        .transform(lambda s: s.shift(1).rolling(window=10, min_periods=3).mean())
    )

    latest = (
        team_games.dropna(subset=["team_ppg_pre"])
        .sort_values(["team_id", "game_date", "game_id"])
        .groupby("team_id", as_index=False)
        .tail(1)
    )
    return {int(r.team_id): float(r.team_ppg_pre) for r in latest.itertuples(index=False)}


def _build_team_ppg_lookup(engine) -> tuple[dict[int, float], str]:
    """
    Prefer season-to-date PPG from data/team_season_ppg.json if present
    (run: python scripts/update_team_stats.py). Otherwise rolling 10-game PPG.
    """
    from_file = _load_season_ppg_from_file()
    if from_file:
        return from_file, f"season_to_date ({TEAM_SEASON_PPG_PATH})"
    return _build_team_ppg_lookup_rolling(engine), "rolling_10_game (DB; matches train_model.py)"


def _count_games_for_date(engine, game_date_str: str) -> int:
    q = text(
        """
        SELECT COUNT(*) AS n
        FROM games g
        WHERE g.game_date = :game_date
        """
    )
    with engine.connect() as conn:
        row = conn.execute(q, {"game_date": game_date_str}).fetchone()
    return int(row.n or 0) if row else 0


def _load_games_not_final_for_date(engine, game_date_str: str) -> pd.DataFrame:
    q = text(
        """
        SELECT
            g.id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_days_rest,
            g.away_days_rest,
            g.closing_spread,
            g.closing_total,
            ht.abbreviation AS home_abbr,
            at.abbreviation AS away_abbr,
            ht.full_name AS home_team_name,
            at.full_name AS away_team_name
        FROM games g
        JOIN teams ht ON ht.id = g.home_team_id
        JOIN teams at ON at.id = g.away_team_id
        WHERE g.game_date = :game_date
          AND (g.status IS NULL OR g.status <> 'Final')
        ORDER BY g.id
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(q, conn, params={"game_date": game_date_str})
    return df


def _current_season_for_date(game_date_str: str) -> str:
    d = datetime.strptime(game_date_str, "%Y-%m-%d").date()
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    return f"{d.year - 1}-{str(d.year)[2:]}"


def _build_team_star_status(engine, season: str) -> dict[int, bool]:
    """
    Proxy for team 'Star' status:
    team has at least one active player averaging >= 25 PPG this season.
    """
    q = text(
        """
        SELECT p.team_id, MAX(avg_pts) AS max_avg_pts
        FROM (
            SELECT p.id, p.team_id, AVG(pbs.pts) AS avg_pts
            FROM players p
            JOIN player_box_scores_traditional pbs ON p.id = pbs.player_id
            JOIN games g ON g.id = pbs.game_id
            WHERE p.is_active = TRUE
              AND pbs.dnp_status = FALSE
              AND g.season = :season
            GROUP BY p.id, p.team_id
        ) p
        GROUP BY p.team_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"season": season}).fetchall()
    return {int(r.team_id): float(r.max_avg_pts or 0.0) >= 25.0 for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--game-date",
        default=None,
        help="Override date in YYYY-MM-DD format (default: today's NY date).",
    )
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_model.py first.")

    meta = _load_meta()
    features = meta.get("features", ["home_ppg", "away_ppg", "rest_diff"])
    threshold = float(meta.get("threshold", 0.5))

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    engine = _get_engine()
    ppg_lookup, ppg_source = _build_team_ppg_lookup(engine)
    print(f"PPG features: {ppg_source}")
    if "season_to_date" in ppg_source:
        print("(train_model.py now uses season-to-date pregame PPG, so feature type is aligned.)")

    ny_today = datetime.now(ZoneInfo("America/New_York")).date()
    target_date = args.game_date or ny_today.strftime("%Y-%m-%d")
    total_rows_today = _count_games_for_date(engine, target_date)
    print(f"Debug: total games rows for {target_date}: {total_rows_today}")
    today_games = _load_games_not_final_for_date(engine, target_date)

    if today_games.empty:
        print(f"No non-final games found for {target_date}.")
        return

    df = today_games.copy()
    league_fallback_ppg = _league_avg_ppg_from_lookup(ppg_lookup)
    df["home_ppg"] = pd.to_numeric(df["home_team_id"].map(ppg_lookup), errors="coerce").fillna(
        league_fallback_ppg
    )
    df["away_ppg"] = pd.to_numeric(df["away_team_id"].map(ppg_lookup), errors="coerce").fillna(
        league_fallback_ppg
    )
    df["rest_diff"] = pd.to_numeric(df["home_days_rest"], errors="coerce").fillna(0) - pd.to_numeric(
        df["away_days_rest"], errors="coerce"
    ).fillna(0)
    df["closing_spread"] = pd.to_numeric(df["closing_spread"], errors="coerce")
    df["closing_total"] = pd.to_numeric(df["closing_total"], errors="coerce")
    med_total = df["closing_total"].median()
    if pd.isna(med_total):
        med_total = 220.0
    df["closing_spread"] = df["closing_spread"].fillna(0.0)
    df["closing_total"] = df["closing_total"].fillna(float(med_total))
    df["is_potential_blowout"] = (df["closing_spread"].abs() > 10.0).astype(int)

    # Blowout penalty: if spread > 12 and favorite has star status, reduce team PPG
    # as a proxy for starter minutes risk in one-sided games.
    season = _current_season_for_date(target_date)
    star_status = _build_team_star_status(engine, season)
    blowout_mask = df["closing_spread"].abs() > 12.0
    # home favorite when spread is negative from home perspective.
    home_fav_mask = blowout_mask & (df["closing_spread"] < 0)
    away_fav_mask = blowout_mask & (df["closing_spread"] > 0)
    home_star = df["home_team_id"].map(star_status).fillna(False)
    away_star = df["away_team_id"].map(star_status).fillna(False)
    df.loc[home_fav_mask & home_star, "home_ppg"] = (
        df.loc[home_fav_mask & home_star, "home_ppg"] * 0.92
    )
    df.loc[away_fav_mask & away_star, "away_ppg"] = (
        df.loc[away_fav_mask & away_star, "away_ppg"] * 0.92
    )

    # Keep only rows with all required model features.
    missing_mask = df[features].isna().any(axis=1)
    missing_rows = df[missing_mask]
    df = df[~missing_mask].copy()

    if df.empty:
        print(f"Games found for {target_date}, but no rows had complete features: {features}")
        if not missing_rows.empty:
            print("Missing features for:")
            for r in missing_rows.itertuples(index=False):
                print(f"  - {r.away_abbr} @ {r.home_abbr}")
        return

    X = df[features]
    home_prob = model.predict_proba(X)[:, 1]
    df["home_win_prob"] = home_prob
    df["away_win_prob"] = 1.0 - df["home_win_prob"]
    df["pick"] = df["home_win_prob"].apply(lambda p: "HOME" if p >= threshold else "AWAY")

    print(f"\nNBA win probabilities for {target_date} (NY)")
    print("-" * 72)
    for r in df.itertuples(index=False):
        matchup = f"{r.away_abbr} @ {r.home_abbr}"
        print(
            f"{matchup:<18} "
            f"Home {r.home_win_prob * 100:6.2f}%  "
            f"Away {r.away_win_prob * 100:6.2f}%  "
            f"Pick={r.pick}"
        )

    if not missing_rows.empty:
        print("\nSkipped games (missing feature values):")
        for r in missing_rows.itertuples(index=False):
            print(f"  - {r.away_abbr} @ {r.home_abbr}")


if __name__ == "__main__":
    main()

