"""
Train an NBA game winner model from Supabase games data.

Features (exact):
  - home_ppg
  - away_ppg
  - rest_diff

Target:
  - home_win (1 if home_score > away_score else 0)

home_ppg / away_ppg are season-to-date pregame offensive PPG
(within the same season, using only games before that game).

Usage:
  python train_model.py
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sqlalchemy import create_engine, text


FEATURES = [
    "home_ppg",
    "away_ppg",
    "rest_diff",
    "rest_risk_flag",
    "late_season_leverage",
    "closing_spread",
    "closing_total",
    "is_potential_blowout",
]
TARGET = "home_win"
TEMPORAL_CUTOFF = date(2026, 3, 1)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
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


def _load_games_df(engine) -> pd.DataFrame:
    q = text(
        """
        SELECT
            g.id,
            g.season,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            g.home_score,
            g.away_score,
            g.home_days_rest,
            g.away_days_rest,
            g.closing_spread,
            g.closing_total,
            g.status
        FROM games g
        WHERE g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND g.game_date IS NOT NULL
          AND TRIM(LOWER(COALESCE(g.status, ''))) IN ('final', 'complete')
        ORDER BY g.game_date, g.id
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(q, conn)
    if df.empty:
        raise RuntimeError(
            "No rows found: need status 'Final' or 'Complete' with non-null home_score, away_score, and game_date."
        )
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _build_team_history(games: pd.DataFrame) -> pd.DataFrame:
    home_rows = pd.DataFrame(
        {
            "game_id": games["id"],
            "season": games["season"],
            "game_date": games["game_date"],
            "team_id": games["home_team_id"],
            "points_scored": games["home_score"],
            "points_allowed": games["away_score"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "game_id": games["id"],
            "season": games["season"],
            "game_date": games["game_date"],
            "team_id": games["away_team_id"],
            "points_scored": games["away_score"],
            "points_allowed": games["home_score"],
        }
    )
    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games.sort_values(["season", "team_id", "game_date", "game_id"]).reset_index(drop=True)
    team_games["won"] = (team_games["points_scored"] > team_games["points_allowed"]).astype(int)
    # Prior-to-game season-to-date PPG (same season only).
    team_games["team_ppg_pre"] = (
        team_games.groupby(["season", "team_id"])["points_scored"]
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )
    # Prior-to-game season win% (same season only).
    team_games["team_win_pct_pre"] = (
        team_games.groupby(["season", "team_id"])["won"]
        .transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    )
    return team_games


def _build_training_frame(games: pd.DataFrame) -> pd.DataFrame:
    team_games = _build_team_history(games)

    home_ppg = (
        team_games.rename(columns={"game_id": "id", "team_id": "home_team_id", "team_ppg_pre": "home_ppg"})[
            ["id", "season", "home_team_id", "home_ppg"]
        ]
    )
    away_ppg = (
        team_games.rename(columns={"game_id": "id", "team_id": "away_team_id", "team_ppg_pre": "away_ppg"})[
            ["id", "season", "away_team_id", "away_ppg"]
        ]
    )
    home_winpct = (
        team_games.rename(
            columns={"game_id": "id", "team_id": "home_team_id", "team_win_pct_pre": "home_win_pct_pre"}
        )[["id", "season", "home_team_id", "home_win_pct_pre"]]
    )
    away_winpct = (
        team_games.rename(
            columns={"game_id": "id", "team_id": "away_team_id", "team_win_pct_pre": "away_win_pct_pre"}
        )[["id", "season", "away_team_id", "away_win_pct_pre"]]
    )

    df = (
        games.merge(home_ppg, on=["id", "season", "home_team_id"], how="left")
        .merge(away_ppg, on=["id", "season", "away_team_id"], how="left")
        .merge(home_winpct, on=["id", "season", "home_team_id"], how="left")
        .merge(away_winpct, on=["id", "season", "away_team_id"], how="left")
    )

    home_rest = pd.to_numeric(df["home_days_rest"], errors="coerce")
    away_rest = pd.to_numeric(df["away_days_rest"], errors="coerce")
    df["rest_diff"] = home_rest.fillna(0) - away_rest.fillna(0)
    df["closing_spread"] = pd.to_numeric(df["closing_spread"], errors="coerce")
    df["closing_total"] = pd.to_numeric(df["closing_total"], errors="coerce")

    # League-average PPG for new teams / insufficient history (min_periods=3 in _build_team_history).
    ppg_stack = pd.concat([df["home_ppg"], df["away_ppg"]], ignore_index=True)
    ppg_observed = ppg_stack.dropna()
    league_avg_ppg = float(ppg_observed.mean()) if len(ppg_observed) > 0 else 110.0
    df["home_ppg"] = df["home_ppg"].fillna(league_avg_ppg)
    df["away_ppg"] = df["away_ppg"].fillna(league_avg_ppg)

    med_total = df["closing_total"].median()
    if pd.isna(med_total):
        med_total = 220.0
    df["closing_spread"] = df["closing_spread"].fillna(0.0)
    df["closing_total"] = df["closing_total"].fillna(float(med_total))
    df["home_win_pct_pre"] = pd.to_numeric(df["home_win_pct_pre"], errors="coerce").fillna(0.5)
    df["away_win_pct_pre"] = pd.to_numeric(df["away_win_pct_pre"], errors="coerce").fillna(0.5)
    df["rest_risk_flag"] = (
        (home_rest.fillna(1) <= 0).astype(int) | (away_rest.fillna(1) <= 0).astype(int)
    ).astype(int)
    is_april = pd.to_datetime(df["game_date"]).dt.month == 4
    low_motivation = (df["home_win_pct_pre"] < 0.40) | (df["away_win_pct_pre"] < 0.40)
    df["late_season_leverage"] = (is_april & low_motivation).astype(int)
    df["is_potential_blowout"] = (df["closing_spread"].abs() > 10.0).astype(int)
    df[TARGET] = (df["home_score"] > df["away_score"]).astype(int)

    # Keep exactly the requested feature set + target.
    keep = ["id", "game_date"] + FEATURES + [TARGET]
    out = df[keep].copy()

    before_n = len(out)
    subset_cols = FEATURES + [TARGET]
    nan_per_col = out[subset_cols].isna().sum()
    nan_per_col_nonzero = nan_per_col[nan_per_col > 0].sort_values(ascending=False)
    worst_col = nan_per_col_nonzero.index[0] if len(nan_per_col_nonzero) else None
    worst_n = int(nan_per_col_nonzero.iloc[0]) if len(nan_per_col_nonzero) else 0

    out = out.dropna(subset=subset_cols).sort_values(["game_date", "id"]).reset_index(drop=True)
    dropped = before_n - len(out)
    if worst_col is not None:
        print(
            f"[debug] dropna: dropped {dropped} row(s); {len(out)} remain. "
            f"Worst NaN column: {worst_col!r} ({worst_n} NaNs)",
            flush=True,
        )
    else:
        print(
            f"[debug] dropna: dropped {dropped} row(s); {len(out)} remain. "
            "No NaNs in feature/target columns pre-drop.",
            flush=True,
        )
    if len(nan_per_col_nonzero):
        print("[debug] NaN counts before drop (subset columns only):", flush=True)
        for col, cnt in nan_per_col_nonzero.items():
            print(f"  {col}: {int(cnt)}", flush=True)

    if len(out) < 100:
        raise RuntimeError(f"Not enough training rows after feature build: {len(out)}")
    return out


def main() -> None:
    print("Loading games from Supabase...")
    engine = _get_engine()
    games = _load_games_df(engine)
    train_df = _build_training_frame(games)

    ny_today = datetime.now(ZoneInfo("America/New_York")).date()
    train_mask = train_df["game_date"] < TEMPORAL_CUTOFF
    val_mask = (train_df["game_date"] >= TEMPORAL_CUTOFF) & (train_df["game_date"] <= ny_today)

    train_slice = train_df.loc[train_mask].copy()
    val_slice = train_df.loc[val_mask].copy()
    if train_slice.empty:
        raise RuntimeError(f"No training rows before cutoff {TEMPORAL_CUTOFF}.")
    if val_slice.empty:
        raise RuntimeError(
            f"No validation rows from cutoff {TEMPORAL_CUTOFF} through {ny_today}."
        )

    X_train, y_train = train_slice[FEATURES], train_slice[TARGET]
    X_test, y_test = val_slice[FEATURES], val_slice[TARGET]

    print(f"Rows total: {len(train_df):,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows : {len(X_test):,}")
    print(f"Cutoff    : {TEMPORAL_CUTOFF} (train < cutoff, val >= cutoff)")
    print(f"Val end   : {ny_today}")
    print(f"Features  : {FEATURES}")
    print("PPG type  : season-to-date pregame (same season)")

    model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=50,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = float("nan")

    print("\nModel metrics")
    print(f"Accuracy: {acc:.4f}")
    print(f"LogLoss : {ll:.4f}")
    print(f"ROC AUC : {auc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "features": FEATURES,
                "target": TARGET,
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_rows": int(len(train_df)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "temporal_split": {
                    "cutoff_date": TEMPORAL_CUTOFF.isoformat(),
                    "validation_end_date": ny_today.isoformat(),
                },
                "ppg_definition": "season_to_date_pregame_same_season",
                "metrics": {"accuracy": float(acc), "log_loss": float(ll), "roc_auc": float(auc)},
                "threshold": 0.5,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved meta : {META_PATH}")


if __name__ == "__main__":
    main()

