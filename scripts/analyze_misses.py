"""
Post-mortem analyzer for yesterday's misses.

Outputs:
  1) Top-10 game misses by abs(actual_margin - predicted_spread)
     where predicted_spread uses games.closing_spread.
  2) Top-10 graded prop misses by abs(actual_stat - predicted_value)
     where predicted_value uses prediction_log.ml_mean fallback to heuristic_mean.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


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


def _yesterday_ny() -> str:
    ny_today = datetime.now(ZoneInfo("America/New_York")).date()
    return (ny_today - timedelta(days=1)).isoformat()


def _load_game_misses(engine, ymd: str) -> pd.DataFrame:
    q = text(
        """
        SELECT
            g.id AS game_id,
            g.game_date,
            ta.abbreviation || ' @ ' || th.abbreviation AS game_label,
            g.closing_spread AS predicted_spread,
            (g.home_score - g.away_score) AS actual_margin,
            ABS((g.home_score - g.away_score) - g.closing_spread) AS residual,
            'MARKET' AS line_source
        FROM games g
        JOIN teams th ON th.id = g.home_team_id
        JOIN teams ta ON ta.id = g.away_team_id
        WHERE g.game_date = CAST(:gd AS date)
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
          AND g.closing_spread IS NOT NULL
          AND TRIM(LOWER(COALESCE(g.status, ''))) IN ('final', 'complete')
        ORDER BY residual DESC NULLS LAST
        LIMIT 10
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params={"gd": ymd})


def _load_prop_misses(engine, ymd: str) -> pd.DataFrame:
    q = text(
        """
        SELECT
            pl.game_date,
            pl.player_name,
            pl.team_abbr,
            pl.opponent,
            pl.stat,
            COALESCE(pl.ml_mean, pl.heuristic_mean) AS predicted_value,
            pl.actual_value AS actual_value,
            ABS(pl.actual_value - COALESCE(pl.ml_mean, pl.heuristic_mean)) AS residual,
            pl.line_source
        FROM prediction_log pl
        WHERE TRIM(pl.game_date) = TRIM(:gd)
          AND pl.stat IN ('pts', 'reb', 'ast', 'stl', 'blk', 'fg3')
          AND pl.actual_value IS NOT NULL
          AND COALESCE(pl.ml_mean, pl.heuristic_mean) IS NOT NULL
        ORDER BY residual DESC NULLS LAST
        LIMIT 10
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn, params={"gd": ymd})


def main() -> None:
    ymd = _yesterday_ny()
    engine = _get_engine()

    game_df = _load_game_misses(engine, ymd)
    prop_df = _load_prop_misses(engine, ymd)

    print(f"Post-Mortem for {ymd}")
    print("=" * 88)

    print("\nTop 10 Biggest Game Misses (by residual)")
    if game_df.empty:
        print("No completed games with closing_spread for yesterday.")
    else:
        print(
            game_df.rename(
                columns={
                    "game_label": "Game",
                    "predicted_spread": "Predicted Value",
                    "actual_margin": "Actual Value",
                    "residual": "Residual",
                    "line_source": "line_source",
                }
            ).to_string(index=False)
        )

    print("\nTop 10 Biggest Player Prop Misses (by residual)")
    if prop_df.empty:
        print("No graded player props found for yesterday.")
    else:
        out = prop_df.copy()
        out["Game/Player"] = (
            out["player_name"].astype(str)
            + " ("
            + out["team_abbr"].astype(str)
            + " vs "
            + out["opponent"].astype(str)
            + " "
            + out["stat"].str.upper().astype(str)
            + ")"
        )
        cols = ["Game/Player", "predicted_value", "actual_value", "residual", "line_source"]
        out = out[cols].rename(
            columns={
                "predicted_value": "Predicted Value",
                "actual_value": "Actual Value",
                "residual": "Residual",
            }
        )
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()

