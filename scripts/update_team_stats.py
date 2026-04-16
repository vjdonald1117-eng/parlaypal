"""
Compute season-to-date PPG (points per game) for every team from completed rows
in the games table and save a JSON file for predict_today.py / ML features.

Each completed game contributes one row per team (home points + away points).
PPG = SUM(team points) / COUNT(games played) for the chosen season.

Output (default):
  data/team_season_ppg.json

Usage:
  python scripts/update_team_stats.py
  python scripts/update_team_stats.py --season 2025-26
  python scripts/update_team_stats.py --output data/team_season_ppg.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _season_for_today() -> str:
    d = datetime.now(ZoneInfo("America/New_York")).date()
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    return f"{d.year - 1}-{str(d.year)[2:]}"


def _get_engine():
    load_dotenv(os.path.join(_repo_root(), ".env"))
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def compute_season_ppg(engine, season: str) -> dict[int, float]:
    """
    Average points scored per game (offense) for each team_id in the season.
    Only rows with both scores present are counted.
    """
    q = text(
        """
        WITH per_team AS (
            SELECT g.home_team_id AS team_id, g.home_score::double precision AS pts
            FROM games g
            WHERE g.season = :season
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.game_date IS NOT NULL
            UNION ALL
            SELECT g.away_team_id AS team_id, g.away_score::double precision AS pts
            FROM games g
            WHERE g.season = :season
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
              AND g.game_date IS NOT NULL
        )
        SELECT team_id, SUM(pts) / COUNT(*)::double precision AS ppg
        FROM per_team
        GROUP BY team_id
        ORDER BY team_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"season": season}).fetchall()
    return {int(r.team_id): round(float(r.ppg), 4) for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Write team season PPG JSON from games table.")
    ap.add_argument(
        "--season",
        default=None,
        help="NBA season string e.g. 2025-26 (default: inferred from today, NY time)",
    )
    ap.add_argument(
        "--output",
        default=os.path.join(_repo_root(), "data", "team_season_ppg.json"),
        help="Output JSON path",
    )
    args = ap.parse_args()

    season = args.season or _season_for_today()
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    engine = _get_engine()
    ppg_by_team = compute_season_ppg(engine, season)

    payload = {
        "season": season,
        "metric": "season_to_date_ppg_offense",
        "description": "Average points scored per game (home + away appearances) from games table.",
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_teams": len(ppg_by_team),
        "ppg_by_team_id": {str(k): v for k, v in sorted(ppg_by_team.items())},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(ppg_by_team)} teams  season={season}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
