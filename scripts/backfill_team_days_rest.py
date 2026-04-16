"""
Backfill team days of rest.

Computes the number of days since each team's previous game and writes:
  - team_box_scores.days_rest
  - games.home_days_rest / games.away_days_rest

This uses Postgres window functions (fast, no client-side loops).

Usage
-----
  python scripts/backfill_team_days_rest.py --dry-run
  python scripts/backfill_team_days_rest.py
  python scripts/backfill_team_days_rest.py --season 2025-26
  python scripts/backfill_team_days_rest.py --since 2025-10-01
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _get_engine():
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (check .env).")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=5,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def _games_has_columns(conn, cols: list[str]) -> dict[str, bool]:
    q = text(
        """
        select column_name
        from information_schema.columns
        where table_schema = 'public' and table_name = 'games'
        """
    )
    existing = {r[0] for r in conn.execute(q).fetchall()}
    return {c: c in existing for c in cols}


def _build_where_filters(args) -> tuple[str, dict[str, Any]]:
    where = []
    params: dict[str, Any] = {}
    if args.season:
        where.append("g.season = :season")
        params["season"] = args.season
    if args.since:
        where.append("g.game_date >= :since")
        params["since"] = args.since
    if args.until:
        where.append("g.game_date <= :until")
        params["until"] = args.until
    if not where:
        return "", params
    return " AND " + " AND ".join(where), params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None, help="Season string like 2025-26")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--dry-run", action="store_true", help="Compute counts only; no DB writes")
    args = ap.parse_args()

    eng = _get_engine()
    with eng.begin() as conn:
        flags = _games_has_columns(conn, ["home_days_rest", "away_days_rest"])
        if not (flags["home_days_rest"] and flags["away_days_rest"]):
            raise RuntimeError(
                "games table is missing home_days_rest/away_days_rest columns. "
                "This repo expects per-team rest stored as two columns on games."
            )

        extra_where, params = _build_where_filters(args)

        # Window diff returns an integer number of days (NULL for each team's first game).
        team_games_cte = f"""
        with team_games as (
            select
                tbs.game_id,
                tbs.team_id,
                tbs.is_home,
                g.game_date,
                (g.game_date - lag(g.game_date) over (
                    partition by tbs.team_id
                    order by g.game_date, tbs.game_id
                ))::int as days_rest
            from team_box_scores tbs
            join games g on g.id = tbs.game_id
            where 1=1 {extra_where}
        )
        """

        # How many rows would change?
        count_q = text(
            team_games_cte
            + """
            select
                count(*) as n_rows,
                count(*) filter (where days_rest is null) as n_null,
                count(*) filter (where days_rest is not null) as n_non_null
            from team_games
            """
        )
        counts = conn.execute(count_q, params).mappings().one()
        print(
            f"[days_rest] team_games rows={counts['n_rows']} "
            f"(null={counts['n_null']}, non_null={counts['n_non_null']})"
        )

        if args.dry_run:
            print("[dry-run] no updates applied.")
            return

        # 1) Update team_box_scores.days_rest
        upd_team_box = text(
            team_games_cte
            + """
            update team_box_scores tbs
            set days_rest = tg.days_rest
            from team_games tg
            where tbs.game_id = tg.game_id
              and tbs.team_id = tg.team_id
            """
        )
        res1 = conn.execute(upd_team_box, params)
        print(f"[update] team_box_scores.days_rest updated rows={res1.rowcount}")

        # 2) Update games.home_days_rest / away_days_rest
        upd_games = text(
            team_games_cte
            + """
            , home as (
                select game_id, days_rest from team_games where is_home = true
            )
            , away as (
                select game_id, days_rest from team_games where is_home = false
            )
            update games g
            set
                home_days_rest = home.days_rest,
                away_days_rest = away.days_rest
            from home
            join away on away.game_id = home.game_id
            where g.id = home.game_id
            """
        )
        res2 = conn.execute(upd_games, params)
        print(f"[update] games.home_days_rest/away_days_rest updated rows={res2.rowcount}")


if __name__ == "__main__":
    main()

