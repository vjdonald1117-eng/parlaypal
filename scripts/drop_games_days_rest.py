"""
Apply SQL cleanup to remove legacy games.days_rest.

Usage:
  python scripts/drop_games_days_rest.py
  python scripts/drop_games_days_rest.py --dry-run
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Only print whether column exists")
    args = ap.parse_args()

    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )

    with engine.begin() as conn:
        exists = conn.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'games'
                      AND column_name = 'days_rest'
                )
                """
            )
        ).scalar_one()

        print(f"games.days_rest exists: {bool(exists)}")
        if args.dry_run:
            print("[dry-run] no schema changes applied.")
            return

        conn.execute(text("ALTER TABLE public.games DROP COLUMN IF EXISTS days_rest"))
        print("Dropped legacy column: games.days_rest (if it existed).")


if __name__ == "__main__":
    main()

