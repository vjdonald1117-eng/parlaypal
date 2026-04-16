"""
NBA Player Duplicate Cleanup
============================
Finds players with the same full_name but different IDs (created by mid-season
trades causing the scraper to insert a new player row instead of updating
team_id), consolidates all box score history to the canonical (most recent)
player record, then deletes the stale duplicate rows.

Usage
-----
  python fix_duplicates.py          # dry run - shows what would change
  python fix_duplicates.py --apply  # commits all changes to the database
"""

import os
import sys
import logging
from collections import defaultdict

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

DRY_RUN = "--apply" not in sys.argv

db_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"sslmode": "require", "options": "-c timezone=utc"},
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – find duplicates
# ---------------------------------------------------------------------------

def find_duplicate_players(session: Session) -> dict[str, list[dict]]:
    """Return {full_name: [player_dict, ...]} for names with >1 player record."""
    rows = session.execute(text("""
        SELECT id, team_id, full_name
        FROM players
        ORDER BY full_name, id
    """)).fetchall()

    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_name[row.full_name].append({
            "id":        row.id,
            "team_id":   row.team_id,
            "full_name": row.full_name,
        })

    return {name: players for name, players in by_name.items() if len(players) > 1}


# ---------------------------------------------------------------------------
# Step 2 – pick the canonical (current-team) player_id
# ---------------------------------------------------------------------------

def _most_recent_game_id(session: Session, player_id: int) -> int:
    """Largest game_id recorded for this player across both box score tables."""
    row = session.execute(text("""
        SELECT COALESCE(MAX(game_id), -1) FROM (
            SELECT game_id FROM player_box_scores_traditional WHERE player_id = :pid
            UNION ALL
            SELECT game_id FROM player_box_scores_advanced    WHERE player_id = :pid
        ) sub
    """), {"pid": player_id}).scalar()
    return row if row is not None else -1


def pick_canonical(session: Session, players: list[dict]) -> int:
    """
    The canonical player_id is the one with the highest most-recent game_id
    (i.e. the record that has activity on their new team).
    Ties are broken by the highest player_id.
    """
    return max(
        players,
        key=lambda p: (_most_recent_game_id(session, p["id"]), p["id"]),
    )["id"]


# ---------------------------------------------------------------------------
# Step 3 – merge and delete
# ---------------------------------------------------------------------------

def _repoint_table(session: Session, table: str, stale_id: int, canonical_id: int) -> int:
    """
    Move rows from stale_id to canonical_id, skipping any game_id that already
    has a row for canonical_id (protects the unique constraint).
    Returns number of rows re-pointed.
    """
    result = session.execute(text(f"""
        UPDATE {table}
        SET player_id = :canonical_id
        WHERE player_id = :stale_id
          AND game_id NOT IN (
              SELECT game_id FROM {table} WHERE player_id = :canonical_id
          )
    """), {"canonical_id": canonical_id, "stale_id": stale_id})
    moved = result.rowcount

    # Delete any remaining rows that couldn't be re-pointed (conflict with canonical)
    session.execute(text(f"""
        DELETE FROM {table} WHERE player_id = :stale_id
    """), {"stale_id": stale_id})

    return moved


def _repoint_injuries(session: Session, stale_id: int, canonical_id: int) -> int:
    """
    Re-point injury_reports rows, skipping any (player_id, game_id,
    report_sequence) that would violate the unique constraint.
    """
    result = session.execute(text("""
        UPDATE injury_reports
        SET player_id = :canonical_id
        WHERE player_id = :stale_id
          AND (game_id, report_sequence) NOT IN (
              SELECT game_id, report_sequence
              FROM injury_reports
              WHERE player_id = :canonical_id
          )
    """), {"canonical_id": canonical_id, "stale_id": stale_id})
    moved = result.rowcount

    session.execute(text("""
        DELETE FROM injury_reports WHERE player_id = :stale_id
    """), {"stale_id": stale_id})

    return moved


def fix_duplicates(session: Session, duplicates: dict[str, list[dict]]) -> None:
    total_box_rows_merged = 0
    total_players_deleted = 0

    for full_name, players in sorted(duplicates.items()):
        canonical_id = pick_canonical(session, players)
        stale_players = [p for p in players if p["id"] != canonical_id]
        canonical_team = next(p["team_id"] for p in players if p["id"] == canonical_id)

        log.info(
            "%-30s  canonical_id=%-10d  team_id=%-10s  stale=%s",
            full_name,
            canonical_id,
            str(canonical_team),
            [(p["id"], p["team_id"]) for p in stale_players],
        )

        for stale in stale_players:
            stale_id = stale["id"]

            # Count what we're about to move
            trad_total = session.execute(text(
                "SELECT COUNT(*) FROM player_box_scores_traditional WHERE player_id = :pid"
            ), {"pid": stale_id}).scalar()
            adv_total = session.execute(text(
                "SELECT COUNT(*) FROM player_box_scores_advanced WHERE player_id = :pid"
            ), {"pid": stale_id}).scalar()
            injury_total = session.execute(text(
                "SELECT COUNT(*) FROM injury_reports WHERE player_id = :pid"
            ), {"pid": stale_id}).scalar()

            log.info(
                "  stale_id=%-10d  trad=%d  adv=%d  injuries=%d",
                stale_id, trad_total, adv_total, injury_total,
            )

            if DRY_RUN:
                continue

            moved_trad    = _repoint_table(session, "player_box_scores_traditional", stale_id, canonical_id)
            moved_adv     = _repoint_table(session, "player_box_scores_advanced",    stale_id, canonical_id)
            moved_inj     = _repoint_injuries(session, stale_id, canonical_id)

            session.execute(text("DELETE FROM players WHERE id = :stale_id"), {"stale_id": stale_id})

            log.info(
                "    merged: trad=%d  adv=%d  injuries=%d  -> deleted player %d",
                moved_trad, moved_adv, moved_inj, stale_id,
            )

            total_box_rows_merged += moved_trad + moved_adv
            total_players_deleted += 1

    if DRY_RUN:
        log.info("")
        log.info("DRY RUN complete – no changes written.")
        log.info("Re-run with --apply to commit.")
    else:
        session.commit()
        log.info("")
        log.info(
            "Done. Merged %d box score rows. Deleted %d stale player records.",
            total_box_rows_merged, total_players_deleted,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    mode = "DRY RUN" if DRY_RUN else "APPLY"
    log.info("=" * 60)
    log.info("NBA Player Duplicate Cleanup  [%s]", mode)
    log.info("=" * 60)

    with Session(db_engine) as session:
        duplicates = find_duplicate_players(session)

        if not duplicates:
            log.info("No duplicate player names found. Database is clean.")
            return

        log.info("Found %d player name(s) with duplicate records:", len(duplicates))
        log.info("")
        fix_duplicates(session, duplicates)

    log.info("")
    log.info("=" * 60)
    log.info("Cleanup complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
