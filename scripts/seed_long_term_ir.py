"""
scripts/seed_long_term_ir.py
==============================
One-time historical backfill for long-term IR / season-ending injuries.

Background
----------
ESPN's injury API is current-only — the ?dates= parameter has no effect and
always returns today's snapshot.  This means we cannot pull a true day-by-day
historical injury feed.

Instead, this script takes a two-phase approach:

  Phase 1 — Retroactive fix on existing records
    Re-runs detect_season_ending() against the stored notes and injury_type
    text of every row already in injury_reports.  Any row that the updated
    detection logic would flag gets is_season_ending = TRUE and status/is_out
    forced to 'Out' / TRUE.  This corrects every record the daily scraper has
    ever written — today's and all previous dates.

  Phase 2 — Full-season date-range backfill
    Fetches the current ESPN injury snapshot once.
    For each season-ending player found (using the same detect_season_ending
    logic), loops every calendar date from SEASON_START (Oct 21 2025) through
    yesterday.  For any date where that player has NO injury record at all, one
    'Out / is_season_ending=True' row is inserted so the daily scraper's
    carry_forward logic has an unbroken chain to work from.

Result
------
After this script runs once:
  • All existing injury records are correctly flagged.
  • Every season-ending player has at least one is_season_ending=True record
    per calendar date in the season, so carry_forward in scrape_injuries.py
    will keep them marked as Out forever — even if they fall off ESPN's feed.

Usage
-----
  python scripts/seed_long_term_ir.py             # live
  python scripts/seed_long_term_ir.py --dry-run   # preview
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from scrape_injuries import (      # noqa: E402
    detect_season_ending,
    fetch_espn_injuries,
    parse_entry,
    load_player_index,
    load_team_index,
    match_player,
)
from models.player_projections import engine  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASON_START  = date(2025, 10, 21)   # first game of 2025-26 regular season
DATA_SOURCE_BACKFILL    = "ESPN-BACKFILL"
DATA_SOURCE_RETROACTIVE = "ESPN-RETROACTIVE"


# ---------------------------------------------------------------------------
# Phase 1: Retroactively fix existing records
# ---------------------------------------------------------------------------

def phase1_fix_existing_records(session: Session, dry_run: bool) -> int:
    """
    Re-run detect_season_ending on every stored notes+injury_type value.
    Update rows that should be flagged but aren't.
    Returns number of rows updated.
    """
    log.info("Phase 1: scanning %d existing injury_reports records ...",
             session.execute(text("SELECT COUNT(*) FROM injury_reports")).scalar())

    rows = session.execute(
        text("""
            SELECT id, player_id, notes, injury_type, body_part,
                   is_season_ending, status
            FROM injury_reports
            WHERE notes IS NOT NULL OR injury_type IS NOT NULL
        """)
    ).fetchall()

    updated = 0
    for r in rows:
        full_text = " ".join(filter(None, [r.notes or "", r.injury_type or "", r.body_part or ""]))
        should_flag = detect_season_ending(full_text)

        if should_flag and not r.is_season_ending:
            if dry_run:
                log.info(
                    "  [DRY P1] id=%-6d  player_id=%-8d  injury=%s",
                    r.id, r.player_id, r.injury_type or "—",
                )
            else:
                session.execute(
                    text("""
                        UPDATE injury_reports
                        SET is_season_ending = TRUE,
                            status           = 'Out',
                            is_out           = TRUE
                        WHERE id = :id
                    """),
                    {"id": r.id},
                )
            updated += 1

    if not dry_run and updated:
        session.commit()

    log.info("Phase 1 complete: %d record(s) %s.",
             updated, "would be updated" if dry_run else "updated")
    return updated


# ---------------------------------------------------------------------------
# Phase 2: Date-range backfill from current ESPN snapshot
# ---------------------------------------------------------------------------

def phase2_backfill_date_range(
    session:     Session,
    player_index: dict,
    team_index:   dict,
    dry_run:     bool,
) -> tuple[int, int]:
    """
    Fetch current ESPN injuries once, identify season-ending players, then
    loop every date from SEASON_START through yesterday.  For each
    (player, date) pair where no injury record exists, insert one.

    Returns (players_processed, records_inserted).
    """
    yesterday = date.today() - timedelta(days=1)

    log.info(
        "Phase 2: fetching current ESPN snapshot and backfilling "
        "%s -> %s ...",
        SEASON_START, yesterday,
    )

    # --- Fetch + parse current ESPN injuries ---
    raw_entries = fetch_espn_injuries()
    season_ending_players: list[dict] = []

    for entry in raw_entries:
        rec = parse_entry(entry)
        if rec and rec.get("is_season_ending"):
            season_ending_players.append(rec)

    log.info("ESPN snapshot: %d season-ending player(s) identified.",
             len(season_ending_players))

    if not season_ending_players:
        log.info("Phase 2: nothing to backfill.")
        return 0, 0

    # --- Resolve each player to a DB id ---
    resolved: list[dict] = []
    for rec in season_ending_players:
        result = match_player(rec["player_name"], player_index)
        if result is None:
            log.warning("  Phase 2: no DB match for %r — skipping.", rec["player_name"])
            continue
        player_id, player_team_id = result

        team_id = (
            team_index.get(rec["team_display_name"].lower())
            or player_team_id
        )
        if not team_id:
            log.warning("  Phase 2: no team for %r — skipping.", rec["player_name"])
            continue

        resolved.append({
            "player_id":    player_id,
            "team_id":      team_id,
            "player_name":  rec["player_name"],
            "injury_type":  rec["injury_type"],
            "body_part":    rec["body_part"],
            "side":         rec["side"],
            "estimated_return": rec["estimated_return"],
            "notes":        rec["notes"],
        })

    log.info("Phase 2: %d player(s) resolved to DB ids.", len(resolved))

    # --- Build per-player set of existing record dates ---
    # Load all existing injury_report dates for these players in one query.
    player_ids = list({r["player_id"] for r in resolved})

    existing_dates: dict[int, set[date]] = {pid: set() for pid in player_ids}
    if player_ids:
        rows = session.execute(
            text("""
                SELECT player_id, report_date
                FROM injury_reports
                WHERE player_id = ANY(:pids)
            """),
            {"pids": player_ids},
        ).fetchall()
        for r in rows:
            existing_dates[r.player_id].add(r.report_date)

    # --- Date range ---
    all_dates: list[date] = []
    d = SEASON_START
    while d <= yesterday:
        all_dates.append(d)
        d += timedelta(days=1)

    log.info("Phase 2: date range %s -> %s (%d dates).",
             SEASON_START, yesterday, len(all_dates))

    # --- Insert missing records ---
    total_inserted = 0

    for player_rec in resolved:
        pid      = player_rec["player_id"]
        tid      = player_rec["team_id"]
        name     = player_rec["player_name"]
        covered  = existing_dates.get(pid, set())
        missing  = [d for d in all_dates if d not in covered]

        if not missing:
            log.info("  %-28s  already has full coverage — skipping.", name)
            continue

        log.info(
            "  %-28s  inserting %d record(s) for %d missing date(s) "
            "(out of %d total dates).",
            name, len(missing), len(missing), len(all_dates),
        )

        # sequence is always 1 for backfill rows: we verified `missing` dates
        # have zero existing records, so no round-trip to compute next_sequence.
        notes_text = (
            f"[backfill:season-ending] {player_rec['notes'] or ''}".strip()[:2000]
        )

        if dry_run:
            # Just count — no per-row logging to keep dry-run fast
            total_inserted += len(missing)
        else:
            # Batch insert all missing dates for this player in one call
            batch = [
                {
                    "player_id":        pid,
                    "team_id":          tid,
                    "report_date":      d,
                    "report_sequence":  1,
                    "injury_type":      player_rec["injury_type"],
                    "body_part":        player_rec["body_part"],
                    "side":             player_rec["side"],
                    "estimated_return": player_rec["estimated_return"],
                    "notes":            notes_text,
                    "data_source":      DATA_SOURCE_BACKFILL,
                }
                for d in missing
            ]
            session.execute(
                text("""
                    INSERT INTO injury_reports
                        (player_id, team_id, game_id, report_date,
                         report_sequence, status, injury_type, body_part,
                         side, is_out, is_season_ending,
                         estimated_return, notes, data_source)
                    VALUES
                        (:player_id, :team_id, NULL, :report_date,
                         :report_sequence, 'Out', :injury_type, :body_part,
                         :side, TRUE, TRUE,
                         :estimated_return, :notes, :data_source)
                """),
                batch,
            )
            session.commit()
            total_inserted += len(missing)

    return len(resolved), total_inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    yesterday = date.today() - timedelta(days=1)

    log.info("=" * 62)
    log.info("Long-Term IR Backfill  —  seed_long_term_ir.py")
    log.info("Mode         : %s", "DRY RUN" if dry_run else "LIVE")
    log.info("Season start : %s", SEASON_START)
    log.info("Through      : %s", yesterday)
    log.info("=" * 62)
    log.info("")
    log.info("NOTE: ESPN injury API is current-only (date param has no effect).")
    log.info("Phase 1 fixes existing DB records using stored notes text.")
    log.info("Phase 2 uses today's ESPN snapshot to backfill missing dates.")
    log.info("")

    with Session(engine) as session:
        # ---- Phase 1 --------------------------------------------------------
        log.info("-" * 62)
        p1_updated = phase1_fix_existing_records(session, dry_run)

        # ---- Phase 2 --------------------------------------------------------
        log.info("")
        log.info("-" * 62)
        player_index = load_player_index(session)
        team_index   = load_team_index(session)
        log.info("DB index: %d active players, %d teams.",
                 len(player_index), len(team_index))
        log.info("")

        p2_players, p2_inserted = phase2_backfill_date_range(
            session, player_index, team_index, dry_run
        )

    # ---- Summary ------------------------------------------------------------
    log.info("")
    log.info("=" * 62)
    log.info("Backfill complete.")
    log.info("  Phase 1 — existing records updated : %d", p1_updated)
    log.info("  Phase 2 — players processed        : %d", p2_players)
    log.info("  Phase 2 — new records inserted     : %d", p2_inserted)
    if dry_run:
        log.info("  (DRY RUN — no changes written to DB)")
    log.info("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill long-term IR injury records for the full season"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be inserted without writing to the database.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
