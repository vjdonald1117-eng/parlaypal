"""
scripts/reset_rosters.py
========================
One-time utility to fix scrambled team_id values in the players table.

Uses nba_api's CommonTeamRoster endpoint to fetch the current roster for
every team in the DB.  Updates are keyed on the NBA player ID — no fragile
name-matching — so trades, multi-word names, and accented characters are all
handled correctly.

What it does
------------
  1. Queries the teams table for all 30 team IDs.
  2. For each team calls CommonTeamRoster (current season) via nba_api.
  3. Upserts every player on that roster:
       - If the player already exists in the DB → update team_id + is_active.
       - If the player is new → insert a minimal record so future box score
         processing can link to them.
  4. Marks any player currently flagged is_active = TRUE who did NOT appear
     on any roster as is_active = FALSE (waived / two-way / unsigned).
  5. Prints a summary of every team_id change so you can verify the moves.

Usage
-----
  python scripts/reset_rosters.py
  python scripts/reset_rosters.py --dry-run   # print changes without writing
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

from nba_api.stats.endpoints import commonteamroster  # noqa: E402

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(
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
# Season helper (mirrors update_recent_games.py)
# ---------------------------------------------------------------------------
from datetime import date as _date

def _current_season() -> str:
    today = _date.today()
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


# ---------------------------------------------------------------------------
# Fetch roster via nba_api with retry
# ---------------------------------------------------------------------------
_API_SLEEP = 1.5   # seconds between requests — stay well within rate limits
_RETRY_WAITS = [10, 20, 30]


def fetch_team_roster(team_id: int, season: str) -> list[dict]:
    """
    Return a list of player dicts for the given team and season.
    Each dict has: player_id, first_name, last_name, full_name, position.
    Returns [] on persistent failure.
    """
    last_exc = None
    for attempt, wait in enumerate([0] + _RETRY_WAITS, start=1):
        if wait:
            log.warning("  Retry %d/%d for team %d after %ds: %s",
                        attempt - 1, len(_RETRY_WAITS), team_id, wait, last_exc)
            time.sleep(wait)
        time.sleep(_API_SLEEP)
        try:
            resp = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                timeout=30,
            )
            df = resp.get_data_frames()[0]
            if df.empty:
                return []

            players = []
            for _, row in df.iterrows():
                pid = int(row["PLAYER_ID"])
                # nba_api returns the full name in "PLAYER"; split it for first/last
                full = str(row.get("PLAYER", "")).strip()
                parts = full.split(" ", 1)
                first = parts[0] if parts else ""
                last  = parts[1] if len(parts) > 1 else ""
                pos   = str(row.get("POSITION", "")).strip() or None
                players.append({
                    "player_id":  pid,
                    "full_name":  full,
                    "first_name": first,
                    "last_name":  last,
                    "position":   pos,
                })
            return players

        except Exception as exc:
            last_exc = exc
            if attempt > len(_RETRY_WAITS):
                log.error("  All retries exhausted for team %d: %s", team_id, exc)
                return []

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    season = _current_season()
    log.info("=" * 60)
    log.info("reset_rosters.py — fix player team_ids via nba_api rosters")
    log.info("Season  : %s", season)
    log.info("Dry run : %s", dry_run)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load all teams from DB
    # ------------------------------------------------------------------
    with Session(engine) as session:
        team_rows = session.execute(
            text("SELECT id, abbreviation, full_name FROM teams ORDER BY abbreviation")
        ).fetchall()

    if not team_rows:
        log.error("No teams found in DB. Run migrate.py first.")
        sys.exit(1)

    log.info("Teams in DB: %d", len(team_rows))
    team_ids = [r.id for r in team_rows]
    id_to_abbr = {r.id: r.abbreviation for r in team_rows}

    # ------------------------------------------------------------------
    # 2. Fetch current rosters from NBA API
    # ------------------------------------------------------------------
    # roster_map: {player_id: {player_id, full_name, first_name, last_name,
    #                          position, team_id}}
    roster_map: dict[int, dict] = {}
    failed_teams: list[int] = []

    for tid in team_ids:
        abbr = id_to_abbr[tid]
        players = fetch_team_roster(tid, season)
        if not players:
            log.warning("  %s (%d): no roster returned — skipping team.", abbr, tid)
            failed_teams.append(tid)
            continue
        log.info("  %s (%d): %d players", abbr, tid, len(players))
        for p in players:
            p["team_id"] = tid
            roster_map[p["player_id"]] = p

    if not roster_map:
        log.error("No roster data retrieved at all. Aborting.")
        sys.exit(1)

    log.info("Total players found across all rosters: %d", len(roster_map))
    if failed_teams:
        log.warning(
            "Could not fetch rosters for %d team(s): %s — "
            "those teams' players will not be updated.",
            len(failed_teams),
            [id_to_abbr[t] for t in failed_teams],
        )

    # ------------------------------------------------------------------
    # 3. Compare against DB and compute changes
    # ------------------------------------------------------------------
    with Session(engine) as session:
        db_players = session.execute(
            text("SELECT id, full_name, team_id, is_active FROM players")
        ).fetchall()

    db_by_id: dict[int, object] = {r.id: r for r in db_players}

    to_update:   list[dict] = []   # existing players whose team_id changes
    to_deactivate: list[int] = []  # active players not on any roster
    to_insert:   list[dict] = []   # players on a roster but not yet in DB

    roster_pids = set(roster_map.keys())

    for pid, p in roster_map.items():
        if pid in db_by_id:
            db_row = db_by_id[pid]
            if db_row.team_id != p["team_id"] or not db_row.is_active:
                to_update.append({
                    "player_id":    pid,
                    "new_team_id":  p["team_id"],
                    "old_team_id":  db_row.team_id,
                    "full_name":    p["full_name"],
                    "old_active":   db_row.is_active,
                })
        else:
            to_insert.append(p)

    for db_row in db_players:
        if db_row.is_active and db_row.id not in roster_pids:
            # Only deactivate if their team is not in the failed list
            # (avoid falsely deactivating players whose team call failed)
            if db_row.team_id not in failed_teams:
                to_deactivate.append(db_row.id)

    # ------------------------------------------------------------------
    # 4. Print what will change
    # ------------------------------------------------------------------
    log.info("")
    log.info("=" * 60)
    log.info("Planned changes")
    log.info("=" * 60)

    if to_update:
        log.info("team_id / is_active corrections (%d):", len(to_update))
        for u in sorted(to_update, key=lambda x: x["full_name"]):
            old_abbr = id_to_abbr.get(u["old_team_id"], f"tid={u['old_team_id']}")
            new_abbr = id_to_abbr.get(u["new_team_id"], f"tid={u['new_team_id']}")
            active_tag = "" if u["old_active"] else "  [was inactive]"
            print(f"  UPDATE  {u['full_name']:<30s}  {old_abbr} → {new_abbr}{active_tag}")
    else:
        log.info("No team_id corrections needed.")

    if to_insert:
        log.info("New players to insert (%d):", len(to_insert))
        for p in sorted(to_insert, key=lambda x: x["full_name"]):
            abbr = id_to_abbr.get(p["team_id"], f"tid={p['team_id']}")
            print(f"  INSERT  {p['full_name']:<30s}  {abbr}")

    if to_deactivate:
        log.info("Players to mark inactive (%d) — not on any current roster:", len(to_deactivate))
        for pid in to_deactivate:
            db_row = db_by_id[pid]
            old_abbr = id_to_abbr.get(db_row.team_id, f"tid={db_row.team_id}")
            print(f"  DEACT   {db_row.full_name:<30s}  {old_abbr}")

    if dry_run:
        log.info("")
        log.info("Dry run — no DB writes performed.")
        return

    # ------------------------------------------------------------------
    # 5. Apply changes
    # ------------------------------------------------------------------
    with Session(engine) as session:
        # Corrections to existing players
        for u in to_update:
            session.execute(
                text("""
                    UPDATE players
                    SET team_id   = :team_id,
                        is_active = TRUE
                    WHERE id = :pid
                """),
                {"team_id": u["new_team_id"], "pid": u["player_id"]},
            )

        # Insert players who are on a roster but not yet in the DB
        # (e.g., rookies signed after historical data was seeded)
        for p in to_insert:
            session.execute(
                text("""
                    INSERT INTO players
                        (id, team_id, first_name, last_name, full_name,
                         position, is_active)
                    VALUES
                        (:id, :team_id, :first_name, :last_name, :full_name,
                         :position, TRUE)
                    ON CONFLICT (id) DO UPDATE
                        SET team_id   = EXCLUDED.team_id,
                            is_active = TRUE
                """),
                {
                    "id":         p["player_id"],
                    "team_id":    p["team_id"],
                    "first_name": p["first_name"],
                    "last_name":  p["last_name"],
                    "full_name":  p["full_name"],
                    "position":   p["position"],
                },
            )

        # Deactivate players no longer on any roster
        if to_deactivate:
            session.execute(
                text("UPDATE players SET is_active = FALSE WHERE id = ANY(:ids)"),
                {"ids": to_deactivate},
            )

        session.commit()

    log.info("")
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Updated   : %d", len(to_update))
    log.info("  Inserted  : %d", len(to_insert))
    log.info("  Deactivated: %d", len(to_deactivate))
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix scrambled player team_ids using current NBA rosters."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned changes without writing to the database.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
