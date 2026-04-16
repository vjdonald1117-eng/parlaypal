"""
NBA Injury Report Scraper — ESPN Public API
============================================
Fetches the current NBA injury report from ESPN's public JSON endpoint,
matches players and teams to your Supabase database, and inserts new
records into the injury_reports table.

No credentials required. No Playwright. No brittle HTML scraping.

Source
------
  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

Season-Ending Detection
-----------------------
If the injury description contains any of the following keywords, the row is
flagged is_season_ending = True and status is forced to 'Out':
  'season', 'surgery', 'torn', 'ir', 'acl', 'achilles', 'indefinitely'

Carry-Forward Logic
-------------------
Before inserting today's ESPN entries, the scraper queries the database for
ANY player whose most recent injury record has is_season_ending = True on a
date BEFORE today.  If that player does NOT already appear in today's ESPN
pull, a new 'Out / is_season_ending=True' row is automatically inserted so
the projection model always sees those players as unavailable.

Usage
-----
  python scrape_injuries.py              # fetch + insert
  python scrape_injuries.py --dry-run    # preview without writing to DB
"""

import argparse
import logging
import os
import re
import sys
from datetime import date, datetime
from difflib import get_close_matches

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

load_dotenv()

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
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

db_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=5,
    connect_args={"sslmode": "require", "options": "-c timezone=utc"},
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
DATA_SOURCE     = "ESPN"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ESPN status string -> our canonical status
ESPN_STATUS_MAP = {
    "out":          "Out",
    "day-to-day":   "Day-To-Day",
    "questionable": "Questionable",
    "doubtful":     "Doubtful",
    "probable":     "Probable",
    "gtd":          "GTD",
    "active":       "Available",
    "available":    "Available",
    "suspension":   "Out",   # suspended players are unavailable; injury_type will note it
}

# Simulation policy: Questionable players are treated as unavailable until final report.
OUT_STATUSES = {"Out", "Doubtful", "Questionable"}

# Season-ending detection patterns.
#
# "surgery" and "indefinitely" are bare substring triggers — they are
# unambiguous in injury context.
#
# "season" alone is NOT a trigger because ESPN notes frequently use it in
# a stats context ("on the season he's averaging...") or a time context
# ("fifth game of the season").  Instead only specific season-ending phrases
# are matched:  "for the season", "rest of the season", "season-ending", etc.
#
# "acl" and "achilles" alone are NOT triggers because tendinitis /
# tendinopathy / soreness are day-to-day.  Only structural damage phrases
# (torn, rupture, tear) trigger.
#
# "IR" as a standalone word/acronym (word-boundary, avoids "bird"/"first").
SEASON_ENDING_SUBSTR: frozenset[str] = frozenset({
    "surgery",
    "indefinitely",
})

_SEASON_ENDING_PHRASES = re.compile(
    r"""
    # Season-ending language
    for\s+the\s+season        |   # "out for the season"
    rest\s+of\s+the\s+season  |   # "rest of the season"
    remainder\s+of.*season    |   # "remainder of the season"
    season[\s-]ending         |   # "season-ending" / "season ending"
    miss\s+the\s+season       |   # "will miss the season"

    # ACL structural damage (not tendinitis / soreness)
    torn\s+acl                |   # torn ACL
    acl\s+tear                |   # ACL tear
    acl\s+rupture             |   # ACL rupture

    # Achilles structural damage (not tendinitis / tendinopathy)
    torn\s+achilles           |   # torn Achilles
    achilles\s+rupture        |   # Achilles rupture
    achilles\s+tear               # Achilles tear
    """,
    re.IGNORECASE | re.VERBOSE,
)

_IR_RE = re.compile(r"\bir\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Season-ending detection
# ---------------------------------------------------------------------------

def detect_season_ending(text: str) -> bool:
    """
    Return True if *text* signals a season-ending / long-term IR injury.

    Triggers on:
      - Bare substrings: 'surgery', 'indefinitely'
      - Season-ending phrases: 'for the season', 'season-ending', 'rest of the season'
      - ACL structural damage: 'torn acl', 'acl tear', 'acl rupture'
      - Achilles structural damage: 'torn achilles', 'achilles rupture', 'achilles tear'
      - Standalone 'IR' (word boundary)

    Does NOT trigger on:
      - 'season' alone  ('on the season', 'game of the season')
      - 'achilles' alone  (tendinitis / tendinopathy / soreness)
      - 'acl' alone      (same reason)
    """
    lower = text.lower()
    if any(kw in lower for kw in SEASON_ENDING_SUBSTR):
        return True
    if _SEASON_ENDING_PHRASES.search(text):
        return True
    if _IR_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Fetch from ESPN API
# ---------------------------------------------------------------------------

def fetch_espn_injuries() -> list[dict]:
    """
    Call the ESPN public NBA injury API and return a flat list of individual
    injury entry dicts.

    The API nests injuries under team objects:
      { "injuries": [ { "displayName": "Atlanta Hawks", "injuries": [...] }, ... ] }

    Each returned dict has an extra "_team_displayName" key injected from its
    parent team object for downstream team resolution.
    """
    try:
        resp = requests.get(ESPN_INJURY_URL, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Failed to fetch ESPN injury data: %s", exc)
        sys.exit(1)

    data = resp.json()
    team_objects = data.get("injuries", [])

    flat: list[dict] = []
    for team_obj in team_objects:
        team_name = team_obj.get("displayName", "")
        for entry in team_obj.get("injuries", []):
            entry["_team_displayName"] = team_name
            flat.append(entry)

    log.info(
        "ESPN API: %d teams, %d individual injury entries.",
        len(team_objects), len(flat),
    )
    return flat


# ---------------------------------------------------------------------------
# Parse ESPN injury entry
# ---------------------------------------------------------------------------

def parse_entry(entry: dict) -> dict | None:
    """
    Map one ESPN injury entry to the fields needed for injury_reports.
    Returns None if mandatory fields are missing.
    Sets is_season_ending=True when the notes or injury description contain
    a season-ending keyword.
    """
    athlete = entry.get("athlete", {})
    details = entry.get("details") or {}

    player_name = athlete.get("displayName", "").strip()
    if not player_name:
        return None

    # Team full name injected by fetch_espn_injuries() from the parent team object.
    team_display_name = entry.get("_team_displayName", "").strip()

    # Status — normalise to our controlled vocabulary
    raw_status = entry.get("status", "").strip()
    status = ESPN_STATUS_MAP.get(raw_status.lower())
    if not status:
        log.debug("Unrecognised status %r for %s — skipping.", raw_status, player_name)
        return None

    # Injury details — suspensions have no details block
    is_suspension = raw_status.lower() == "suspension"
    body_part   = details.get("type")        or None   # e.g. "Knee", "Toe"
    detail_word = details.get("detail")      or None   # e.g. "Sprain", "Soreness"
    side_raw    = details.get("side")        or None   # "Left" | "Right" | None
    return_date = details.get("returnDate")  or None   # "YYYY-MM-DD" string

    # Build injury_type as "{Side} {BodyPart} {Detail}" where parts are available
    if is_suspension:
        injury_type = "Suspension"
    else:
        injury_parts = [p for p in [side_raw, body_part, detail_word] if p]
        injury_type  = " ".join(injury_parts).strip() or None

    # Normalise side
    side = side_raw.capitalize() if side_raw else None
    if side and side not in ("Left", "Right"):
        side = None

    # Parse ESPN's returnDate string -> Python date
    estimated_return = None
    if return_date:
        try:
            estimated_return = date.fromisoformat(return_date[:10])
        except ValueError:
            pass

    # Notes = ESPN's long comment (most informative field)
    notes_text = entry.get("longComment") or entry.get("shortComment") or ""
    notes_lower = notes_text.lower()

    # ESPN occasionally lags status fields while notes already indicate an upgrade.
    # Keep this narrow so only explicit probable/questionable language overrides.
    if status in {"Out", "Doubtful"}:
        if "upgraded to probable" in notes_lower or "likely going to be upgraded to probable" in notes_lower:
            status = "Probable"
        elif "listed as questionable" in notes_lower:
            status = "Questionable"

    # ESPN injury entry id for deduplication
    espn_id = str(entry.get("id", ""))

    # Parse report date from ESPN timestamp
    raw_date = entry.get("date", "")
    try:
        report_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        report_date = date.today()

    # -------------------------------------------------------------------
    # Season-ending detection
    # Scan all available text: notes, injury_type, body_part, detail_word
    # -------------------------------------------------------------------
    full_text = " ".join(filter(None, [
        notes_text,
        injury_type,
        body_part,
        detail_word,
    ]))
    is_season_ending = detect_season_ending(full_text)

    # Season-ending injuries are always Out, regardless of ESPN's status string
    if is_season_ending:
        status = "Out"

    return {
        "player_name":       player_name,
        "team_display_name": team_display_name,
        "status":            status,
        "is_out":            status in OUT_STATUSES,
        "is_season_ending":  is_season_ending,
        "body_part":         body_part,
        "injury_type":       injury_type,
        "side":              side,
        "estimated_return":  estimated_return,
        "report_date":       report_date,
        "espn_id":           espn_id,
        "notes":             notes_text[:2000],
    }


# ---------------------------------------------------------------------------
# Database lookups
# ---------------------------------------------------------------------------

def load_player_index(session: Session) -> dict[str, tuple[int, int]]:
    """
    Returns a dict: lowercase full_name -> (player_id, team_id).
    Also registers unambiguous last-name shortcuts.
    """
    rows = session.execute(
        text("SELECT id, team_id, full_name FROM players WHERE is_active = TRUE")
    ).fetchall()

    index: dict[str, tuple[int, int]] = {}
    last_name_count: dict[str, int]   = {}

    for r in rows:
        pid, tid, name = r.id, r.team_id, r.full_name
        if not name:
            continue
        index[name.strip().lower()] = (pid, tid)
        last = name.strip().split()[-1].lower()
        last_name_count[last] = last_name_count.get(last, 0) + 1

    # Unambiguous last-name shortcuts
    for r in rows:
        pid, tid, name = r.id, r.team_id, r.full_name
        if not name:
            continue
        last = name.strip().split()[-1].lower()
        if last_name_count.get(last, 0) == 1 and last not in index:
            index[last] = (pid, tid)

    return index


def load_team_index(session: Session) -> dict[str, int]:
    """
    Returns a dict with two lookup keys per team:
      - uppercase abbreviation  e.g. "ATL"
      - lowercase full name     e.g. "atlanta hawks"
    """
    rows = session.execute(text("SELECT id, abbreviation, full_name FROM teams")).fetchall()
    index: dict[str, int] = {}
    for r in rows:
        index[r.abbreviation.upper()] = r.id
        index[r.full_name.lower()]    = r.id
    return index


def match_player(
    name: str,
    player_index: dict[str, tuple[int, int]],
) -> tuple[int, int] | None:
    """Resolve a player display name to (player_id, team_id) via exact then fuzzy match."""
    key = name.strip().lower()

    if key in player_index:
        return player_index[key]

    # Fuzzy match against all known names
    candidates = get_close_matches(key, player_index.keys(), n=1, cutoff=0.82)
    if candidates:
        log.debug("Fuzzy matched %r -> %r", name, candidates[0])
        return player_index[candidates[0]]

    # Last name only
    last = key.split()[-1] if key.split() else ""
    if last and last in player_index:
        return player_index[last]

    return None


# ---------------------------------------------------------------------------
# Deduplication + sequence
# ---------------------------------------------------------------------------

def already_inserted(session: Session, player_id: int, espn_id: str) -> bool:
    """True if this ESPN injury entry was already inserted for this player."""
    row = session.execute(
        text(
            "SELECT 1 FROM injury_reports "
            "WHERE player_id = :pid "
            "  AND data_source = :src "
            "  AND notes LIKE :pat "
            "LIMIT 1"
        ),
        {"pid": player_id, "src": DATA_SOURCE, "pat": f"[espn:{espn_id}]%"},
    ).fetchone()
    return row is not None


def next_sequence(session: Session, player_id: int, report_date: date) -> int:
    """Next report_sequence for (player_id, game_id=NULL, report_date)."""
    row = session.execute(
        text(
            "SELECT COALESCE(MAX(report_sequence), 0) FROM injury_reports "
            "WHERE player_id = :pid AND game_id IS NULL AND report_date = :d"
        ),
        {"pid": player_id, "d": report_date},
    ).fetchone()
    return (row[0] if row else 0) + 1


# ---------------------------------------------------------------------------
# Carry-forward: season-ending injuries
# ---------------------------------------------------------------------------

def carry_forward_season_ending(
    session: Session,
    today: date,
    espn_player_ids: set[int],
    dry_run: bool = False,
) -> int:
    """
    Find every player whose most recent injury record has is_season_ending=True
    and whose report is from BEFORE today.  If that player does NOT appear in
    today's ESPN pull (espn_player_ids), insert a new Out record for today so
    the projection model never drops them by mistake.

    Returns the number of rows inserted (or that would be inserted in dry-run).
    """
    rows = session.execute(
        text("""
            SELECT DISTINCT ON (ir.player_id)
                ir.player_id,
                ir.team_id,
                ir.injury_type,
                ir.body_part,
                ir.side,
                ir.estimated_return,
                ir.notes
            FROM injury_reports ir
            WHERE ir.is_season_ending = TRUE
              AND ir.report_date      < :today
            ORDER BY ir.player_id, ir.report_date DESC
        """),
        {"today": today},
    ).fetchall()

    if not rows:
        log.info("Carry-forward: no season-ending players found in DB.")
        return 0

    log.info(
        "Carry-forward: %d season-ending player(s) found; "
        "%d already in today's ESPN pull.",
        len(rows),
        sum(1 for r in rows if r.player_id in espn_player_ids),
    )

    carried = 0
    for r in rows:
        # Skip if ESPN already reported this player today — their fresh record
        # will be inserted by the normal flow.
        if r.player_id in espn_player_ids:
            continue

        # Skip if a carry-forward (or any) record for today already exists.
        existing = session.execute(
            text(
                "SELECT 1 FROM injury_reports "
                "WHERE player_id = :pid AND report_date = :d LIMIT 1"
            ),
            {"pid": r.player_id, "d": today},
        ).fetchone()
        if existing:
            log.debug(
                "  Carry-forward skip (already has today record): player_id=%d",
                r.player_id,
            )
            continue

        seq   = next_sequence(session, r.player_id, today)
        notes = f"[carry-forward:season-ending] {r.notes or ''}".strip()[:2000]

        if dry_run:
            log.info(
                "  [DRY RUN CARRY]  player_id=%-8d  team_id=%-5d  %s",
                r.player_id, r.team_id, r.injury_type or "—",
            )
        else:
            session.execute(
                text("""
                    INSERT INTO injury_reports
                        (player_id, team_id, game_id, report_date, report_sequence,
                         status, injury_type, body_part, side,
                         is_out, is_season_ending,
                         estimated_return, notes, data_source)
                    VALUES
                        (:player_id, :team_id, NULL, :report_date, :report_sequence,
                         'Out', :injury_type, :body_part, :side,
                         TRUE, TRUE,
                         :estimated_return, :notes, 'ESPN-CARRYFORWARD')
                """),
                {
                    "player_id":       r.player_id,
                    "team_id":         r.team_id,
                    "report_date":     today,
                    "report_sequence": seq,
                    "injury_type":     r.injury_type,
                    "body_part":       r.body_part,
                    "side":            r.side,
                    "estimated_return": r.estimated_return,
                    "notes":           notes,
                },
            )
            log.info(
                "  Carried forward:  player_id=%-8d  %s",
                r.player_id, r.injury_type or "—",
            )

        carried += 1

    return carried


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------

def insert_reports(
    session: Session,
    parsed: list[dict],
    player_index: dict[str, tuple[int, int]],
    team_index: dict[str, int],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Resolve + insert injury report rows.
    Returns (inserted_count, skipped_count).
    """
    inserted = skipped = 0

    for rec in parsed:
        # --- Resolve player ---
        player_result = match_player(rec["player_name"], player_index)
        if player_result is None:
            log.warning("  No player match: %r — skipping.", rec["player_name"])
            skipped += 1
            continue
        player_id, player_team_id = player_result

        # --- Resolve team (ESPN gives full name; fall back to player's stored team) ---
        team_id = (
            team_index.get(rec["team_display_name"].lower())
            or player_team_id
        )
        if not team_id:
            log.warning(
                "  No team match for %r (player %r) — skipping.",
                rec["team_display_name"], rec["player_name"],
            )
            skipped += 1
            continue

        # --- Dedup ---
        if already_inserted(session, player_id, rec["espn_id"]):
            log.debug(
                "  Already in DB: %r (espn_id=%s) — skipping.",
                rec["player_name"], rec["espn_id"],
            )
            skipped += 1
            continue

        seq   = next_sequence(session, player_id, rec["report_date"])
        notes = f"[espn:{rec['espn_id']}] {rec['notes']}"

        se_flag = rec.get("is_season_ending", False)

        row = {
            "player_id":        player_id,
            "team_id":          team_id,
            "game_id":          None,
            "report_date":      rec["report_date"],
            "report_sequence":  seq,
            "status":           rec["status"],
            "injury_type":      rec["injury_type"],
            "body_part":        rec["body_part"],
            "side":             rec["side"],
            "is_out":           rec["is_out"],
            "is_season_ending": se_flag,
            "estimated_return": rec["estimated_return"],
            "notes":            notes[:2000],
            "data_source":      DATA_SOURCE,
        }

        if dry_run:
            se_tag = " [SEASON-ENDING]" if se_flag else ""
            log.info(
                "  [DRY RUN] %s (id=%d) | %-15s | %s%s",
                rec["player_name"], player_id,
                rec["status"], rec["injury_type"] or "—", se_tag,
            )
        else:
            session.execute(
                text("""
                    INSERT INTO injury_reports
                        (player_id, team_id, game_id, report_date, report_sequence,
                         status, injury_type, body_part, side, is_out, is_season_ending,
                         estimated_return, notes, data_source)
                    VALUES
                        (:player_id, :team_id, :game_id, :report_date, :report_sequence,
                         :status, :injury_type, :body_part, :side, :is_out, :is_season_ending,
                         :estimated_return, :notes, :data_source)
                """),
                row,
            )
            se_tag = " [SEASON-ENDING]" if se_flag else ""
            log.info(
                "  Inserted: %-22s | %-15s | %s%s",
                rec["player_name"], rec["status"], rec["injury_type"] or "—", se_tag,
            )

        inserted += 1

    return inserted, skipped


def apply_note_status_corrections(session: Session, report_day: date, dry_run: bool = False) -> int:
    """
    Correct stale status fields when ESPN notes explicitly indicate probable/questionable.
    This updates existing rows for the same report_date (helps when espn_id dedupe prevents reinsert).
    """
    q = text(
        """
        SELECT id, status, notes
        FROM injury_reports
        WHERE report_date = :d
          AND status IN ('Out', 'Doubtful')
          AND data_source IN ('ESPN', 'ESPN-CARRYFORWARD')
        """
    )
    rows = session.execute(q, {"d": report_day}).fetchall()
    n = 0
    for r in rows:
        notes = str(r.notes or "").lower()
        new_status = None
        if "upgraded to probable" in notes or "likely going to be upgraded to probable" in notes:
            new_status = "Probable"
        elif "listed as questionable" in notes:
            new_status = "Questionable"
        if not new_status:
            continue
        if dry_run:
            log.info("  [DRY RUN FIX] id=%s %s -> %s", r.id, r.status, new_status)
        else:
            session.execute(
                text(
                    """
                    UPDATE injury_reports
                    SET status = :st,
                        is_out = CASE WHEN :st IN ('Out', 'Doubtful') THEN TRUE ELSE FALSE END
                    WHERE id = :id
                    """
                ),
                {"st": new_status, "id": int(r.id)},
            )
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    today = date.today()

    log.info("=" * 60)
    log.info("NBA Injury Scraper — ESPN API")
    log.info("Mode : %s", "DRY RUN (no DB writes)" if dry_run else "LIVE")
    log.info("=" * 60)

    # 1. Fetch
    raw_entries = fetch_espn_injuries()

    # 2. Parse
    parsed: list[dict] = []
    for entry in raw_entries:
        result = parse_entry(entry)
        if result:
            parsed.append(result)

    se_count = sum(1 for p in parsed if p["is_season_ending"])
    log.info(
        "Parsed %d valid injury records (%d season-ending) from %d entries.",
        len(parsed), se_count, len(raw_entries),
    )

    if not parsed and not True:   # still run carry-forward even if ESPN is empty
        log.info("Nothing to insert. Done.")
        return

    with Session(db_engine) as session:
        player_index = load_player_index(session)
        team_index   = load_team_index(session)
        log.info(
            "DB index: %d active players, %d teams.",
            len(player_index), len(team_index),
        )

        # ------------------------------------------------------------------
        # 3. Carry-forward: season-ending players missing from today's ESPN pull
        # ------------------------------------------------------------------
        log.info("-" * 60)
        log.info("Step 3: Carry-forward season-ending injuries ...")

        # Build the set of player_ids ESPN has reported today (resolved to DB IDs)
        espn_player_ids: set[int] = set()
        for rec in parsed:
            result = match_player(rec["player_name"], player_index)
            if result:
                espn_player_ids.add(result[0])

        carried = carry_forward_season_ending(
            session, today, espn_player_ids, dry_run=dry_run
        )

        if not dry_run and carried:
            session.commit()
            log.info("Committed %d carry-forward row(s).", carried)

        # ------------------------------------------------------------------
        # 4. Insert today's ESPN records
        # ------------------------------------------------------------------
        log.info("-" * 60)
        log.info("Step 4: Inserting today's ESPN injury records ...")

        if parsed:
            inserted, skipped = insert_reports(
                session, parsed, player_index, team_index, dry_run=dry_run
            )
        else:
            inserted = skipped = 0
            log.info("  ESPN returned 0 parsed records — nothing to insert.")

        if not dry_run:
            session.commit()
            log.info("Committed.")

        log.info("-" * 60)
        log.info("Step 5: Applying note-based status corrections ...")
        corrected = apply_note_status_corrections(session, today, dry_run=dry_run)
        if not dry_run:
            session.commit()
        log.info("Corrected %d row(s) from notes.", corrected)

    log.info("")
    log.info("=" * 60)
    log.info(
        "Done.  Carried forward: %d  |  Inserted: %d  |  Skipped: %d",
        carried, inserted, skipped,
    )
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NBA injuries from ESPN API")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be inserted without writing to the database.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
