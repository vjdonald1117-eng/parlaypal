"""
NBA Analytics Engine  - Historical Data Seeder
=============================================
Fetches game, team, and player box score data from nba_api and inserts it
into Supabase via SQLAlchemy.

Usage
-----
  python seed_historical_data.py          # TEST_MODE=True  ->  first 2 weeks of 2024-25
  TEST_MODE=False python seed_historical_data.py  # full 5-season backfill

Rate limiting
-------------
  Every API call is preceded by a 2-second sleep.
  Transient failures (timeout / 429) are retried up to 3 times with
  exponential back-off (10s, 20s, 30s).
"""

import os
import re
import time
import logging
from datetime import date
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
)
from nba_api.stats.static import teams as nba_teams_static

# ORM models live in migrate.py; importing them here reuses the same schema
# definitions so there is one source of truth.
from migrate import (
    Base, Team, Player, Game, TeamBoxScore,
    PlayerBoxScoreTraditional, PlayerBoxScoreAdvanced,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

db_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"sslmode": "require", "options": "-c timezone=utc"},
)

# -- Season config -------------------------------------------------------------
ALL_SEASONS = ["2025-26", "2024-25", "2023-24"]

# Set TEST_MODE = False (or via env var) to run the full backfill.
TEST_MODE      = os.getenv("TEST_MODE", "true").lower() != "false"
TEST_DATE_FROM = "2025-10-28"   # 2025-26 regular season opener
TEST_DATE_TO   = "2025-11-11"   # ~2 weeks later

# -- Static conference / division lookup (NBA alignment as of 2024-25) ---------
TEAM_CONF_DIV: dict[int, tuple[str, str]] = {
    # East - Atlantic
    1610612738: ("East", "Atlantic"),   # BOS  Boston Celtics
    1610612751: ("East", "Atlantic"),   # BKN  Brooklyn Nets
    1610612752: ("East", "Atlantic"),   # NYK  New York Knicks
    1610612755: ("East", "Atlantic"),   # PHI  Philadelphia 76ers
    1610612761: ("East", "Atlantic"),   # TOR  Toronto Raptors
    # East - Central
    1610612741: ("East", "Central"),    # CHI  Chicago Bulls
    1610612739: ("East", "Central"),    # CLE  Cleveland Cavaliers
    1610612765: ("East", "Central"),    # DET  Detroit Pistons
    1610612754: ("East", "Central"),    # IND  Indiana Pacers
    1610612749: ("East", "Central"),    # MIL  Milwaukee Bucks
    # East - Southeast
    1610612737: ("East", "Southeast"),  # ATL  Atlanta Hawks
    1610612766: ("East", "Southeast"),  # CHA  Charlotte Hornets
    1610612748: ("East", "Southeast"),  # MIA  Miami Heat
    1610612753: ("East", "Southeast"),  # ORL  Orlando Magic
    1610612764: ("East", "Southeast"),  # WAS  Washington Wizards
    # West - Northwest
    1610612743: ("West", "Northwest"),  # DEN  Denver Nuggets
    1610612750: ("West", "Northwest"),  # MIN  Minnesota Timberwolves
    1610612760: ("West", "Northwest"),  # OKC  Oklahoma City Thunder
    1610612757: ("West", "Northwest"),  # POR  Portland Trail Blazers
    1610612762: ("West", "Northwest"),  # UTA  Utah Jazz
    # West - Pacific
    1610612744: ("West", "Pacific"),    # GSW  Golden State Warriors
    1610612746: ("West", "Pacific"),    # LAC  LA Clippers
    1610612747: ("West", "Pacific"),    # LAL  Los Angeles Lakers
    1610612756: ("West", "Pacific"),    # PHX  Phoenix Suns
    1610612758: ("West", "Pacific"),    # SAC  Sacramento Kings
    # West - Southwest
    1610612742: ("West", "Southwest"),  # DAL  Dallas Mavericks
    1610612745: ("West", "Southwest"),  # HOU  Houston Rockets
    1610612763: ("West", "Southwest"),  # MEM  Memphis Grizzlies
    1610612740: ("West", "Southwest"),  # NOP  New Orleans Pelicans
    1610612759: ("West", "Southwest"),  # SAS  San Antonio Spurs
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------
_API_SLEEP_SECS   = 2    # mandatory pause between every request
_RETRY_WAITS      = [10, 20, 30]   # seconds to wait on attempt 1, 2, 3


def api_call(endpoint_cls, *args, **kwargs):
    """
    Instantiate an nba_api endpoint class with rate-limit sleep and
    exponential-back-off retry on any exception.
    """
    kwargs.setdefault("timeout", 60)
    last_exc = None
    for attempt, wait in enumerate([0] + _RETRY_WAITS, start=1):
        if wait:
            log.warning("      Retry %d/%d - waiting %ds after: %s",
                        attempt - 1, len(_RETRY_WAITS), wait, last_exc)
            time.sleep(wait)
        time.sleep(_API_SLEEP_SECS)
        try:
            return endpoint_cls(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt > len(_RETRY_WAITS):
                log.error("      All retries exhausted for %s: %s",
                          endpoint_cls.__name__, exc)
                raise
    raise last_exc  # unreachable, but satisfies type checkers


# -----------------------------------------------------------------------------
# Data-transformation helpers
# -----------------------------------------------------------------------------

def parse_minutes(raw) -> float | None:
    """
    Convert NBA API minute strings ('35:27', '00:00', None) to decimal minutes.
    Returns None for DNP / not-available entries so downstream code can treat
    None as "did not play".
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("None", "0:00", "00:00"):
        return None
    if ":" in s:
        try:
            mins, secs = s.split(":")
            return int(mins) + int(secs) / 60
        except (ValueError, AttributeError):
            return None
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None


def classify_dnp_reason(comment: str) -> str:
    """
    Map the raw box-score comment string to the controlled vocabulary used
    in the dnp_reason column:
      REST | ILLNESS | INJURY | SUSPENSION | PERSONAL | INACTIVE | COACH_DECISION
    """
    c = (comment or "").upper().strip()
    if not c:
        return "COACH_DECISION"
    if "REST" in c:
        return "REST"
    if any(k in c for k in ("ILL", "SICK", "COVID")):
        return "ILLNESS"
    if any(k in c for k in ("INJUR",)):
        return "INJURY"
    if "SUSPEND" in c:
        return "SUSPENSION"
    if "PERSONAL" in c:
        return "PERSONAL"
    if any(k in c for k in ("INACTIVE", "NWT", "NOT WITH", "G LEAGUE", "TWO-WAY")):
        return "INACTIVE"
    # Catches: DNP-CD, DND, COACH'S DECISION, blank comments with no minutes
    return "COACH_DECISION"


def days_between(earlier: date | None, later: date) -> int | None:
    """Return integer day delta, or None if no prior date exists."""
    return (later - earlier).days if earlier else None


def calc_fantasy_pts(pts, reb, ast, stl, blk, tov) -> float | None:
    """DraftKings-style fantasy points formula."""
    if any(v is None for v in (pts, reb, ast, stl, blk, tov)):
        return None
    return round(pts + 1.2 * reb + 1.5 * ast + 3 * stl + 3 * blk - tov, 2)


def _int(v) -> int | None:
    try:
        return int(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _flt(v) -> float | None:
    try:
        return float(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------------
# PostgreSQL upsert helper
# -----------------------------------------------------------------------------

def upsert_rows(session: Session, model, rows: list[dict],
                conflict_cols: list[str]) -> None:
    """
    Batch-upsert a list of dicts into *model*'s table.
    On conflict (identified by *conflict_cols*), all other columns are updated.
    created_at is excluded from updates to preserve original insert timestamp.
    """
    if not rows:
        return
    stmt = pg_insert(model.__table__).values(rows)
    non_conflict = [
        c for c in rows[0]
        if c not in conflict_cols and c != "created_at"
    ]
    stmt = stmt.on_conflict_do_update(
        index_elements=conflict_cols,
        set_={c: stmt.excluded[c] for c in non_conflict},
    )
    session.execute(stmt)


# -----------------------------------------------------------------------------
# Step 1 - Seed all 30 teams
# -----------------------------------------------------------------------------

def seed_teams(session: Session) -> None:
    log.info("--- Step 1: Seeding teams ---")
    rows = []
    for t in nba_teams_static.get_teams():
        conf, div = TEAM_CONF_DIV.get(t["id"], (None, None))
        rows.append({
            "id":           t["id"],
            "abbreviation": t["abbreviation"],
            "full_name":    t["full_name"],
            "short_name":   t["nickname"],
            "city":         t["city"],
            "conference":   conf,
            "division":     div,
        })
    upsert_rows(session, Team, rows, ["id"])
    session.commit()
    log.info("    Upserted %d teams.", len(rows))


# -----------------------------------------------------------------------------
# Step 2 - Fetch the team-game log for a season / date window
# -----------------------------------------------------------------------------

def fetch_game_log(season: str, date_from: str | None,
                   date_to: str | None) -> pd.DataFrame:
    """
    Returns one row per team per game within the requested window.
    Adds derived columns: GAME_ID_INT and IS_HOME.
    """
    log.info("  Fetching game log  season=%s  %s -> %s",
             season, date_from or "season start", date_to or "today")
    kwargs: dict = dict(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    if date_from:
        kwargs["date_from_nullable"] = date_from
    if date_to:
        kwargs["date_to_nullable"] = date_to

    resp    = api_call(leaguegamefinder.LeagueGameFinder, **kwargs)
    df      = resp.get_data_frames()[0]

    if df.empty:
        return df

    df["GAME_DATE"]    = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["GAME_ID_INT"]  = df["GAME_ID"].astype(int)
    # "BOS vs. NYK" / "BOS vs NYK" -> home team row;  "NYK @ BOS" -> away team row
    def _lgf_matchup_is_home_row(matchup: str) -> bool:
        m = str(matchup or "").strip()
        if "@" in m:
            return False
        return bool(re.search(r"(?i)\bvs\.?\b", m))

    df["IS_HOME"] = df["MATCHUP"].astype(str).map(_lgf_matchup_is_home_row)

    log.info("    %d team-game rows across %d unique games.",
             len(df), df["GAME_ID"].nunique())
    return df


# -----------------------------------------------------------------------------
# Step 3 - Insert games + team_box_scores from the game log
# -----------------------------------------------------------------------------

def insert_games_and_team_boxes(session: Session, df: pd.DataFrame,
                                season: str) -> list[int]:
    """
    Builds and upserts `games` and `team_box_scores` rows.
    Returns the list of unique game IDs processed.

    days_rest for each team is computed here by sorting that team's games
    chronologically and diffing consecutive dates  - so it is always accurate
    within the loaded window (no DB round-trip needed).
    """
    log.info("--- Step 3: Inserting games + team box scores ---")

    # -- Compute days_rest per team --------------------------------------------
    # Build {team_id: sorted list of unique game dates}
    team_dates: dict[int, list[date]] = defaultdict(list)
    for _, row in df.iterrows():
        team_dates[int(row["TEAM_ID"])].append(row["GAME_DATE"])
    team_sorted: dict[int, list[date]] = {
        tid: sorted(set(dates)) for tid, dates in team_dates.items()
    }
    # Flat map: (team_id, game_date) -> days_rest
    team_rest: dict[tuple, int | None] = {}
    for tid, dates in team_sorted.items():
        prev = None
        for d in dates:
            team_rest[(tid, d)] = days_between(prev, d)
            prev = d

    # -- Pair home + away rows per game ----------------------------------------
    game_rows:     list[dict] = []
    team_box_rows: list[dict] = []

    for game_id_int, gdf in df.groupby("GAME_ID_INT"):
        home_df = gdf[gdf["IS_HOME"]]
        away_df = gdf[~gdf["IS_HOME"]]

        if home_df.empty or away_df.empty:
            log.warning("  Game %d: cannot determine home/away  - skipping.", game_id_int)
            continue

        home  = home_df.iloc[0]
        away  = away_df.iloc[0]
        gdate: date = home["GAME_DATE"]

        home_tid = int(home["TEAM_ID"])
        away_tid = int(away["TEAM_ID"])
        h_rest   = team_rest.get((home_tid, gdate))
        a_rest   = team_rest.get((away_tid, gdate))

        game_rows.append({
            "id":             game_id_int,
            "season":         season,
            "season_type":    "Regular Season",
            "game_date":      gdate,
            "home_team_id":   home_tid,
            "away_team_id":   away_tid,
            "home_score":     _int(home["PTS"]),
            "away_score":     _int(away["PTS"]),
            "status":         "Final",
            "home_days_rest": h_rest,
            "away_days_rest": a_rest,
        })

        for side_row, is_home, rest in (
            (home, True,  h_rest),
            (away, False, a_rest),
        ):
            team_box_rows.append({
                "game_id":    game_id_int,
                "team_id":    int(side_row["TEAM_ID"]),
                "is_home":    is_home,
                "days_rest":  rest,
                "pts":        _int(side_row["PTS"]),
                "fgm":        _int(side_row["FGM"]),
                "fga":        _int(side_row["FGA"]),
                "fg_pct":     _flt(side_row["FG_PCT"]),
                "fg3m":       _int(side_row["FG3M"]),
                "fg3a":       _int(side_row["FG3A"]),
                "fg3_pct":    _flt(side_row["FG3_PCT"]),
                "ftm":        _int(side_row["FTM"]),
                "fta":        _int(side_row["FTA"]),
                "ft_pct":     _flt(side_row["FT_PCT"]),
                "oreb":       _int(side_row["OREB"]),
                "dreb":       _int(side_row["DREB"]),
                "reb":        _int(side_row["REB"]),
                "ast":        _int(side_row["AST"]),
                "stl":        _int(side_row["STL"]),
                "blk":        _int(side_row["BLK"]),
                "tov":        _int(side_row["TOV"]),
                "pf":         _int(side_row["PF"]),
                "plus_minus": _int(side_row["PLUS_MINUS"]),
            })

    upsert_rows(session, Game,         game_rows,     ["id"])
    upsert_rows(session, TeamBoxScore, team_box_rows, ["game_id", "team_id"])
    session.commit()

    log.info("    Upserted %d games  |  %d team box score rows.",
             len(game_rows), len(team_box_rows))

    return [r["id"] for r in game_rows]


# -----------------------------------------------------------------------------
# Step 4 - Process per-game player box scores
# -----------------------------------------------------------------------------

def process_player_box_scores(session: Session, game_ids: list[int]) -> None:
    """
    For each game (processed in chronological order), fetches the V3
    traditional and advanced box scores, seeds any new players encountered,
    then inserts both box score tables.

    days_rest per player is computed by tracking the last game date each
    player actually appeared in (i.e. was not DNP).  We process games in
    date order so the tracker accumulates correctly across the full window.
    """
    log.info("--- Step 4: Processing player box scores (%d games) ---",
             len(game_ids))

    # Load game dates from DB so we can sort correctly
    game_date_map: dict[int, date] = {}
    for r in session.execute(text("SELECT id, game_date FROM games ORDER BY game_date")):
        game_date_map[r.id] = r.game_date

    sorted_ids = sorted(game_ids, key=lambda gid: game_date_map.get(gid, date.min))

    # player_id -> date of last game they actually played (not DNP)
    player_last_played: dict[int, date] = {}

    total = len(sorted_ids)
    for idx, game_id in enumerate(sorted_ids, 1):
        gdate      = game_date_map.get(game_id)
        game_id_str = str(game_id).zfill(10)   # NBA API expects 10-char zero-padded string

        log.info("  [%d/%d] Game %-12s  date=%s", idx, total, game_id_str, gdate)

        # -- Traditional box score ---------------------------------------------
        try:
            trad_resp = api_call(
                boxscoretraditionalv3.BoxScoreTraditionalV3,
                game_id=game_id_str,
            )
        except Exception as exc:
            log.error("    SKIP trad box score  - %s", exc)
            continue

        trad_frames    = trad_resp.get_data_frames()
        player_trad_df = trad_frames[0]   # index 0 = player rows

        # -- Advanced box score ------------------------------------------------
        adv_player_df = pd.DataFrame()
        try:
            adv_resp      = api_call(
                boxscoreadvancedv3.BoxScoreAdvancedV3,
                game_id=game_id_str,
            )
            adv_player_df = adv_resp.get_data_frames()[0]
        except Exception as exc:
            log.warning("    Advanced box score unavailable  - %s", exc)

        # -- Seed players ------------------------------------------------------
        player_rows = _build_player_rows(player_trad_df)
        upsert_rows(session, Player, player_rows, ["id"])
        session.commit()
        log.info("    Players upserted : %d", len(player_rows))

        # -- Traditional player box scores -------------------------------------
        trad_rows, newly_played = _build_trad_rows(
            player_trad_df, game_id, gdate, player_last_played
        )
        # Update tracker with players who actually played this game
        player_last_played.update(newly_played)
        upsert_rows(session, PlayerBoxScoreTraditional, trad_rows,
                    ["game_id", "player_id"])
        session.commit()

        dnp_count = sum(1 for r in trad_rows if r["dnp_status"])
        log.info("    Trad rows upserted: %d  (played=%d  DNP=%d)",
                 len(trad_rows), len(trad_rows) - dnp_count, dnp_count)

        # -- Advanced player box scores -----------------------------------------
        if not adv_player_df.empty:
            adv_rows = _build_adv_rows(adv_player_df, game_id)
            upsert_rows(session, PlayerBoxScoreAdvanced, adv_rows,
                        ["game_id", "player_id"])
            session.commit()
            log.info("    Adv rows upserted : %d", len(adv_rows))


# -----------------------------------------------------------------------------
# Row-builder functions
# -----------------------------------------------------------------------------

def _build_player_rows(df: pd.DataFrame) -> list[dict]:
    """Minimal player seed from box score  - position + team captured here."""
    rows = []
    seen: set[int] = set()
    for _, r in df.iterrows():
        pid = int(r["personId"])
        if pid in seen:
            continue
        seen.add(pid)
        rows.append({
            "id":         pid,
            "team_id":    int(r["teamId"]),
            "first_name": str(r["firstName"]),
            "last_name":  str(r["familyName"]),
            "full_name":  f"{r['firstName']} {r['familyName']}".strip(),
            "position":   str(r["position"]) if r["position"] else None,
            "is_active":  True,
        })
    return rows


def _build_trad_rows(
    df: pd.DataFrame,
    game_id: int,
    game_date: date,
    player_last_played: dict[int, date],
) -> tuple[list[dict], dict[int, date]]:
    """
    Build player_box_scores_traditional rows and return a dict of
    {player_id: game_date} for players who actually played (for updating
    the days_rest tracker in the caller).
    """
    df = df.drop_duplicates(subset=["personId"], keep="last")
    rows:        list[dict]      = []
    newly_played: dict[int, date] = {}

    for _, r in df.iterrows():
        pid  = int(r["personId"])
        mins = parse_minutes(r.get("minutes"))

        # -- DNP detection -----------------------------------------------------
        is_dnp = mins is None
        dnp_reason = (
            classify_dnp_reason(str(r.get("comment", ""))) if is_dnp else None
        )

        # -- days_rest: days since this player last *played* -------------------
        # Only games where the player was active (not DNP) count.
        last = player_last_played.get(pid)
        days_rest = days_between(last, game_date)
        if not is_dnp:
            newly_played[pid] = game_date

        # -- Stat columns ------------------------------------------------------
        pts  = _int(r.get("points"))
        reb  = _int(r.get("reboundsTotal"))
        ast  = _int(r.get("assists"))
        stl  = _int(r.get("steals"))
        blk  = _int(r.get("blocks"))
        tov  = _int(r.get("turnovers"))

        rows.append({
            "game_id":        game_id,
            "player_id":      pid,
            "team_id":        int(r["teamId"]),
            # -- Core requirement fields ----------------------------------------
            "dnp_status":     is_dnp,
            "dnp_reason":     dnp_reason,
            "minutes_played": mins,
            "days_rest":      days_rest,
            # -- Shooting ------------------------------------------------------
            "fgm":            _int(r.get("fieldGoalsMade")),
            "fga":            _int(r.get("fieldGoalsAttempted")),
            "fg_pct":         _flt(r.get("fieldGoalsPercentage")),
            "fg3m":           _int(r.get("threePointersMade")),
            "fg3a":           _int(r.get("threePointersAttempted")),
            "fg3_pct":        _flt(r.get("threePointersPercentage")),
            "ftm":            _int(r.get("freeThrowsMade")),
            "fta":            _int(r.get("freeThrowsAttempted")),
            "ft_pct":         _flt(r.get("freeThrowsPercentage")),
            # -- Counting stats ------------------------------------------------
            "oreb":           _int(r.get("reboundsOffensive")),
            "dreb":           _int(r.get("reboundsDefensive")),
            "reb":            reb,
            "ast":            ast,
            "stl":            stl,
            "blk":            blk,
            "tov":            tov,
            "pf":             _int(r.get("foulsPersonal")),
            "pts":            pts,
            "plus_minus":     _int(r.get("plusMinusPoints")),
            # -- Calculated ----------------------------------------------------
            "fantasy_pts":    calc_fantasy_pts(pts, reb, ast, stl, blk, tov),
        })

    return rows, newly_played


def _build_adv_rows(df: pd.DataFrame, game_id: int) -> list[dict]:
    df = df.drop_duplicates(subset=["personId"], keep="last")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "game_id":        game_id,
            "player_id":      int(r["personId"]),
            "team_id":        int(r["teamId"]),
            "minutes_played": parse_minutes(r.get("minutes")),
            "off_rating":     _flt(r.get("offensiveRating")),
            "def_rating":     _flt(r.get("defensiveRating")),
            "net_rating":     _flt(r.get("netRating")),
            "ast_pct":        _flt(r.get("assistPercentage")),
            "ast_to_tov":     _flt(r.get("assistToTurnover")),
            "ast_ratio":      _flt(r.get("assistRatio")),
            "oreb_pct":       _flt(r.get("offensiveReboundPercentage")),
            "dreb_pct":       _flt(r.get("defensiveReboundPercentage")),
            "reb_pct":        _flt(r.get("reboundPercentage")),
            "efg_pct":        _flt(r.get("effectiveFieldGoalPercentage")),
            "ts_pct":         _flt(r.get("trueShootingPercentage")),
            "usg_pct":        _flt(r.get("usagePercentage")),
            "to_ratio":       _flt(r.get("turnoverRatio")),
            "pace":           _flt(r.get("pace")),
            "pie":            _flt(r.get("PIE")),
        })
    return rows


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("NBA Analytics Engine  - Historical Data Seeder")
    if TEST_MODE:
        log.info("MODE : TEST  (%s -> %s, season 2025-26 only)",
                 TEST_DATE_FROM, TEST_DATE_TO)
    else:
        log.info("MODE : FULL  (seasons: %s)", ", ".join(ALL_SEASONS))
    log.info("=" * 60)

    with Session(db_engine) as session:

        # -- Step 1: teams (static, always run) -------------------------------
        seed_teams(session)

        # -- Season loop -------------------------------------------------------
        seasons = ["2025-26"] if TEST_MODE else ALL_SEASONS

        for season in seasons:
            date_from = TEST_DATE_FROM if TEST_MODE else None
            date_to   = TEST_DATE_TO   if TEST_MODE else None

            log.info("")
            log.info("--- Season: %s ---", season)

            # -- Step 2: game log ----------------------------------------------
            game_log = fetch_game_log(season, date_from, date_to)
            if game_log.empty:
                log.warning("  No games found  - skipping season.")
                continue

            # -- Step 3: games + team box scores ------------------------------
            game_ids = insert_games_and_team_boxes(session, game_log, season)

            # -- Step 4: player box scores (one API call-pair per game) --------
            process_player_box_scores(session, game_ids)

    log.info("")
    log.info("=" * 60)
    log.info("Seeding complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
