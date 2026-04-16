"""
NBA Analytics Engine - Recent Games Updater
===========================================
Fetches games from the last 48 hours, skips any already in the database,
and inserts new games with full team and player box scores.

Usage
-----
  python update_recent_games.py              # last 48 hours
  LOOKBACK_HOURS=72 python update_recent_games.py  # custom lookback
"""

import os
import re
import time
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests_cache
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv3,
    boxscoreadvancedv3,
)
from nba_api.stats.library.http import NBAStatsHTTP

from migrate import (
    Base, Team, Player, Game, TeamBoxScore,
    PlayerBoxScoreTraditional, PlayerBoxScoreAdvanced,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "48"))

db_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"sslmode": "require", "options": "-c timezone=utc"},
)

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
# HTTP cache (stats.nba.com / nba_api)
# ---------------------------------------------------------------------------
NBA_HTTP_CACHE_TTL_SECONDS = 15 * 60
NBA_HTTP_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
NBA_HTTP_CACHE_NAME = os.path.join(NBA_HTTP_CACHE_DIR, "nba_http_cache")


def _build_cached_http_session() -> requests_cache.CachedSession:
    os.makedirs(NBA_HTTP_CACHE_DIR, exist_ok=True)
    return requests_cache.CachedSession(
        cache_name=NBA_HTTP_CACHE_NAME,
        backend="sqlite",
        expire_after=NBA_HTTP_CACHE_TTL_SECONDS,
        allowable_methods=("GET",),
    )


HTTP_SESSION = _build_cached_http_session()
# Ensure nba_api endpoints (LeagueGameFinder, BoxScoreTraditionalV3, etc.) share this cache globally.
NBAStatsHTTP._session = HTTP_SESSION
_cached_request = HTTP_SESSION.request


def _logged_cached_request(method, url, *args, **kwargs):
    response = _cached_request(method, url, *args, **kwargs)
    cache_status = "HIT" if getattr(response, "from_cache", False) else "MISS"
    log.info("[nba_http_cache] %s %s %s", cache_status, str(method).upper(), response.url)
    return response


HTTP_SESSION.request = _logged_cached_request

# ---------------------------------------------------------------------------
# Rate limiting / retry (mirrors seed_historical_data.py)
# ---------------------------------------------------------------------------
_API_SLEEP_SECS = 2
_RETRY_WAITS    = [10, 20, 30]


def api_call(endpoint_cls, *args, **kwargs):
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
    raise last_exc


# ---------------------------------------------------------------------------
# Helpers (mirrors seed_historical_data.py)
# ---------------------------------------------------------------------------

_ISO_DUR_RE = re.compile(r"^PT(\d+)M([\d.]+)S$")


def parse_minutes(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in ("None", "0:00", "00:00"):
        return None
    # NBA Stats API v3 returns ISO 8601 duration: "PT28M30.00S"
    m = _ISO_DUR_RE.match(s)
    if m:
        val = int(m.group(1)) + float(m.group(2)) / 60
        return val if val > 0 else None
    # Legacy "MM:SS" format (v2 endpoints / seed scripts)
    if ":" in s:
        try:
            mins, secs = s.split(":")
            val = int(mins) + int(secs) / 60
            return val if val > 0 else None
        except (ValueError, AttributeError):
            return None
    try:
        val = float(s)
        return val if val > 0 else None
    except ValueError:
        return None


def classify_dnp_reason(comment: str) -> str:
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
    return "COACH_DECISION"


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


def calc_fantasy_pts(pts, reb, ast, stl, blk, tov) -> float | None:
    if any(v is None for v in (pts, reb, ast, stl, blk, tov)):
        return None
    return round(pts + 1.2 * reb + 1.5 * ast + 3 * stl + 3 * blk - tov, 2)


def upsert_rows(session: Session, model, rows: list[dict],
                conflict_cols: list[str]) -> None:
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


# ---------------------------------------------------------------------------
# Season derivation
# ---------------------------------------------------------------------------

def current_season() -> str:
    """Return the NBA season string for today, e.g. '2025-26'."""
    today = date.today()
    # NBA seasons start in October; before October belongs to the previous season
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


# ---------------------------------------------------------------------------
# DB look-up helpers for days_rest
# ---------------------------------------------------------------------------

def team_last_game_date(session: Session, team_id: int,
                        before: date) -> date | None:
    """Most recent game date for a team strictly before *before*."""
    row = session.execute(text("""
        SELECT MAX(game_date) FROM games
        WHERE (home_team_id = :tid OR away_team_id = :tid)
          AND game_date < :before
    """), {"tid": team_id, "before": before}).fetchone()
    return row[0] if row and row[0] else None


def player_last_played_date(session: Session, player_id: int,
                            before: date) -> date | None:
    """Most recent game date where this player was NOT a DNP, before *before*."""
    row = session.execute(text("""
        SELECT MAX(g.game_date)
        FROM player_box_scores_traditional pbs
        JOIN games g ON pbs.game_id = g.id
        WHERE pbs.player_id = :pid
          AND pbs.dnp_status = FALSE
          AND g.game_date < :before
    """), {"pid": player_id, "before": before}).fetchone()
    return row[0] if row and row[0] else None


def days_between(earlier: date | None, later: date) -> int | None:
    return (later - earlier).days if earlier else None


# ---------------------------------------------------------------------------
# Fetch recent game log
# ---------------------------------------------------------------------------

def _lgf_matchup_is_home_row(matchup: str) -> bool:
    """LeagueGameFinder: home rows use 'vs' / 'vs.'; away rows use '@'."""
    m = str(matchup or "").strip()
    if "@" in m:
        return False
    return bool(re.search(r"(?i)\bvs\.?\b", m))


def fetch_recent_game_log(date_from: str, date_to: str,
                          season: str) -> pd.DataFrame:
    log.info("Fetching game log  season=%s  %s -> %s", season, date_from, date_to)
    resp = api_call(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        date_from_nullable=date_from,
        date_to_nullable=date_to,
    )
    df = resp.get_data_frames()[0]
    if df.empty:
        return df
    df["GAME_DATE"]   = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["GAME_ID_INT"] = df["GAME_ID"].astype(int)
    df["IS_HOME"]     = df["MATCHUP"].astype(str).map(_lgf_matchup_is_home_row)
    log.info("  %d team-game rows across %d unique games.",
             len(df), df["GAME_ID"].nunique())
    return df


# ---------------------------------------------------------------------------
# Filter to games not yet in DB
# ---------------------------------------------------------------------------

def filter_new_games(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    candidate_ids = df["GAME_ID_INT"].unique().tolist()
    rows = session.execute(
        text("SELECT id FROM games WHERE id = ANY(:ids)"),
        {"ids": candidate_ids},
    ).fetchall()
    existing = {r[0] for r in rows}
    new_ids = [gid for gid in candidate_ids if gid not in existing]
    log.info("  %d total games fetched, %d already in DB, %d new.",
             len(candidate_ids), len(existing), len(new_ids))
    return df[df["GAME_ID_INT"].isin(new_ids)]


# ---------------------------------------------------------------------------
# Insert games + team box scores
# ---------------------------------------------------------------------------

def insert_games_and_team_boxes(session: Session, df: pd.DataFrame,
                                season: str) -> list[int]:
    log.info("Inserting games + team box scores...")
    game_rows:     list[dict] = []
    team_box_rows: list[dict] = []

    for game_id_int, gdf in df.groupby("GAME_ID_INT"):
        home_df = gdf[gdf["IS_HOME"]]
        away_df = gdf[~gdf["IS_HOME"]]
        if home_df.empty or away_df.empty:
            log.warning("  Game %d: cannot determine home/away - skipping.", game_id_int)
            continue

        home  = home_df.iloc[0]
        away  = away_df.iloc[0]
        gdate: date = home["GAME_DATE"]

        home_tid = int(home["TEAM_ID"])
        away_tid = int(away["TEAM_ID"])

        # days_rest from DB: last game for each team before this one
        h_rest = days_between(team_last_game_date(session, home_tid, gdate), gdate)
        a_rest = days_between(team_last_game_date(session, away_tid, gdate), gdate)

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
    log.info("  Inserted %d games | %d team box score rows.",
             len(game_rows), len(team_box_rows))
    return [r["id"] for r in game_rows]


# ---------------------------------------------------------------------------
# Insert player box scores
# ---------------------------------------------------------------------------

def process_player_box_scores(session: Session, game_ids: list[int]) -> None:
    log.info("Processing player box scores (%d games)...", len(game_ids))

    # Sort by game date so days_rest is computed in chronological order
    game_date_map: dict[int, date] = {}
    for r in session.execute(text("SELECT id, game_date FROM games ORDER BY game_date")):
        game_date_map[r.id] = r.game_date
    sorted_ids = sorted(game_ids, key=lambda gid: game_date_map.get(gid, date.min))

    total = len(sorted_ids)
    for idx, game_id in enumerate(sorted_ids, 1):
        gdate      = game_date_map.get(game_id)
        game_id_str = str(game_id).zfill(10)

        log.info("  [%d/%d] Game %-12s  date=%s", idx, total, game_id_str, gdate)

        # Look up the authoritative home/away team IDs we stored when the game
        # was inserted.  These are the source of truth for player team assignment.
        game_teams = session.execute(
            text("SELECT home_team_id, away_team_id FROM games WHERE id = :gid"),
            {"gid": game_id},
        ).fetchone()
        if not game_teams:
            log.error("    Game %d not found in DB after insert — skipping.", game_id)
            continue
        home_tid = game_teams.home_team_id
        away_tid = game_teams.away_team_id

        # Traditional box score
        try:
            trad_resp = api_call(
                boxscoretraditionalv3.BoxScoreTraditionalV3,
                game_id=game_id_str,
            )
        except Exception as exc:
            log.error("    SKIP trad box score - %s", exc)
            continue

        # Parse directly from the raw response dict so we can assign teamId from
        # the team-level header (home vs away section) rather than the per-player
        # field, which the NBA Stats v3 API frequently returns incorrectly (all
        # players end up with the home team's ID).
        player_trad_df = _parse_trad_v3_to_df(trad_resp.get_dict(), home_tid, away_tid)
        if player_trad_df.empty:
            log.warning("    No player rows parsed for game %d — skipping.", game_id)
            continue

        # Advanced box score
        adv_player_df = pd.DataFrame()
        try:
            adv_resp      = api_call(
                boxscoreadvancedv3.BoxScoreAdvancedV3,
                game_id=game_id_str,
            )
            adv_player_df = _parse_adv_v3_to_df(adv_resp.get_dict(), home_tid, away_tid)
        except Exception as exc:
            log.warning("    Advanced box score unavailable - %s", exc)

        # Seed / update players.  Returns {api_person_id: canonical_db_player_id}
        # and the validated {box_score_teamId: real_team_id} mapping for this game.
        player_id_map, team_id_map = _seed_players(session, player_trad_df, game_id)
        session.commit()

        # Traditional rows - days_rest looked up from DB per player
        trad_rows = _build_trad_rows(session, player_trad_df, game_id, gdate,
                                     player_id_map, team_id_map)
        upsert_rows(session, PlayerBoxScoreTraditional, trad_rows,
                    ["game_id", "player_id"])
        session.commit()

        dnp_count = sum(1 for r in trad_rows if r["dnp_status"])
        log.info("    Trad: %d rows  (played=%d  DNP=%d)",
                 len(trad_rows), len(trad_rows) - dnp_count, dnp_count)

        # Advanced rows
        if not adv_player_df.empty:
            adv_rows = _build_adv_rows(adv_player_df, game_id, player_id_map, team_id_map)
            upsert_rows(session, PlayerBoxScoreAdvanced, adv_rows,
                        ["game_id", "player_id"])
            session.commit()
            log.info("    Adv : %d rows", len(adv_rows))


# ---------------------------------------------------------------------------
# v3 API response parsers
# ---------------------------------------------------------------------------

def _get_game_section(raw: dict) -> dict:
    """
    Navigate to the dict that contains 'homeTeam' and 'awayTeam' keys,
    regardless of the outer wrapper key used by the NBA Stats v3 API
    (historically "boxScoreTraditional", "boxScoreAdvanced", or "game").
    """
    if "homeTeam" in raw and "awayTeam" in raw:
        return raw
    for v in raw.values():
        if isinstance(v, dict) and "homeTeam" in v and "awayTeam" in v:
            return v
    return {}


def _parse_trad_v3_to_df(raw: dict, home_tid: int, away_tid: int) -> pd.DataFrame:
    """
    Build a flat player DataFrame from a BoxScoreTraditionalV3 raw response dict.

    THE KEY FIX: teamId is taken from the *team-level* field in homeTeam /
    awayTeam, NOT from any per-player field.  This prevents the NBA Stats API
    bug where all players receive the same teamId (typically the home team's)
    because the v3 response nests teamId above the players array.
    """
    game_section = _get_game_section(raw)
    rows = []
    for section_key, tid in (("homeTeam", home_tid), ("awayTeam", away_tid)):
        team_sec = game_section.get(section_key, {})
        for p in team_sec.get("players", []):
            stats = p.get("statistics", {})
            rows.append({
                "personId":                  p.get("personId"),
                "firstName":                 p.get("firstName", ""),
                "familyName":                p.get("familyName", ""),
                # Authoritative team assignment — from the team section header,
                # NOT from any player-level field that the API may return wrong.
                "teamId":                    tid,
                "position":                  p.get("position", ""),
                # DNP reason: prefer machine-readable code, fall back to description
                "comment": (
                    p.get("notPlayingReason", "")
                    or p.get("notPlayingDescription", "")
                    or p.get("comment", "")
                ),
                "minutes":                   stats.get("minutes"),
                "points":                    stats.get("points"),
                "reboundsTotal":             stats.get("reboundsTotal"),
                "reboundsOffensive":         stats.get("reboundsOffensive"),
                "reboundsDefensive":         stats.get("reboundsDefensive"),
                "assists":                   stats.get("assists"),
                "steals":                    stats.get("steals"),
                "blocks":                    stats.get("blocks"),
                "turnovers":                 stats.get("turnovers"),
                "fieldGoalsMade":            stats.get("fieldGoalsMade"),
                "fieldGoalsAttempted":       stats.get("fieldGoalsAttempted"),
                "fieldGoalsPercentage":      stats.get("fieldGoalsPercentage"),
                "threePointersMade":         stats.get("threePointersMade"),
                "threePointersAttempted":    stats.get("threePointersAttempted"),
                "threePointersPercentage":   stats.get("threePointersPercentage"),
                "freeThrowsMade":            stats.get("freeThrowsMade"),
                "freeThrowsAttempted":       stats.get("freeThrowsAttempted"),
                "freeThrowsPercentage":      stats.get("freeThrowsPercentage"),
                "foulsPersonal":             stats.get("foulsPersonal"),
                "plusMinusPoints":           stats.get("plusMinusPoints"),
            })
    return pd.DataFrame(rows)


def _parse_adv_v3_to_df(raw: dict, home_tid: int, away_tid: int) -> pd.DataFrame:
    """
    Build a flat player DataFrame from a BoxScoreAdvancedV3 raw response dict.
    Same authoritative teamId fix as _parse_trad_v3_to_df.
    """
    game_section = _get_game_section(raw)
    rows = []
    for section_key, tid in (("homeTeam", home_tid), ("awayTeam", away_tid)):
        team_sec = game_section.get(section_key, {})
        for p in team_sec.get("players", []):
            stats = p.get("statistics", {})
            rows.append({
                "personId":                       p.get("personId"),
                "teamId":                         tid,
                "minutes":                        stats.get("minutes"),
                "offensiveRating":                stats.get("offensiveRating"),
                "defensiveRating":                stats.get("defensiveRating"),
                "netRating":                      stats.get("netRating"),
                "assistPercentage":               stats.get("assistPercentage"),
                "assistToTurnover":               stats.get("assistToTurnover"),
                "assistRatio":                    stats.get("assistRatio"),
                "offensiveReboundPercentage":     stats.get("offensiveReboundPercentage"),
                "defensiveReboundPercentage":     stats.get("defensiveReboundPercentage"),
                "reboundPercentage":              stats.get("reboundPercentage"),
                "effectiveFieldGoalPercentage":   stats.get("effectiveFieldGoalPercentage"),
                "trueShootingPercentage":         stats.get("trueShootingPercentage"),
                "usagePercentage":                stats.get("usagePercentage"),
                "turnoverRatio":                  stats.get("turnoverRatio"),
                "pace":                           stats.get("pace"),
                # "PIE" key differs by nba_api version — check all known names
                "PIE": (
                    stats.get("playerImpactEstimate")
                    or stats.get("pie")
                    or stats.get("PIE")
                ),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _resolve_team_id_map(session: Session, df: pd.DataFrame,
                         game_id: int) -> dict[int, int]:
    """
    Build a validated mapping of {box_score_teamId → real_team_id} for a game.

    The NBA box score API can return incorrect teamId values (e.g. all players
    tagged with the home team's ID instead of their own team's ID).  This
    function cross-references the box score's teamId values against the two
    teams stored in the games table and returns a corrected mapping.

    If the box score teamIds already match the game teams exactly, the mapping
    is identity ({tid: tid, ...}).  If there is a mismatch, it is logged and
    the mapping resolves each raw teamId to the nearest valid team (home or
    away) using the game record as the authoritative source.
    """
    game_row = session.execute(text("""
        SELECT home_team_id, away_team_id FROM games WHERE id = :gid
    """), {"gid": game_id}).fetchone()

    if not game_row:
        log.warning("    Game %d not found in DB; team_id validation skipped.", game_id)
        return {}

    home_tid, away_tid = game_row.home_team_id, game_row.away_team_id
    valid_tids = {home_tid, away_tid}

    # Unique teamId values reported in the box score for this game
    raw_tids = {int(t) for t in df["teamId"].dropna().unique()}

    # Happy path: box score matches the game record exactly
    if raw_tids.issubset(valid_tids):
        return {t: t for t in raw_tids}

    # --- Mismatch detected ---
    unexpected = raw_tids - valid_tids
    log.error(
        "    Game %d: box score teamId(s) %s do not match game teams "
        "(home=%d, away=%d).  Attempting auto-correction.",
        game_id, unexpected, home_tid, away_tid,
    )

    # Build a corrected mapping: match each raw teamId to whichever valid team
    # it is closer to numerically (last resort heuristic; works when the API
    # returns the opponent's ID instead of the player's own team).
    mapping: dict[int, int] = {}
    valid_list = [home_tid, away_tid]
    for raw in raw_tids:
        if raw in valid_tids:
            mapping[raw] = raw
        else:
            # Assign to the valid team whose ID is numerically closest
            corrected = min(valid_list, key=lambda v: abs(v - raw))
            log.warning(
                "    Mapping raw teamId %d → %d (home=%d, away=%d)",
                raw, corrected, home_tid, away_tid,
            )
            mapping[raw] = corrected

    return mapping


def _seed_players(session: Session, df: pd.DataFrame,
                  game_id: int) -> tuple[dict[int, int], dict[int, int]]:
    """
    Upsert player records from a box score DataFrame, correctly handling
    mid-season trades and box score teamId mismatches.

    Returns
    -------
    player_id_map : {api_person_id: canonical_db_player_id}
        Used by downstream row-builders to reference the right player.
    team_id_map : {raw_box_score_teamId: validated_real_team_id}
        Used by downstream row-builders to write correct team_id values
        even when the API returns wrong teamId data.
    """
    team_id_map = _resolve_team_id_map(session, df, game_id)

    # Load the current name→id map from the DB once
    name_to_db_id: dict[str, int] = {
        row.full_name: row.id
        for row in session.execute(
            text("SELECT id, full_name FROM players")
        ).fetchall()
    }

    api_to_canonical: dict[int, int] = {}
    new_player_rows: list[dict] = []
    seen: set[int] = set()

    for _, r in df.iterrows():
        api_pid = int(r["personId"])
        if api_pid in seen:
            continue
        seen.add(api_pid)

        full_name    = f"{r['firstName']} {r['familyName']}".strip()
        raw_team_id  = int(r["teamId"])
        # Use the validated team_id; fall back to raw if we have no mapping
        team_id      = team_id_map.get(raw_team_id, raw_team_id)

        existing_db_id = name_to_db_id.get(full_name)

        if existing_db_id is not None and existing_db_id != api_pid:
            # Player exists under a different ID (trade-duplicate scenario).
            # Update the canonical record's team_id and map the new api_pid to it.
            log.info(
                "    Trade handled: %s  api_id=%d → db_id=%d  new team_id=%d",
                full_name, api_pid, existing_db_id, team_id,
            )
            session.execute(text("""
                UPDATE players
                SET team_id = :team_id, is_active = TRUE
                WHERE id = :pid
            """), {"team_id": team_id, "pid": existing_db_id})
            api_to_canonical[api_pid] = existing_db_id

        else:
            # Normal path: upsert by player ID.
            # ON CONFLICT (id) DO UPDATE updates team_id to the validated value,
            # correctly reflecting the player's current team after a trade.
            new_player_rows.append({
                "id":         api_pid,
                "team_id":    team_id,          # validated — not the raw API value
                "first_name": str(r["firstName"]),
                "last_name":  str(r["familyName"]),
                "full_name":  full_name,
                "position":   str(r["position"]) if r["position"] else None,
                "is_active":  True,
            })
            api_to_canonical[api_pid] = api_pid
            name_to_db_id[full_name] = api_pid

    if new_player_rows:
        upsert_rows(session, Player, new_player_rows, ["id"])

    return api_to_canonical, team_id_map


def _build_trad_rows(session: Session, df: pd.DataFrame,
                     game_id: int, game_date: date,
                     player_id_map: dict[int, int] | None = None,
                     team_id_map:   dict[int, int] | None = None) -> list[dict]:
    df = df.drop_duplicates(subset=["personId"], keep="last")
    rows: list[dict] = []
    if player_id_map is None:
        player_id_map = {}
    if team_id_map is None:
        team_id_map = {}

    for _, r in df.iterrows():
        api_pid  = int(r["personId"])
        # Use canonical DB id (handles traded players remapped in _seed_players)
        pid      = player_id_map.get(api_pid, api_pid)
        # Use validated team_id so box score rows match the correct team
        raw_tid  = int(r["teamId"])
        team_id  = team_id_map.get(raw_tid, raw_tid)
        mins = parse_minutes(r.get("minutes"))

        is_dnp     = mins is None
        dnp_reason = classify_dnp_reason(str(r.get("comment", ""))) if is_dnp else None

        # days_rest: query DB for last game this player actually played
        last_played = player_last_played_date(session, pid, game_date)
        days_rest   = days_between(last_played, game_date)

        pts = _int(r.get("points"))
        reb = _int(r.get("reboundsTotal"))
        ast = _int(r.get("assists"))
        stl = _int(r.get("steals"))
        blk = _int(r.get("blocks"))
        tov = _int(r.get("turnovers"))

        rows.append({
            "game_id":        game_id,
            "player_id":      pid,
            "team_id":        team_id,
            "dnp_status":     is_dnp,
            "dnp_reason":     dnp_reason,
            "minutes_played": mins,
            "days_rest":      days_rest,
            "fgm":            _int(r.get("fieldGoalsMade")),
            "fga":            _int(r.get("fieldGoalsAttempted")),
            "fg_pct":         _flt(r.get("fieldGoalsPercentage")),
            "fg3m":           _int(r.get("threePointersMade")),
            "fg3a":           _int(r.get("threePointersAttempted")),
            "fg3_pct":        _flt(r.get("threePointersPercentage")),
            "ftm":            _int(r.get("freeThrowsMade")),
            "fta":            _int(r.get("freeThrowsAttempted")),
            "ft_pct":         _flt(r.get("freeThrowsPercentage")),
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
            "fantasy_pts":    calc_fantasy_pts(pts, reb, ast, stl, blk, tov),
        })
    return rows


def _build_adv_rows(df: pd.DataFrame, game_id: int,
                    player_id_map: dict[int, int] | None = None,
                    team_id_map:   dict[int, int] | None = None) -> list[dict]:
    if player_id_map is None:
        player_id_map = {}
    if team_id_map is None:
        team_id_map = {}
    df = df.drop_duplicates(subset=["personId"], keep="last")
    rows = []
    for _, r in df.iterrows():
        api_pid = int(r["personId"])
        pid     = player_id_map.get(api_pid, api_pid)
        raw_tid = int(r["teamId"])
        team_id = team_id_map.get(raw_tid, raw_tid)
        rows.append({
            "game_id":        game_id,
            "player_id":      pid,
            "team_id":        team_id,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    today     = date.today()
    date_from = (today - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%d")
    date_to   = today.strftime("%Y-%m-%d")
    season    = current_season()

    log.info("=" * 60)
    log.info("NBA Recent Games Updater")
    log.info("Lookback : %d hours  (%s -> %s)", LOOKBACK_HOURS, date_from, date_to)
    log.info("Season   : %s", season)
    log.info("=" * 60)

    with Session(db_engine) as session:
        df = fetch_recent_game_log(date_from, date_to, season)

        if df.empty:
            log.info("No games found in window. Nothing to do.")
            return

        df = filter_new_games(session, df)

        if df.empty:
            log.info("All games already in database. Nothing to do.")
            return

        game_ids = insert_games_and_team_boxes(session, df, season)
        process_player_box_scores(session, game_ids)

    log.info("")
    log.info("=" * 60)
    log.info("Update complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
