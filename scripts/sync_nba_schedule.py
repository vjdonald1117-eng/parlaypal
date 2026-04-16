"""
Sync NBA schedule for today + next 2 days into Supabase games table.

Features:
  - Pulls schedule from stats.nba.com scoreboardv3 JSON (official NBA GAME_IDs; no nba_api import).
  - Idempotent upsert on games.id (safe to run multiple times).
  - Fills home_days_rest / away_days_rest for upcoming games by looking at
    each team's most recent game_date in the DB.

Usage:
  python scripts/sync_nba_schedule.py
  python scripts/sync_nba_schedule.py --dry-run
"""

import sys

# Must run before heavy imports: SQLAlchemy can block for a long time on some Windows setups.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass
print("[sync_nba_schedule] starting (imports next…)", flush=True)

import argparse
import bisect
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import requests
import requests_cache
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter

load_dotenv()
print("[sync_nba_schedule] stdlib + requests + dotenv loaded; SQLAlchemy is deferred until DB", flush=True)

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba"
_sqlalchemy_mod = None
NBA_HTTP_CACHE_TTL_SECONDS = 15 * 60
NBA_HTTP_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
NBA_HTTP_CACHE_NAME = os.path.join(NBA_HTTP_CACHE_DIR, "nba_http_cache")

# One shared Session for NBA stats + Odds API (connection reuse per host).
def _build_http_session() -> requests.Session:
    os.makedirs(NBA_HTTP_CACHE_DIR, exist_ok=True)
    s = requests_cache.CachedSession(
        cache_name=NBA_HTTP_CACHE_NAME,
        backend="sqlite",
        expire_after=NBA_HTTP_CACHE_TTL_SECONDS,
        allowable_methods=("GET",),
    )
    # Pool connections per host; retries handled explicitly for scoreboard.
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


HTTP_SESSION = _build_http_session()

# NBA stats scoreboard (same endpoint as nba_api ScoreboardV3; we call it directly to avoid importing nba_api).
NBA_STATS_SCOREBOARD_URL = "https://stats.nba.com/stats/scoreboardv3"


def _sql():
    """Lazy import so schedule fetch can run even if SQLAlchemy init is slow or stuck."""
    global _sqlalchemy_mod
    if _sqlalchemy_mod is None:
        import sqlalchemy as sa

        _sqlalchemy_mod = sa
    return _sqlalchemy_mod


def _hb(msg: str) -> None:
    """Heartbeat logger; flush immediately to avoid buffered 'silent hang' terminals."""
    print(msg, flush=True)


# Base headers; User-Agent is rotated per retry attempt (see _nba_stats_headers).
NBA_STATS_HEADERS_BASE = {
    "Host": "stats.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/stats",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Rotate on retries — some edge blocks are UA-specific.
NBA_STATS_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Edg/136.0.0.0 Chrome/136.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/18.0 Safari/605.1.15"
    ),
]


def _nba_stats_headers(user_agent: str) -> dict[str, str]:
    h = dict(NBA_STATS_HEADERS_BASE)
    h["User-Agent"] = user_agent
    return h


# Alias (browser-like headers for stats.nba.com / scoreboardv3).
get_nba_headers = _nba_stats_headers


class ScoreboardFetchResult(NamedTuple):
    """Result of stats.nba.com scoreboardv3 fetch (never raises for HTTP errors)."""

    payload: dict[str, Any] | None
    last_http_status: int | None  # last HTTP status received, if any


def _fetch_scoreboard_payload(game_date: date) -> ScoreboardFetchResult:
    """
    GET stats.nba.com scoreboardv3 JSON (no nba_api). Backoff 2/4/8s, UA rotation.
    Uses (connect, read) timeouts so a stuck TCP/SSL connect cannot hang forever.
    """
    # (connect_seconds, read_seconds) — 30s read for slow NBA CDN responses.
    connect_timeout_s = 10
    read_timeout_s = 30
    backoff_seconds = [2, 4, 8]
    max_attempts = 1 + len(backoff_seconds)
    params = {"GameDate": game_date.isoformat(), "LeagueID": "00"}

    last_err: Exception | None = None
    last_http_status: int | None = None
    for attempt in range(max_attempts):
        ua = NBA_STATS_USER_AGENTS[attempt % len(NBA_STATS_USER_AGENTS)]
        if attempt > 0:
            wait = backoff_seconds[attempt - 1]
            _hb(
                f"[DEBUG] scoreboardv3 retry {attempt + 1}/{max_attempts} "
                f"after {wait}s backoff; UA[{attempt % len(NBA_STATS_USER_AGENTS)}]={ua[:48]}..."
            )
            time.sleep(wait)

        _hb(
            f"[DEBUG] scoreboardv3 GET attempt {attempt + 1}/{max_attempts} "
            f"date={game_date.isoformat()} "
            f"timeout=(connect={connect_timeout_s}s, read={read_timeout_s}s)"
        )
        t0 = time.monotonic()
        try:
            r = HTTP_SESSION.get(
                NBA_STATS_SCOREBOARD_URL,
                params=params,
                headers=get_nba_headers(ua),
                timeout=(connect_timeout_s, read_timeout_s),
            )
            elapsed = time.monotonic() - t0
            last_http_status = r.status_code
            cache_status = "HIT" if getattr(r, "from_cache", False) else "MISS"
            _hb(
                f"[DEBUG] scoreboardv3 HTTP status={r.status_code} date={game_date.isoformat()} "
                f"elapsed={elapsed:.2f}s cache={cache_status}"
            )
            if r.status_code != 200:
                last_err = RuntimeError(f"HTTP {r.status_code}")
                if r.status_code in (403, 429):
                    _hb(
                        f"[WARN] scoreboardv3 HTTP {r.status_code} (rate-limit or block) for "
                        f"{game_date.isoformat()} — will use Odds-only DB refresh if schedule is empty."
                    )
                continue
            data = r.json()
            if not isinstance(data, dict):
                last_err = TypeError("scoreboard JSON is not an object")
                continue
            return ScoreboardFetchResult(data, r.status_code)
        except requests.exceptions.Timeout as exc:
            last_err = exc
            elapsed = time.monotonic() - t0
            _hb(
                f"[DEBUG] scoreboardv3 Timeout (attempt {attempt + 1}) after {elapsed:.2f}s: {exc}"
            )
        except requests.RequestException as exc:
            last_err = exc
            elapsed = time.monotonic() - t0
            _hb(
                f"[DEBUG] scoreboardv3 RequestException (attempt {attempt + 1}) after {elapsed:.2f}s: {exc}"
            )
        except Exception as exc:
            last_err = exc
            elapsed = time.monotonic() - t0
            _hb(f"[DEBUG] scoreboardv3 error (attempt {attempt + 1}) after {elapsed:.2f}s: {exc}")

    _hb(f"[DEBUG] scoreboardv3 gave up after {max_attempts} attempts; last_err={last_err}")
    return ScoreboardFetchResult(None, last_http_status)


ODDS_BOOK_PRIORITY = [
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "williamhill_us",
    "espnbet",
]


def _season_for_game_date(d: date) -> str:
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    return f"{d.year - 1}-{str(d.year)[2:]}"


def _status_from_id(game_status_id: int) -> str:
    if game_status_id == 3:
        return "Final"
    if game_status_id == 2:
        return "In Progress"
    return "Scheduled"


def _get_engine():
    load_dotenv()
    print("[init] .env loaded for DB settings", flush=True)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    print("[init] Importing SQLAlchemy and creating engine…", flush=True)
    sa = _sql()
    return sa.create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def _ensure_games_market_columns(conn) -> None:
    text = _sql().text
    conn.execute(
        text(
            """
            ALTER TABLE public.games
            ADD COLUMN IF NOT EXISTS closing_spread DOUBLE PRECISION
            """
        )
    )
    conn.execute(
        text(
            """
            ALTER TABLE public.games
            ADD COLUMN IF NOT EXISTS closing_total DOUBLE PRECISION
            """
        )
    )


def _team_full_name_map(conn) -> dict[int, str]:
    rows = conn.execute(_sql().text("SELECT id, full_name FROM teams")).fetchall()
    return {int(r.id): str(r.full_name or "").strip() for r in rows}


def _odds_get(path: str, params: dict[str, str]) -> list[dict]:
    if not ODDS_API_KEY:
        _hb("[DEBUG] Odds API key missing; returning no market lines.")
        return []
    p = dict(params)
    p["apiKey"] = ODDS_API_KEY
    _hb(f"[DEBUG] Before Odds API request: path={path}")
    try:
        r = HTTP_SESSION.get(
            f"{ODDS_BASE_URL}{path}",
            params=p,
            headers={"User-Agent": "ParlayPal/1.0", "Accept": "application/json"},
            timeout=30,
        )
        cache_status = "HIT" if getattr(r, "from_cache", False) else "MISS"
        _hb(f"[DEBUG] After Odds API request: status={r.status_code} cache={cache_status}")
        if r.status_code != 200:
            return []
        data = r.json()
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    except requests.RequestException as exc:
        _hb(f"[DEBUG] Odds API request exception: {exc}")
        return []


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _extract_market_lines(event: dict) -> tuple[float | None, float | None]:
    """
    Returns (home_spread, total_points) from preferred bookmaker.
    home_spread uses home team's spread line (negative means home favored).
    total_points is O/U line.
    """
    home_name = str(event.get("home_team") or "").strip()
    away_name = str(event.get("away_team") or "").strip()
    if not home_name or not away_name:
        return None, None
    n_home = _norm_name(home_name)
    n_away = _norm_name(away_name)

    bms = event.get("bookmakers") or []
    by_key = {str(b.get("key")): b for b in bms if isinstance(b, dict)}
    ordered = [by_key[k] for k in ODDS_BOOK_PRIORITY if k in by_key] + [
        b for b in bms if isinstance(b, dict) and str(b.get("key")) not in ODDS_BOOK_PRIORITY
    ]

    for bm in ordered:
        spreads = next((m for m in (bm.get("markets") or []) if m.get("key") == "spreads"), None)
        totals = next((m for m in (bm.get("markets") or []) if m.get("key") == "totals"), None)

        home_spread: float | None = None
        total_points: float | None = None

        if spreads:
            for o in spreads.get("outcomes", []):
                nm = _norm_name(str(o.get("name") or ""))
                if nm == n_home and o.get("point") is not None:
                    try:
                        home_spread = float(o["point"])
                    except (TypeError, ValueError):
                        pass
                    break

        if totals:
            # totals outcomes usually have same point for Over/Under; take first valid
            for o in totals.get("outcomes", []):
                if o.get("point") is not None:
                    try:
                        total_points = float(o["point"])
                        break
                    except (TypeError, ValueError):
                        continue

        if home_spread is not None or total_points is not None:
            return home_spread, total_points

    return None, None


def _fetch_market_lines_for_dates(target_dates: list[date]) -> dict[tuple[str, str, str], tuple[float | None, float | None]]:
    """
    Key: (YYYY-MM-DD, normalized_home_name, normalized_away_name) -> (home_spread, total)
    """
    if not ODDS_API_KEY:
        return {}
    td = {d.isoformat() for d in target_dates}
    events = _odds_get(
        "/odds",
        {"regions": "us", "markets": "h2h,spreads,totals", "oddsFormat": "american"},
    )
    out: dict[tuple[str, str, str], tuple[float | None, float | None]] = {}
    for ev in events:
        ct = str(ev.get("commence_time") or "")
        if not ct:
            continue
        try:
            ev_date = datetime.fromisoformat(ct.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        except ValueError:
            continue
        if ev_date not in td:
            continue
        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not home or not away:
            continue
        out[(ev_date, _norm_name(home), _norm_name(away))] = _extract_market_lines(ev)
    return out


def _fetch_day_schedule(game_date: date) -> list[dict[str, Any]]:
    def _score_or_none(v: Any) -> int | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None

    _hb(f"[DEBUG] Before scoreboardv3 fetch: date={game_date.isoformat()}")
    try:
        result = _fetch_scoreboard_payload(game_date)
    except Exception as exc:
        _hb(f"[scoreboard] Unexpected error for {game_date.isoformat()}: {exc}")
        return []
    if result.payload is None:
        if result.last_http_status in (403, 429):
            _hb(
                f"[WARN] No scoreboard JSON for {game_date.isoformat()} "
                f"(HTTP {result.last_http_status}); Odds-only DB path may run if needed."
            )
        return []
    payload = result.payload
    _hb(f"[DEBUG] After scoreboardv3 fetch: date={game_date.isoformat()}")

    games = (payload.get("scoreboard") or {}).get("games") or []
    if not games:
        return []

    # Build tri-code -> team_id/score per game from nested homeTeam/awayTeam (same data as LineScore dataframe).
    team_map: dict[int, dict[str, dict[str, Any]]] = {}
    for game in games:
        if not isinstance(game, dict):
            continue
        try:
            gid = int(str(game.get("gameId")))
        except (TypeError, ValueError):
            continue
        for side in ("homeTeam", "awayTeam"):
            t = game.get(side)
            if not isinstance(t, dict):
                continue
            try:
                tri = str(t.get("teamTricode") or "").upper()
                tid = int(t["teamId"])
            except (TypeError, ValueError, KeyError):
                continue
            team_map.setdefault(gid, {})[tri] = {
                "team_id": tid,
                "score": _score_or_none(t.get("score")),
            }

    rows: list[dict[str, Any]] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        try:
            gid = int(str(game.get("gameId")))
            game_code = str(game.get("gameCode") or "")
            status_id = int(game.get("gameStatus"))
        except (TypeError, ValueError):
            continue

        # gameCode looks like "20260406/NYKATL": away 3-char tri + home 3-char tri.
        m = re.search(r"/([A-Z]{3})([A-Z]{3})$", game_code.upper())
        if not m:
            continue
        away_tri, home_tri = m.group(1), m.group(2)
        tmap = team_map.get(gid, {})
        home_rec = tmap.get(home_tri)
        away_rec = tmap.get(away_tri)
        if not home_rec or not away_rec:
            continue

        home_tid = int(home_rec["team_id"])
        away_tid = int(away_rec["team_id"])
        rows.append(
            {
                "id": gid,
                "season": _season_for_game_date(game_date),
                "season_type": "Regular Season",
                "game_date": game_date,
                "home_team_id": home_tid,
                "away_team_id": away_tid,
                "home_score": home_rec.get("score"),
                "away_score": away_rec.get("score"),
                "status": _status_from_id(status_id),
            }
        )
    return rows


def _load_games_from_db_for_dates(conn, target_dates: list[date]) -> list[dict[str, Any]]:
    """
    Load existing games rows when stats.nba.com returns nothing (403/429/timeouts).
    Lets Odds API still refresh closing_spread / closing_total for tonight's slate if IDs exist in DB.
    """
    if not target_dates:
        return []
    text = _sql().text
    mn, mx = min(target_dates), max(target_dates)
    rows = conn.execute(
        text(
            """
            SELECT id, season, season_type, game_date, home_team_id, away_team_id,
                   home_score, away_score, status, home_days_rest, away_days_rest,
                   closing_spread, closing_total
            FROM games
            WHERE game_date >= :mn AND game_date <= :mx
            ORDER BY game_date, id
            """
        ),
        {"mn": mn, "mx": mx},
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r.id),
                "season": str(r.season),
                "season_type": str(r.season_type),
                "game_date": r.game_date,
                "home_team_id": int(r.home_team_id),
                "away_team_id": int(r.away_team_id),
                "home_score": r.home_score,
                "away_score": r.away_score,
                "status": str(r.status),
                "home_days_rest": r.home_days_rest,
                "away_days_rest": r.away_days_rest,
                "closing_spread": r.closing_spread,
                "closing_total": r.closing_total,
            }
        )
    return out


def _load_team_game_dates_map(conn) -> dict[int, list[date]]:
    rows = conn.execute(
        _sql().text(
            """
            WITH team_games AS (
                SELECT home_team_id AS team_id, game_date FROM games
                UNION ALL
                SELECT away_team_id AS team_id, game_date FROM games
            )
            SELECT team_id, game_date
            FROM team_games
            WHERE game_date IS NOT NULL
            """
        )
    ).fetchall()
    team_dates: dict[int, list[date]] = {}
    for r in rows:
        team_id = int(r.team_id)
        d = r.game_date
        if d is None:
            continue
        team_dates.setdefault(team_id, []).append(d)
    for tid in list(team_dates.keys()):
        # unique + sorted so bisect can find most recent date < target game date
        team_dates[tid] = sorted(set(team_dates[tid]))
    return team_dates


def _days_between(earlier: date | None, later: date) -> int | None:
    if earlier is None:
        return None
    return int((later - earlier).days)


def _latest_prior_date(sorted_dates: list[date], target: date) -> date | None:
    idx = bisect.bisect_left(sorted_dates, target) - 1
    if idx < 0:
        return None
    return sorted_dates[idx]


def _upsert_games(conn, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = _sql().text(
        """
        INSERT INTO games (
            id, season, season_type, game_date,
            home_team_id, away_team_id,
            home_score, away_score, status,
            home_days_rest, away_days_rest,
            closing_spread, closing_total
        ) VALUES (
            :id, :season, :season_type, :game_date,
            :home_team_id, :away_team_id,
            :home_score, :away_score, :status,
            :home_days_rest, :away_days_rest,
            :closing_spread, :closing_total
        )
        ON CONFLICT (id) DO UPDATE SET
            season = EXCLUDED.season,
            season_type = EXCLUDED.season_type,
            game_date = EXCLUDED.game_date,
            home_team_id = EXCLUDED.home_team_id,
            away_team_id = EXCLUDED.away_team_id,
            status = EXCLUDED.status,
            home_score = COALESCE(EXCLUDED.home_score, games.home_score),
            away_score = COALESCE(EXCLUDED.away_score, games.away_score),
            home_days_rest = EXCLUDED.home_days_rest,
            away_days_rest = EXCLUDED.away_days_rest,
            closing_spread = COALESCE(EXCLUDED.closing_spread, games.closing_spread),
            closing_total = COALESCE(EXCLUDED.closing_total, games.closing_total)
        """
    )
    conn.execute(stmt, rows)
    return len(rows)


def _warn_missing_vegas_for_today(rows: list[dict[str, Any]], ny_today: date) -> None:
    """Alert if spread or total missing for any game on today's calendar date (America/New_York)."""
    for r in rows:
        if r["game_date"] != ny_today:
            continue
        if r.get("closing_spread") is None or r.get("closing_total") is None:
            print(
                f"ACTION REQUIRED: Vegas data missing for [Game ID {r['id']}]",
                flush=True,
            )


def _count_games_with_market_lines(rows: list[dict[str, Any]]) -> int:
    """Games where Odds API supplied at least one of spread / total for this run."""
    n = 0
    for r in rows:
        if r.get("closing_spread") is not None or r.get("closing_total") is not None:
            n += 1
    return n


def _print_sync_summary(
    *,
    games_synced: int,
    spreads_updated: int,
    dry_run: bool,
    skip_odds: bool,
) -> None:
    odds_note = " (--skip-odds)" if skip_odds else ""
    print("", flush=True)
    print("=" * 72, flush=True)
    print("SYNC SUMMARY", flush=True)
    print(
        f"  Games synced:        {games_synced}"
        f"{' (dry-run; not written)' if dry_run else ''}",
        flush=True,
    )
    print(
        f"  Spreads updated:     {spreads_updated}  "
        f"(games with spread and/or total from odds){odds_note}",
        flush=True,
    )
    print(
        "  Model status:        Not run by this script "
        "(use run_daily.bat for training + prediction)",
        flush=True,
    )
    print("=" * 72, flush=True)


def sync_schedule() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Fetch and compute rows without writing")
    ap.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip odds/spread/total fetch and only process NBA schedule.",
    )
    args = ap.parse_args()

    print("[init] sync_nba_schedule.py starting...", flush=True)
    print("[init] .env loaded at module import", flush=True)
    ny_today = datetime.now(ZoneInfo("America/New_York")).date()
    target_dates = [ny_today + timedelta(days=i) for i in range(3)]
    print(f"[init] Target dates: {[d.isoformat() for d in target_dates]}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for d in target_dates:
        day_rows = _fetch_day_schedule(d)
        all_rows.extend(day_rows)
        print(f"[fetch] {d}: {len(day_rows)} game(s)", flush=True)

    engine: Any = None
    if not all_rows:
        print(
            "[WARN] No rows from stats.nba.com; attempting Odds-only refresh from existing DB games "
            "in this date window (requires prior games + DATABASE_URL).",
            flush=True,
        )
        try:
            engine = _get_engine()
            with engine.begin() as conn:
                all_rows = _load_games_from_db_for_dates(conn, target_dates)
                if all_rows:
                    print(f"[fallback] Loaded {len(all_rows)} game row(s) from DB for odds/market merge.", flush=True)
        except Exception as exc:
            print(f"[ERROR] Odds-only DB fallback failed (will not crash): {exc}", flush=True)
            all_rows = []

    if not all_rows:
        print(
            "No schedule rows for today + next 2 days (NBA empty and no DB games in range).",
            flush=True,
        )
        _print_sync_summary(
            games_synced=0,
            spreads_updated=0,
            dry_run=args.dry_run,
            skip_odds=args.skip_odds,
        )
        return

    # Compute rest in chronological order with a cache seeded from DB.
    all_rows.sort(key=lambda r: (r["game_date"], r["id"]))
    print("[init] Building DB engine...", flush=True)
    if engine is None:
        engine = _get_engine()
    print("[init] DB engine created", flush=True)
    props_upserted = 0
    with engine.begin() as conn:
        print("[init] DB transaction opened", flush=True)
        _ensure_games_market_columns(conn)
        team_name_map = _team_full_name_map(conn)
        if args.skip_odds:
            print("[DEBUG] --skip-odds enabled; bypassing _fetch_market_lines_for_dates", flush=True)
            market_map = {}
        else:
            print("[DEBUG] Before _fetch_market_lines_for_dates", flush=True)
            market_map = _fetch_market_lines_for_dates(target_dates)
            print(f"[DEBUG] After _fetch_market_lines_for_dates: entries={len(market_map)}", flush=True)
        team_dates = _load_team_game_dates_map(conn)
        for r in all_rows:
            gdate = r["game_date"]
            home_tid = int(r["home_team_id"])
            away_tid = int(r["away_team_id"])

            home_list = team_dates.get(home_tid, [])
            away_list = team_dates.get(away_tid, [])
            home_last = _latest_prior_date(home_list, gdate)
            away_last = _latest_prior_date(away_list, gdate)
            r["home_days_rest"] = _days_between(home_last, gdate)
            r["away_days_rest"] = _days_between(away_last, gdate)
            hname = _norm_name(team_name_map.get(home_tid, ""))
            aname = _norm_name(team_name_map.get(away_tid, ""))
            spread_total = market_map.get((gdate.isoformat(), hname, aname))
            if spread_total:
                r["closing_spread"], r["closing_total"] = spread_total
            else:
                r["closing_spread"], r["closing_total"] = (None, None)

            # Add this upcoming game date into each team history so day+1/day+2
            # rest is computed from earlier scheduled games in this same sync batch.
            if home_tid not in team_dates:
                team_dates[home_tid] = [gdate]
            elif gdate not in team_dates[home_tid]:
                bisect.insort(team_dates[home_tid], gdate)
            if away_tid not in team_dates:
                team_dates[away_tid] = [gdate]
            elif gdate not in team_dates[away_tid]:
                bisect.insort(team_dates[away_tid], gdate)

        _warn_missing_vegas_for_today(all_rows, ny_today)

        if args.dry_run:
            print(f"[dry-run] would upsert {len(all_rows)} games", flush=True)
            for r in all_rows[:8]:
                print(
                    f"  id={r['id']} date={r['game_date']} "
                    f"{r['away_team_id']}@{r['home_team_id']} "
                    f"rest {r['away_days_rest']}/{r['home_days_rest']} "
                    f"status={r['status']} spread={r['closing_spread']} total={r['closing_total']}",
                    flush=True,
                )
            _print_sync_summary(
                games_synced=len(all_rows),
                spreads_updated=_count_games_with_market_lines(all_rows),
                dry_run=True,
                skip_odds=args.skip_odds,
            )
            return

        n = _upsert_games(conn, all_rows)
        print(f"[upsert] processed {n} game row(s)", flush=True)
        if not args.skip_odds:
            try:
                from player_props_sync import sync_player_props_for_dates

                props_upserted = int(sync_player_props_for_dates(conn, HTTP_SESSION, target_dates))
            except Exception as exc:
                # Keep schedule sync resilient; player-props sync should not crash the run.
                print(f"[player_props] sync failed but schedule sync will continue: {exc}", flush=True)
        else:
            print("[player_props] skipped because --skip-odds is enabled", flush=True)
        _print_sync_summary(
            games_synced=n,
            spreads_updated=_count_games_with_market_lines(all_rows),
            dry_run=False,
            skip_odds=args.skip_odds,
        )
        print(f"[player_props] Upserted rows: {props_upserted}", flush=True)


def main() -> None:
    sync_schedule()


if __name__ == "__main__":
    main()

