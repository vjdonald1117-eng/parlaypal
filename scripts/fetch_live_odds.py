"""
scripts/fetch_live_odds.py
===========================
Fetches live NBA player prop lines and odds from The Odds API for today's
and tomorrow's games.

Bookmakers  : DraftKings (priority) then FanDuel
Markets     : player_points | player_rebounds | player_assists | player_steals | player_blocks | player_threes
Regions     : us
Odds format : American

API cost
--------
  1 request for the events list + 1 per event for odds.
  Results are cached to scripts/odds_cache_<stat>.json for CACHE_TTL_MINUTES
  (default 30) to avoid repeat API charges during the same session.

Usage
-----
  # Standalone — prints the full odds table for pts
  python scripts/fetch_live_odds.py
  python scripts/fetch_live_odds.py --stat reb
  python scripts/fetch_live_odds.py --no-cache    # force fresh fetch
  python scripts/fetch_live_odds.py --book-check "James Harden" --stat pts
      # raw DK/FD rows + parsed line (API truth check; no cache)

  # Import into parlay_builder.py
  from scripts.fetch_live_odds import fetch_live_odds, OddsLine
  odds = fetch_live_odds("pts")
  entry = odds.get("jayson tatum")
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from datetime import date, datetime, timezone, timedelta
from difflib import get_close_matches
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path + env
# ---------------------------------------------------------------------------
def _find_project_root(start_dir: Path) -> Path:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "scripts").is_dir() and (candidate / "parlay-ui").is_dir():
            return candidate
    return start_dir.parent


_scripts_dir = Path(__file__).resolve().parent
_repo_dir = _find_project_root(_scripts_dir)
load_dotenv(str(_repo_dir / ".env"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
# Config
# ---------------------------------------------------------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BASE_URL     = "https://api.the-odds-api.com/v4/sports/basketball_nba"

# Priority order: if a player has a line on both books, DK wins.
BOOKMAKER_PRIORITY = ["draftkings", "fanduel"]
BOOKMAKER_LABEL    = {"draftkings": "DK", "fanduel": "FD"}

# Used for game h2h / spreads when DK/FD omit a market (still US region).
H2H_BOOKMAKER_FALLBACK = [
    "draftkings",
    "fanduel",
    "betmgm",
    "betrivers",
    "williamhill_us",
    "espnbet",
    "fanatics",
    "bovada",
    "lowvig",
]

STAT_TO_MARKET = {
    "pts": "player_points",
    "reb": "player_rebounds",
    "ast": "player_assists",
    "stl": "player_steals",
    "blk": "player_blocks",
    "fg3": "player_threes",
}

CACHE_TTL_MINUTES = 30

# Align odds dates with how the backend builds ESPN slates (America/New_York).
_NY_TZ = ZoneInfo("America/New_York")

REQUEST_HEADERS = {
    "User-Agent": "ParlayPal/1.0",
    "Accept":     "application/json",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OddsLine:
    player_name:    str
    line:           float
    over_odds:      int
    under_odds:     int
    bookmaker:      str   # "DK" | "FD"
    bookmaker_key:  str   # "draftkings" | "fanduel"
    event_id:       str
    fetched_at:     str   # ISO timestamp


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(stat: str) -> str:
    return str(_scripts_dir / f"odds_cache_{stat}.json")


def _load_cache(stat: str, target_dates: set[str]) -> dict[str, OddsLine] | None:
    path = _cache_path(stat)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    fetched_at_str = raw.get("fetched_at", "")
    if not fetched_at_str:
        return None

    cached_target_dates = raw.get("target_dates")
    if cached_target_dates is None:
        # Cache file created by older code path; ignore to prevent
        # stale date mismatches (especially around midnight).
        return None
    try:
        cached_set = set(cached_target_dates)
    except TypeError:
        return None
    if cached_set != set(target_dates):
        log.info("Odds cache ignored (target dates mismatch).")
        return None

    try:
        fetched_at = datetime.fromisoformat(fetched_at_str)
    except ValueError:
        return None

    age_minutes = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
    if age_minutes > CACHE_TTL_MINUTES:
        log.info("Odds cache expired (%.0f min old, TTL=%d min).",
                 age_minutes, CACHE_TTL_MINUTES)
        return None

    entries = raw.get("entries", {})
    result: dict[str, OddsLine] = {}
    for key, d in entries.items():
        try:
            result[key] = OddsLine(**d)
        except TypeError:
            continue

    log.info("Loaded %d odds from cache (%.0f min old).", len(result), age_minutes)
    return result


def _save_cache(stat: str, odds: dict[str, OddsLine], target_dates: set[str]) -> None:
    path = _cache_path(stat)
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "stat":       stat,
        "entries":    {k: asdict(v) for k, v in odds.items()},
        # used to avoid stale-date mismatches across calendar boundaries
        "target_dates": sorted(list(target_dates)),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log.info("Cached %d odds entries to %s", len(odds), os.path.basename(path))
    except OSError as exc:
        log.warning("Could not write odds cache: %s", exc)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _events_payload_to_list(data: dict | list | None) -> list[dict]:
    """Normalize /events JSON (array or wrapped object) to a list of event dicts."""
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("events", "data", "results"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [x for x in inner if isinstance(x, dict)]
    return []


def _get(url: str, params: dict, *, timeout: int = 15) -> dict | list | None:
    """Make a GET request; return parsed JSON or None on failure."""
    if not ODDS_API_KEY:
        log.error("ODDS_API_KEY not set in .env — cannot fetch live odds.")
        return None
    params["apiKey"] = ODDS_API_KEY
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, params=params, timeout=timeout)
    except requests.RequestException as exc:
        log.error("Odds API request failed: %s", exc)
        return None
    remaining = resp.headers.get("x-requests-remaining", "?")
    if resp.status_code != 200:
        log.error("Odds API error %d: %s  (remaining=%s)",
                  resp.status_code, resp.text[:200], remaining)
        return None
    log.debug("Odds API OK  remaining=%s", remaining)
    return resp.json()


def _event_ny_date_iso(ev: dict) -> str | None:
    ct = (ev.get("commence_time") or "").strip()
    if not ct:
        return None
    try:
        iso = ct.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso).astimezone(_NY_TZ)
        return dt.date().isoformat()
    except (ValueError, TypeError):
        return ct[:10] if len(ct) >= 10 else None


def _events_for_target_dates(events: list[dict], target_dates: set[str]) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        d = _event_ny_date_iso(ev)
        if d and d in target_dates:
            out.append(ev)
    return out


def _fetch_nba_h2h_odds_bulk() -> list[dict]:
    """
    GET /odds once: h2h + spreads for all books (omit bookmakers filter).
    """
    data = _get(
        f"{BASE_URL}/odds",
        {
            "regions":    "us",
            "markets":    "h2h,spreads",
            "oddsFormat": "american",
        },
        timeout=45,
    )
    events = _events_payload_to_list(data)
    log.info("Odds API bulk /odds: %d event(s).", len(events))
    return events


def _fetch_nba_events(target_dates: set[str]) -> list[dict]:
    """
    Return events whose date in America/New_York is in target_dates (set of 'YYYY-MM-DD').
    Costs 1 API request.
    """
    data = _get(f"{BASE_URL}/events", {})
    events = _events_payload_to_list(data)
    if not events:
        return []
    matching = []
    for ev in events:
        ev_date = _event_ny_date_iso(ev)
        if ev_date and ev_date in target_dates:
            matching.append(ev)
    log.info("Events API: %d total, %d match target dates %s.",
             len(events), len(matching), sorted(target_dates))
    return matching


def _fetch_event_odds(
    event_id: str,
    market: str,
    *,
    restrict_bookmakers: bool = True,
) -> dict | None:
    """
    Fetch odds for one event. Player props use DK/FD only by default.
    H2H uses restrict_bookmakers=False so all US books are returned.
    """
    params: dict[str, str] = {
        "regions":    "us",
        "markets":    market,
        "oddsFormat": "american",
    }
    if restrict_bookmakers:
        params["bookmakers"] = ",".join(BOOKMAKER_PRIORITY)
    return _get(f"{BASE_URL}/events/{event_id}/odds", params, timeout=25)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _main_line_juice_score(over_price: int, under_price: int) -> float:
    """
    Lower = closer to a typical main market (-110 / -110). Alternate lines
    usually skew one side; use this to pick one line when the API returns
    multiple O/U pairs for the same player.
    """
    return float(abs(over_price + 110) + abs(under_price + 110))


def _parse_event_odds(
    event_data: dict,
    market: str,
    fetched_at: str,
) -> dict[str, OddsLine]:
    """
    Extract one OddsLine per player from a single event's odds response.
    Priority: DraftKings > FanDuel.
    Keyed by lowercase player name.

    When a book returns several lines per player (main + alternates), outcomes
    are grouped by (player, point); we keep the pair whose juice is closest to
    -110/-110 so the displayed line matches the primary board.
    """
    event_id  = event_data.get("id", "")
    per_player: dict[str, OddsLine] = {}

    # Process bookmakers in priority order so DK wins over FD
    bookmakers_by_key = {
        bm["key"]: bm for bm in event_data.get("bookmakers", [])
    }

    for bm_key in BOOKMAKER_PRIORITY:
        bm = bookmakers_by_key.get(bm_key)
        if not bm:
            continue
        market_data = next(
            (m for m in bm.get("markets", []) if m["key"] == market), None
        )
        if not market_data:
            continue

        # player -> point -> {"over": {...}, "under": {...}}
        by_player_point: dict[str, dict[float, dict[str, dict[str, float | int]]]] = (
            defaultdict(dict)
        )
        for o in market_data.get("outcomes", []):
            player = o.get("description", "").strip()
            if not player or o.get("point") is None:
                continue
            side = (o.get("name") or "").lower()
            if side not in ("over", "under"):
                continue
            try:
                pt = float(o["point"])
            except (TypeError, ValueError):
                continue
            cell = by_player_point[player].setdefault(pt, {})
            cell[side] = {
                "price": int(o["price"]),
                "point": pt,
            }

        for player, by_pt in by_player_point.items():
            complete: list[tuple[float, dict[str, float | int], dict[str, float | int]]] = []
            for pt, sides in by_pt.items():
                over_data  = sides.get("over")
                under_data = sides.get("under")
                if not over_data or not under_data:
                    continue
                complete.append((pt, over_data, under_data))

            if not complete:
                continue

            if len(complete) == 1:
                _, over_data, under_data = complete[0]
            else:
                _, over_data, under_data = min(
                    complete,
                    key=lambda t: _main_line_juice_score(
                        int(t[1]["price"]), int(t[2]["price"])
                    ),
                )

            key = player.lower()
            # Only write if we haven't already from a higher-priority book
            if key not in per_player:
                per_player[key] = OddsLine(
                    player_name   = player,
                    line          = float(over_data["point"]),
                    over_odds     = int(over_data["price"]),
                    under_odds    = int(under_data["price"]),
                    bookmaker     = BOOKMAKER_LABEL[bm_key],
                    bookmaker_key = bm_key,
                    event_id      = event_id,
                    fetched_at    = fetched_at,
                )

    return per_player


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_live_odds(
    stat:          str = "pts",
    use_cache:     bool = True,
    include_today: bool = True,
) -> dict[str, OddsLine]:
    """
    Return a dict mapping lowercase player name -> OddsLine for the given stat.

    Checks the local cache first (TTL = CACHE_TTL_MINUTES).
    Falls back to a live API call if the cache is stale or missing.

    Parameters
    ----------
    stat          : "pts" | "reb" | "ast" | "stl" | "blk" | "fg3"
    use_cache     : set False to force a fresh API fetch
    include_today : also include today's games (for same-day use)
    """
    stat = stat.lower()
    if stat not in STAT_TO_MARKET:
        log.error("Unknown stat %r — valid options: %s", stat, list(STAT_TO_MARKET))
        return {}

    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set — skipping live odds fetch.")
        return {}

    market = STAT_TO_MARKET[stat]
    # Align "today/tomorrow" with how the backend computes ESPN slate dates.
    today = datetime.now(_NY_TZ).date()
    target_dates = {
        today.strftime("%Y-%m-%d"),
        (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    if not include_today:
        target_dates = {(today + timedelta(days=1)).strftime("%Y-%m-%d")}

    # Try cache with matching target dates
    if use_cache:
        cached = _load_cache(stat, target_dates)
        if cached is not None:
            return cached

    # 1. Get event IDs
    events = _fetch_nba_events(target_dates)
    if not events:
        log.warning("No NBA events found for %s.", sorted(target_dates))
        return {}

    # 2. Fetch odds per event
    fetched_at = datetime.now(timezone.utc).isoformat()
    all_odds:  dict[str, OddsLine] = {}
    lines_found = 0

    for ev in events:
        eid       = ev["id"]
        home      = ev.get("home_team", "")
        away      = ev.get("away_team", "")
        ev_date   = ev.get("commence_time", "")[:10]
        log.info("Fetching %s odds: %s @ %s  (%s) ...", market, away, home, ev_date)

        ev_data = _fetch_event_odds(eid, market)
        if not ev_data:
            log.warning("  No data returned for event %s", eid)
            continue

        ev_odds = _parse_event_odds(ev_data, market, fetched_at)
        log.info("  Found %d player line(s).", len(ev_odds))
        lines_found += len(ev_odds)
        all_odds.update(ev_odds)   # DK-priority already applied per event

    log.info("Live odds: %d total player line(s) across %d event(s).",
             lines_found, len(events))

    if all_odds:
        _save_cache(stat, all_odds, target_dates)

    return all_odds


def run_book_check(
    name_substring: str,
    stat: str = "pts",
    *,
    include_today: bool = True,
) -> None:
    """
    Print every raw Over/Under row from DK/FD for players whose description
    contains name_substring (case-insensitive). Use to verify API vs app line.
    Always hits the API (no file cache) for event odds.
    """
    stat = stat.lower()
    if stat not in STAT_TO_MARKET:
        log.error("Unknown stat %r", stat)
        return
    if not ODDS_API_KEY:
        log.error("ODDS_API_KEY not set — cannot run book check.")
        return

    market = STAT_TO_MARKET[stat]
    today = datetime.now(_NY_TZ).date()
    target_dates = {
        today.strftime("%Y-%m-%d"),
        (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    }
    if not include_today:
        target_dates = {(today + timedelta(days=1)).strftime("%Y-%m-%d")}

    needle = name_substring.strip().lower()
    if not needle:
        log.error("Empty name substring.")
        return

    events = _fetch_nba_events(target_dates)
    if not events:
        print("No events for target dates.")
        return

    print(
        f"\nBook check  stat={stat}  market={market}  filter={name_substring!r}  "
        f"dates={sorted(target_dates)}\n"
    )

    for ev in events:
        eid = ev.get("id")
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        ev_data = _fetch_event_odds(str(eid), market)
        if not ev_data or not isinstance(ev_data, dict):
            continue

        bookmakers_by_key = {
            bm["key"]: bm for bm in ev_data.get("bookmakers", []) if isinstance(bm, dict)
        }
        printed_header = False
        for bm_key in BOOKMAKER_PRIORITY:
            bm = bookmakers_by_key.get(bm_key)
            if not bm:
                continue
            market_data = next(
                (m for m in bm.get("markets", []) if m.get("key") == market),
                None,
            )
            if not market_data:
                continue
            rows: list[str] = []
            for o in market_data.get("outcomes", []):
                desc = (o.get("description") or "").strip()
                if needle not in desc.lower():
                    continue
                name = (o.get("name") or "").strip()
                price = o.get("price")
                point = o.get("point")
                rows.append(
                    f"    {BOOKMAKER_LABEL.get(bm_key, bm_key):<4}  {name:<5}  "
                    f"pt={point!s:<8}  {price!s:>5}  {desc}"
                )
            if rows:
                if not printed_header:
                    print(f"--- {away} @ {home}  event={eid} ---")
                    printed_header = True
                for line in sorted(rows):
                    print(line)

        parsed = _parse_event_odds(
            ev_data, market, datetime.now(timezone.utc).isoformat()
        )
        for key, ol in parsed.items():
            if needle in key:
                print(
                    f"  => parser chose: {ol.player_name}  line={ol.line}  "
                    f"{ol.bookmaker}  O {ol.over_odds:+d} / U {ol.under_odds:+d}"
                )


def lookup_player_odds(
    player_name: str,
    odds_dict:   dict[str, OddsLine],
    cutoff:      float = 0.82,
) -> OddsLine | None:
    """
    Look up a player in the odds dict.
    Tries exact lowercase match first, then difflib fuzzy match.
    """
    if not odds_dict:
        return None
    key = player_name.strip().lower()
    if key in odds_dict:
        return odds_dict[key]
    candidates = get_close_matches(key, odds_dict.keys(), n=1, cutoff=cutoff)
    if candidates:
        log.debug("Odds fuzzy match: %r -> %r", player_name, candidates[0])
        return odds_dict[candidates[0]]
    return None


# ---------------------------------------------------------------------------
# EV helpers (exported so parlay_builder can use them)
# ---------------------------------------------------------------------------

def american_to_breakeven(american_odds: int) -> float:
    """Return the breakeven win% for given American odds."""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100) * 100
    return 100 / (american_odds + 100) * 100


def _american_to_implied_decimal(american: int) -> float:
    """Implied win probability (0–1) from American odds."""
    a = int(american)
    if a >= 0:
        return 100.0 / (a + 100.0)
    return float(abs(a)) / (float(abs(a)) + 100.0)


def _h2h_prices_from_bookmaker(bm: dict) -> dict[str, int] | None:
    market_data = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
    if not market_data:
        return None
    prices: dict[str, int] = {}
    for o in market_data.get("outcomes", []):
        name = (o.get("name") or "").strip()
        price = o.get("price")
        if name and price is not None:
            try:
                prices[name] = int(price)
            except (TypeError, ValueError):
                continue
    return prices if len(prices) >= 2 else None


def _parse_h2h_prices(event_data: dict) -> dict[str, int] | None:
    """Bookmaker h2h: team display name -> American price (DK/FD first, then fallbacks)."""
    bookmakers = event_data.get("bookmakers") or []
    bookmakers_by_key = {bm["key"]: bm for bm in bookmakers if isinstance(bm, dict) and bm.get("key")}
    for bm_key in H2H_BOOKMAKER_FALLBACK:
        bm = bookmakers_by_key.get(bm_key)
        if not bm:
            continue
        got = _h2h_prices_from_bookmaker(bm)
        if got:
            return got
    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        got = _h2h_prices_from_bookmaker(bm)
        if got:
            return got
    return None


def _h2h_price_for_team(prices: dict[str, int], team_label: str) -> int | None:
    """
    Map Odds API / ESPN-style team name to h2h price. Names often differ
    slightly (e.g. 'Los Angeles Lakers' vs 'Lakers' in outcomes).
    """
    if not team_label or not prices:
        return None
    if team_label in prices:
        return prices[team_label]
    tl = team_label.lower().strip()
    for k, v in prices.items():
        if k.lower() == tl:
            return v
    keys = list(prices.keys())
    matches = get_close_matches(team_label, keys, n=1, cutoff=0.55)
    if matches:
        return prices[matches[0]]
    last = tl.split()[-1] if tl.split() else ""
    if len(last) >= 4:
        candidates = [k for k in keys if last in k.lower()]
        if len(candidates) == 1:
            return prices[candidates[0]]
    return None


def _spreads_from_bookmaker(bm: dict) -> dict[str, tuple[float, int]] | None:
    market_data = next((m for m in bm.get("markets", []) if m.get("key") == "spreads"), None)
    if not market_data:
        return None
    out: dict[str, tuple[float, int]] = {}
    for o in market_data.get("outcomes", []):
        name = (o.get("name") or "").strip()
        point = o.get("point")
        price = o.get("price")
        if not name or point is None or price is None:
            continue
        try:
            out[name] = (float(point), int(price))
        except (TypeError, ValueError):
            continue
    return out if len(out) >= 2 else None


def _parse_spreads_by_team(event_data: dict) -> dict[str, tuple[float, int]] | None:
    """Bookmaker spreads: team display name -> (point, American price)."""
    bookmakers = event_data.get("bookmakers") or []
    bookmakers_by_key = {bm["key"]: bm for bm in bookmakers if isinstance(bm, dict) and bm.get("key")}
    for bm_key in H2H_BOOKMAKER_FALLBACK:
        bm = bookmakers_by_key.get(bm_key)
        if not bm:
            continue
        got = _spreads_from_bookmaker(bm)
        if got:
            return got
    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        got = _spreads_from_bookmaker(bm)
        if got:
            return got
    return None


def _spread_point_price_for_team(
    by_team: dict[str, tuple[float, int]],
    team_label: str,
) -> tuple[float, int] | None:
    if not team_label or not by_team:
        return None
    if team_label in by_team:
        return by_team[team_label]
    tl = team_label.lower().strip()
    for k, v in by_team.items():
        if k.lower() == tl:
            return v
    keys = list(by_team.keys())
    matches = get_close_matches(team_label, keys, n=1, cutoff=0.55)
    if matches:
        return by_team[matches[0]]
    last = tl.split()[-1] if tl.split() else ""
    if len(last) >= 4:
        candidates = [k for k in keys if last in k.lower()]
        if len(candidates) == 1:
            return by_team[candidates[0]]
    return None


def fetch_h2h_moneyline_board(target_dates: set[str]) -> dict[str, Any]:
    """
    Pull game markets from The Odds API (h2h + spreads).

    Prefers one bulk GET /odds (all bookmakers). Falls back to /events + per-event
    /odds with all books if bulk is empty.

    Returns projection-shaped rows split for the UI:
      - overs  : up to 10 favorites (highest ML implied win %)
      - unders : up to 10 underdogs

    When spreads exist, Line = spread number and verdict encodes ML + spread juice.
    This is market consensus (not an internal sim). Step toward modeled sides is a
    separate ratings / margin model.

    Requires ODDS_API_KEY. On failure, overs/unders may be empty.
    """
    bulk = _events_for_target_dates(_fetch_nba_h2h_odds_bulk(), target_dates)
    events = bulk
    per_event_fetch = False
    if not events:
        events = _fetch_nba_events(target_dates)
        per_event_fetch = True
    if not events:
        log.info("H2H: no events for dates %s", sorted(target_dates))
        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "overs": [],
            "unders": [],
            "n_games": 0,
        }

    favorites: list[dict[str, Any]] = []
    underdogs: list[dict[str, Any]] = []

    for ev in events:
        eid = ev.get("id")
        home = (ev.get("home_team") or "").strip()
        away = (ev.get("away_team") or "").strip()
        if not eid or not home or not away:
            continue
        if per_event_fetch:
            raw = _fetch_event_odds(str(eid), "h2h,spreads", restrict_bookmakers=False)
            if not raw or not isinstance(raw, dict):
                raw = _fetch_event_odds(str(eid), "h2h", restrict_bookmakers=False)
        else:
            raw = ev if isinstance(ev, dict) else None
        if not raw or not isinstance(raw, dict):
            continue
        prices = _parse_h2h_prices(raw)
        if not prices:
            continue
        ph = _h2h_price_for_team(prices, home)
        pa = _h2h_price_for_team(prices, away)
        if ph is None or pa is None:
            continue
        ih, ia = _american_to_implied_decimal(ph), _american_to_implied_decimal(pa)
        matchup = f"{away} @ {home}"

        if ih >= ia:
            fav_team, fav_am, fav_imp = home, ph, ih
            dog_team, dog_am, dog_imp = away, pa, ia
        else:
            fav_team, fav_am, fav_imp = away, pa, ia
            dog_team, dog_am, dog_imp = home, ph, ih

        spreads_map = _parse_spreads_by_team(raw)
        fav_line: float
        dog_line: float
        fav_verdict: str
        dog_verdict: str
        fav_name: str
        dog_name: str

        if spreads_map:
            fav_spread = _spread_point_price_for_team(spreads_map, fav_team)
            dog_spread = _spread_point_price_for_team(spreads_map, dog_team)
            if fav_spread and dog_spread:
                fav_pt, fav_sp_am = fav_spread
                dog_pt, dog_sp_am = dog_spread
                fav_line = float(fav_pt)
                dog_line = float(dog_pt)
                fav_name = f"{fav_team} ({fav_pt:+.1f})"
                dog_name = f"{dog_team} ({dog_pt:+.1f})"
                fav_verdict = f"ML {fav_am:+d} · spread {fav_pt:+.1f} ({fav_sp_am:+d})"
                dog_verdict = f"ML {dog_am:+d} · spread {dog_pt:+.1f} ({dog_sp_am:+d})"
            else:
                fav_line = float(fav_am)
                dog_line = float(dog_am)
                fav_name = f"{fav_team} ML"
                dog_name = f"{dog_team} ML"
                fav_verdict = "Implied favorite (h2h)"
                dog_verdict = "Underdog (h2h)"
        else:
            fav_line = float(fav_am)
            dog_line = float(dog_am)
            fav_name = f"{fav_team} ML"
            dog_name = f"{dog_team} ML"
            fav_verdict = "Implied favorite (h2h)"
            dog_verdict = "Underdog (h2h)"

        favorites.append(
            {
                "player_name": fav_name,
                "team_abbr": "",
                "opponent": dog_team,
                "matchup": matchup,
                "line": fav_line,
                "heuristic_mean": fav_imp,
                "win_probability": fav_imp,
                "ev_per_110": 0.0,
                "verdict": fav_verdict,
                "best_side": "OVER",
                "stat": "ml",
                "ensemble_lock": False,
            }
        )
        underdogs.append(
            {
                "player_name": dog_name,
                "team_abbr": "",
                "opponent": fav_team,
                "matchup": matchup,
                "line": dog_line,
                "heuristic_mean": dog_imp,
                "win_probability": dog_imp,
                "ev_per_110": 0.0,
                "verdict": dog_verdict,
                "best_side": "UNDER",
                "stat": "ml",
                "ensemble_lock": False,
            }
        )

    favorites.sort(key=lambda r: float(r.get("win_probability") or 0), reverse=True)
    underdogs.sort(key=lambda r: float(r.get("win_probability") or 0))

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "overs": favorites[:10],
        "unders": underdogs[:10],
        "n_games": len(favorites),
    }


def compute_ev_american(win_pct: float, american_odds: int, stake: float = 110.0) -> float:
    """
    EV of a single bet of *stake* dollars at *american_odds*.

    win_pct      : 0–100 float (e.g. 57.3)
    american_odds: e.g. -115, +130
    stake        : dollar amount risked (default $110 to match existing display)

    Examples
    --------
      compute_ev_american(57.3, -110, 110) -> standard EV
      compute_ev_american(57.3, -115, 110) -> harder to beat breakeven
    """
    p = win_pct / 100
    if american_odds < 0:
        profit = stake * (100 / abs(american_odds))
    else:
        profit = stake * (american_odds / 100)
    return round(p * profit - (1 - p) * stake, 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_odds_table(odds: dict[str, OddsLine], stat: str) -> None:
    if not odds:
        print("  No live odds found.")
        return
    print(f"\n  Live {stat.upper()} lines — {len(odds)} players\n")
    print(f"  {'Player':<28}  {'Book':<4}  {'Line':>5}  {'Over':>6}  {'Under':>6}")
    print("  " + "-" * 56)
    for entry in sorted(odds.values(), key=lambda x: x.player_name):
        be = american_to_breakeven(entry.over_odds)
        print(
            f"  {entry.player_name:<28}  {entry.bookmaker:<4}  "
            f"{entry.line:>5.1f}  {entry.over_odds:>+6}  {entry.under_odds:>+6}"
            f"  (BE {be:.1f}%)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live NBA player prop odds")
    parser.add_argument(
        "--stat",
        default="pts",
        choices=list(STAT_TO_MARKET.keys()),
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="Force a fresh API fetch, ignoring the local cache")
    parser.add_argument(
        "--book-check",
        metavar="NAME",
        help="Print raw DK/FD outcomes for players matching this substring (case-insensitive)",
    )
    args = parser.parse_args()
    if args.book_check:
        run_book_check(args.book_check, args.stat)
    else:
        result = fetch_live_odds(args.stat, use_cache=not args.no_cache)
        _print_odds_table(result, args.stat)
