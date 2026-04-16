"""
Fetch player prop lines from The Odds API (player_points, player_rebounds, player_assists)
and upsert into public.player_props (game_id, player_id, prop_stat, vegas_line).

Called from sync_nba_schedule after games are synced. Requires ODDS_API_KEY.
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ODDS_BOOK_PRIORITY = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
]

PROP_MARKETS = "player_points,player_rebounds,player_assists"
STAT_BY_MARKET = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
}


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _ensure_player_props_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS public.player_props (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL REFERENCES public.games(id) ON DELETE CASCADE,
                player_id INTEGER NOT NULL REFERENCES public.players(id) ON DELETE CASCADE,
                prop_stat VARCHAR(10) NOT NULL,
                vegas_line DOUBLE PRECISION,
                bookmaker TEXT NOT NULL DEFAULT 'draftkings',
                predicted_value DOUBLE PRECISION,
                over_probability DOUBLE PRECISION,
                simulation_confidence DOUBLE PRECISION,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (game_id, player_id, prop_stat)
            )
            """
        )
    )
    for col, typ in [
        ("predicted_value", "DOUBLE PRECISION"),
        ("over_probability", "DOUBLE PRECISION"),
        ("simulation_confidence", "DOUBLE PRECISION"),
    ]:
        conn.execute(
            text(
                f"ALTER TABLE public.player_props ADD COLUMN IF NOT EXISTS {col} {typ}"
            )
        )
    conn.execute(
        text(
            "ALTER TABLE public.player_props ADD COLUMN IF NOT EXISTS line_source TEXT NOT NULL DEFAULT 'EST'"
        )
    )


def _load_player_name_map(conn) -> dict[str, list[int]]:
    rows = conn.execute(text("SELECT id, full_name FROM players WHERE is_active = TRUE")).fetchall()
    m: dict[str, list[int]] = {}
    for r in rows:
        k = _norm_name(str(r.full_name or ""))
        if not k:
            continue
        m.setdefault(k, []).append(int(r.id))
    return m


def _load_games_index(conn, target_dates: list[date]) -> dict[tuple[str, str, str], int]:
    if not target_dates:
        return {}
    mn, mx = min(target_dates), max(target_dates)
    rows = conn.execute(
        text(
            """
            SELECT g.id, g.game_date, th.full_name AS home_name, ta.full_name AS away_name
            FROM games g
            JOIN teams th ON th.id = g.home_team_id
            JOIN teams ta ON ta.id = g.away_team_id
            WHERE g.game_date >= :mn AND g.game_date <= :mx
            """
        ),
        {"mn": mn, "mx": mx},
    ).fetchall()
    out: dict[tuple[str, str, str], int] = {}
    for r in rows:
        gd = r.game_date.isoformat() if hasattr(r.game_date, "isoformat") else str(r.game_date)
        hk = _norm_name(str(r.home_name or ""))
        ak = _norm_name(str(r.away_name or ""))
        out[(gd, hk, ak)] = int(r.id)
    return out


def _resolve_player_id(description: str, name_map: dict[str, list[int]]) -> int | None:
    d = (description or "").strip()
    if not d:
        return None
    k = _norm_name(d)
    ids = name_map.get(k)
    if ids and len(ids) == 1:
        return ids[0]
    # Try "First Last" vs "Last, First"
    parts = d.replace(",", " ").split()
    if len(parts) >= 2:
        k2 = _norm_name(parts[-1] + parts[0])
        ids = name_map.get(k2)
        if ids and len(ids) == 1:
            return ids[0]
    return None


def _extract_prop_lines_from_event(event: dict[str, Any]) -> list[tuple[str, str, float, str, str]]:
    """
    Returns list of (prop_stat, player_description, line, bookmaker_key, line_source).
    Strictly prioritizes real books in this order:
    draftkings -> fanduel -> betmgm -> caesars.
    If none of those books exist in this event payload, line_source is EST.
    """
    bms = event.get("bookmakers") or []
    by_key = {}
    for b in bms:
        if not isinstance(b, dict):
            continue
        k = str(b.get("key") or "").strip().lower()
        if not k:
            continue
        by_key[k] = b

    ordered_real_books = [by_key[k] for k in ODDS_BOOK_PRIORITY if k in by_key]
    has_real_market_books = len(ordered_real_books) > 0
    if has_real_market_books:
        ordered_books = ordered_real_books
        line_source = "MARKET"
    else:
        ordered_books = [b for b in bms if isinstance(b, dict)]
        line_source = "EST"

    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, float, str, str]] = []

    for bm in ordered_books:
        bk = str(bm.get("key") or "unknown").strip().lower()
        for market in bm.get("markets") or []:
            if not isinstance(market, dict):
                continue
            mkey = str(market.get("key") or "")
            stat = STAT_BY_MARKET.get(mkey)
            if not stat:
                continue
            # Pair Over outcomes (line is on the Over side)
            by_player: dict[str, float] = {}
            for o in market.get("outcomes") or []:
                if not isinstance(o, dict):
                    continue
                nm = str(o.get("name") or "").strip().lower()
                if nm != "over":
                    continue
                desc = str(o.get("description") or "").strip()
                pt = o.get("point")
                if not desc or pt is None:
                    continue
                try:
                    line = float(pt)
                except (TypeError, ValueError):
                    continue
                by_player[desc] = line
            for desc, line in by_player.items():
                sig = (stat, _norm_name(desc))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append((stat, desc, line, bk, line_source))
    return out


def sync_player_props_for_dates(conn, http_session, target_dates: list[date]) -> int:
    """
    Fetch Odds API player props and upsert player_props rows.
    Returns number of rows upserted.
    """
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key or not target_dates:
        return 0

    _ensure_player_props_table(conn)
    name_map = _load_player_name_map(conn)
    games_ix = _load_games_index(conn, target_dates)
    td = {d.isoformat() for d in target_dates}

    r = http_session.get(
        f"{ODDS_BASE}/odds",
        params={
            "regions": "us",
            "markets": PROP_MARKETS,
            "oddsFormat": "american",
            "apiKey": api_key,
        },
        headers={"User-Agent": "ParlayPal/1.0", "Accept": "application/json"},
        timeout=45,
    )
    if r.status_code != 200:
        print(f"[player_props] Odds API HTTP {r.status_code}; skipping prop sync.", flush=True)
        return 0

    events = r.json()
    if not isinstance(events, list):
        return 0

    rows: list[dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ct = str(ev.get("commence_time") or "")
        if not ct:
            continue
        try:
            ev_date = (
                datetime.fromisoformat(ct.replace("Z", "+00:00"))
                .astimezone(ZoneInfo("America/New_York"))
                .date()
                .isoformat()
            )
        except ValueError:
            continue
        if ev_date not in td:
            continue
        home = str(ev.get("home_team") or "").strip()
        away = str(ev.get("away_team") or "").strip()
        if not home or not away:
            continue
        gid = games_ix.get((ev_date, _norm_name(home), _norm_name(away)))
        if gid is None:
            gid = games_ix.get((ev_date, _norm_name(away), _norm_name(home)))
        if gid is None:
            continue

        event_market = 0
        event_est = 0
        for stat, desc, line, bk, line_source in _extract_prop_lines_from_event(ev):
            pid = _resolve_player_id(desc, name_map)
            if pid is None:
                continue
            if line_source == "MARKET":
                event_market += 1
            else:
                event_est += 1
            rows.append(
                {
                    "game_id": gid,
                    "player_id": pid,
                    "prop_stat": stat,
                    "vegas_line": line,
                    "bookmaker": bk,
                    "line_source": line_source,
                }
            )
        print(
            f"[player_props] {away} @ {home} ({ev_date}) Props Found: {event_market} Market / {event_est} EST",
            flush=True,
        )

    if not rows:
        print("[player_props] No prop rows matched to games/players.", flush=True)
        return 0

    stmt = text(
        """
        INSERT INTO public.player_props (
            game_id, player_id, prop_stat, vegas_line, bookmaker, line_source,
            predicted_value, over_probability, simulation_confidence, updated_at
        ) VALUES (
            :game_id, :player_id, :prop_stat, :vegas_line, :bookmaker, :line_source,
            NULL, NULL, NULL, NOW()
        )
        ON CONFLICT (game_id, player_id, prop_stat) DO UPDATE SET
            vegas_line = EXCLUDED.vegas_line,
            bookmaker = EXCLUDED.bookmaker,
            line_source = EXCLUDED.line_source,
            updated_at = NOW()
        """
    )
    for row in rows:
        conn.execute(stmt, row)

    print(f"[player_props] Upserted {len(rows)} prop line(s).", flush=True)
    return len(rows)
