"""
Live NBA Stats lookups to grade prediction_log rows when DB box scores are missing.
Used by api.py for /api/history and /api/update-history.
"""

from __future__ import annotations

import re
import time
import unicodedata
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3

import update_recent_games as urg

# Mirror api.py — avoid importing api (circular).
_TEAM_ABBR_ALIASES: dict[str, str] = {
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "GS": "GSW",
    "UT": "UTA",
    "WSH": "WAS",
    "PHO": "PHX",
    "CHO": "CHA",
    "BK": "BKN",
}


def _canon_abbr(a: str) -> str:
    x = (a or "").strip().upper()
    return _TEAM_ABBR_ALIASES.get(x, x)


def _lgf_matchup_is_home_row(matchup: str) -> bool:
    """
    LeagueGameFinder home rows look like 'WAS vs. BOS' or 'WAS vs BOS';
    away rows look like 'BOS @ WAS'. A strict 'vs.' check misses 'vs' without a dot
    and skips every game, so resolve never runs.
    """
    m = str(matchup or "").strip()
    if "@" in m:
        return False
    return bool(re.search(r"(?i)\bvs\.?\b", m))


def _boxscore_traditional_is_final(raw: dict) -> bool:
    """Only grade from NBA box score when the game is finished (avoid in-progress stats)."""
    found_status: int | None = None
    found_text = ""

    def walk(d: Any) -> None:
        nonlocal found_status, found_text
        if not isinstance(d, dict):
            return
        if "gameStatus" in d and found_status is None:
            try:
                found_status = int(d["gameStatus"])
            except (TypeError, ValueError):
                pass
        if "gameStatusText" in d and not found_text:
            found_text = str(d["gameStatusText"] or "").strip()
        for v in d.values():
            walk(v)

    walk(raw)
    if found_status is not None:
        return found_status == 3
    return found_text.lower().startswith("final")


def _strip_diacritics(s: str) -> str:
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _norm_player_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s).strip()).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", s, flags=re.I)
    return _strip_diacritics(s.strip())


def _players_match(api_first_last: str, log_name: str) -> bool:
    a = _norm_player_key(api_first_last)
    b = _norm_player_key(log_name)
    if a == b:
        return True
    pa, pb = a.split(), b.split()
    if len(pa) >= 2 and len(pb) >= 2:
        return pa[-1] == pb[-1] and pa[0][:1] == pb[0][:1]
    return False


def _nba_season_string(game_date_iso: str) -> str:
    d = datetime.strptime(game_date_iso[:10], "%Y-%m-%d").date()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


# game_id (zero-padded) -> parsed DataFrame from _parse_trad_v3_to_df
_trad_cache: dict[str, pd.DataFrame] = {}
# YYYY-MM-DD -> LeagueGameFinder dataframe (one NBA call per slate date per batch, not per player)
_lgf_by_date: dict[str, pd.DataFrame] = {}


def _league_games_df_for_date(gd: str, *, sleep_s: float) -> Optional[pd.DataFrame]:
    """Cached LeagueGameFinder result for a calendar day (Regular Season then Playoffs)."""
    if len(gd) != 10:
        return None
    if gd in _lgf_by_date:
        cached = _lgf_by_date[gd]
        return None if cached.empty else cached

    season = _nba_season_string(gd)
    df_found: Optional[pd.DataFrame] = None
    for stype in ("Regular Season", "Playoffs"):
        try:
            resp = urg.api_call(
                leaguegamefinder.LeagueGameFinder,
                season_nullable=season,
                season_type_nullable=stype,
                league_id_nullable="00",
                date_from_nullable=gd,
                date_to_nullable=gd,
            )
        except Exception:
            continue
        time.sleep(sleep_s)
        dfx = resp.get_data_frames()[0]
        if dfx is not None and not dfx.empty:
            df_found = dfx
            break

    if df_found is not None and not df_found.empty:
        _lgf_by_date[gd] = df_found.copy()
        return _lgf_by_date[gd]
    _lgf_by_date[gd] = pd.DataFrame()
    return None


def fetch_player_stat_from_nba(
    game_date: str,
    team_abbr: str,
    opponent_abbr: str,
    player_name: str,
    stat: str,
    *,
    sleep_s: float = 0.6,
) -> Optional[float]:
    """
    Return pts / reb / ast / stl / blk / fg3 (3PM) for player in the given matchup on game_date, or None if not found.
    DNP → 0.0 for counting stats.
    """
    stat = (stat or "").lower()
    if stat not in ("pts", "reb", "ast", "stl", "blk", "fg3"):
        return None
    gd = (game_date or "").strip()[:10]
    if len(gd) != 10:
        return None
    df = _league_games_df_for_date(gd, sleep_s=sleep_s)
    if df is None or df.empty:
        return None

    df = df.copy()
    df["GAME_ID_INT"] = df["GAME_ID"].astype(int)
    matched_gid: int | None = None
    home_tid: int | None = None
    away_tid: int | None = None
    matched_gdf: pd.DataFrame | None = None

    c1, c2 = _canon_abbr(team_abbr), _canon_abbr(opponent_abbr)
    for gid, gdf in df.groupby("GAME_ID_INT"):
        abbs = set(gdf["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip())
        if c1 not in abbs or c2 not in abbs:
            continue
        mu = gdf["MATCHUP"].astype(str)
        hm = mu.map(_lgf_matchup_is_home_row)
        home_rows = gdf[hm]
        away_rows = gdf[~hm]
        if home_rows.empty or away_rows.empty:
            continue
        matched_gid = int(gid)
        matched_gdf = gdf
        home_tid = int(home_rows.iloc[0]["TEAM_ID"])
        away_tid = int(away_rows.iloc[0]["TEAM_ID"])
        break

    if matched_gid is None or home_tid is None or away_tid is None:
        return None

    want_tid: int | None = None
    if matched_gdf is not None and c1:
        sub = matched_gdf[
            matched_gdf["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip() == c1
        ]
        if not sub.empty:
            want_tid = int(sub.iloc[0]["TEAM_ID"])

    gid_str = str(matched_gid).zfill(10)
    if gid_str not in _trad_cache:
        try:
            trad = urg.api_call(boxscoretraditionalv3.BoxScoreTraditionalV3, game_id=gid_str)
        except Exception:
            return None
        time.sleep(sleep_s)
        raw = trad.get_dict()
        if not _boxscore_traditional_is_final(raw):
            return None
        _trad_cache[gid_str] = urg._parse_trad_v3_to_df(raw, home_tid, away_tid)

    pdf = _trad_cache[gid_str]
    if pdf is None or pdf.empty:
        return None

    for _, row in pdf.iterrows():
        if want_tid is not None:
            try:
                if int(row.get("teamId") or 0) != want_tid:
                    continue
            except (TypeError, ValueError):
                continue
        fn = f"{row.get('firstName', '')} {row.get('familyName', '')}".strip()
        if not _players_match(fn, player_name):
            continue
        mins = urg.parse_minutes(row.get("minutes"))
        dnp = mins is None
        if stat == "pts":
            v = row.get("points")
        elif stat == "reb":
            v = row.get("reboundsTotal")
        elif stat == "ast":
            v = row.get("assists")
        elif stat == "stl":
            v = row.get("steals")
        elif stat == "fg3":
            v = row.get("threePointersMade")
        else:
            v = row.get("blocks")
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                val = 0.0
            else:
                val = float(v)
        except (TypeError, ValueError):
            val = 0.0
        return 0.0 if dnp else val

    return None


def clear_trad_cache() -> None:
    _trad_cache.clear()
    _lgf_by_date.clear()
