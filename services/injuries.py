"""
services/injuries.py
====================
Late Scratch Guard injury feed helpers.
"""

from __future__ import annotations

import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

CBS_INJURIES_URL = "https://www.cbssports.com/nba/injuries/"
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v|vi)\b", re.IGNORECASE)
_NON_LETTER_RE = re.compile(r"[^a-z\s]")
_SPACE_RE = re.compile(r"\s+")


def normalize_player_name(name: str) -> str:
    """
    Normalize player names for robust cross-source matching.

    Removes accents, punctuation, and common suffixes (Jr, II, ...), then
    lowercases and collapses whitespace.
    """
    if not name:
        return ""
    ascii_name = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    lowered = ascii_name.lower().replace(".", " ")
    no_suffix = _SUFFIX_RE.sub(" ", lowered)
    letters_only = _NON_LETTER_RE.sub(" ", no_suffix)
    return _SPACE_RE.sub(" ", letters_only).strip()


def _fetch_from_cbs() -> set[str]:
    """
    Parse the CBS Sports NBA injuries page and return normalized names
    with status == Out.
    """
    response = requests.get(CBS_INJURIES_URL, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    out_names: set[str] = set()
    for row in soup.select("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.select("td")]
        if len(cells) < 3:
            continue
        player_name = cells[0]
        status_text = cells[2].lower()
        if status_text == "out":
            normalized = normalize_player_name(player_name)
            if normalized:
                out_names.add(normalized)
    return out_names


def _fetch_from_espn() -> set[str]:
    """
    Fallback injury source (public ESPN API) when CBS parsing fails.
    """
    response = requests.get(ESPN_INJURIES_URL, timeout=15)
    response.raise_for_status()
    payload = response.json()
    out_names: set[str] = set()
    for team in payload.get("injuries", []):
        for entry in team.get("injuries", []):
            athlete = entry.get("athlete") or {}
            status = str(entry.get("status") or "").strip().lower()
            if status != "out":
                continue
            normalized = normalize_player_name(str(athlete.get("displayName") or ""))
            if normalized:
                out_names.add(normalized)
    return out_names


def get_injured_players() -> set[str]:
    """
    Return a normalized set of players officially marked Out for today's slate.
    """
    try:
        return _fetch_from_cbs()
    except Exception:
        try:
            return _fetch_from_espn()
        except Exception:
            return set()


_injured_players_cache: tuple[float, set[str]] | None = None


def get_injured_players_cached(ttl_seconds: float = 120.0) -> set[str]:
    """
    Same as get_injured_players() but reuses the last fetch for ttl_seconds.

    Used when simulate_player runs many times in one pipeline so each call
    does not hit the network.
    """
    global _injured_players_cache
    now = time.monotonic()
    if _injured_players_cache is not None:
        ts, data = _injured_players_cache
        if now - ts < ttl_seconds:
            return data
    data = get_injured_players()
    _injured_players_cache = (now, data)
    return data
