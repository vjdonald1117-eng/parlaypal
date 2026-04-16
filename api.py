"""
api.py
======
FastAPI backend for ParlayPal.

Endpoints
---------
  POST /api/refresh        Trigger live scraper + Monte Carlo simulations.
                           Returns full slate of PlayerSim results and logs
                           them to the prediction_log table for history tracking.

  POST /api/sync-game-lines
                           Refresh NBA schedule + Vegas spread/total into the games
                           table (no simulation). ODDS_API_KEY required for lines.

  GET  /api/projections    Return the latest cached simulation results.
                           Query params:
                             stat  — "pts" | "reb" | "ast" | "stl" | "blk" | "fg3"  (default: "pts")
                             team  — 3-letter abbreviation filter, e.g. "LAL"

  GET  /api/history        Return past predictions from prediction_log (read-only;
                           grading runs on POST /api/update-history only).
                           Query params:
                             stat  — filter by stat
                             limit — max rows (default: 100)

Run
---
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

def _find_project_root(start_dir: Path) -> Path:
    """
    Resolve the repository root by walking upward until we find known
    project folders. Falls back to start_dir when markers are absent.
    """
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "scripts").is_dir() and (candidate / "parlay-ui").is_dir():
            return candidate
    return start_dir


_PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
_api_dir = str(_PROJECT_ROOT)
_scripts_dir = _PROJECT_ROOT / "scripts"
load_dotenv(str(_PROJECT_ROOT / ".env"))

from core.logger import configure_app_logging

configure_app_logging()
logger = logging.getLogger(__name__)

from database import PredictionLog, engine
from models.player_projections import current_season, MIN_GP_DEFAULT
from services.injuries import get_injured_players, normalize_player_name
from services.simulations import N_SIMS, simulate_player
from models.parlay_builder import (
    PlayerSim,
    simulate_game_players_joint,
    fetch_scoreboard,
    build_abbr_to_db_id,
    get_top_scorers,
    trigger_db_update,
    _load_xgb_models,
)
from scripts.fetch_live_odds import fetch_live_odds as _fetch_live_odds
from scripts.fetch_live_odds import fetch_h2h_moneyline_board

# NBA slate dates for ESPN scoreboard requests — use Eastern Time, not server local time.
_NY_TZ = ZoneInfo("America/New_York")

# prediction_log ORM + create_all live in database.py

# ---------------------------------------------------------------------------
# In-memory cache — stores the last /api/refresh output per (game_date, stat)
# ---------------------------------------------------------------------------
_cache_store: dict[str, dict[str, Any]] = {}
_refresh_running: bool = False
JOBS: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()


def _cache_key(game_date: str, stat: str) -> str:
    return f"{game_date}:{stat}"


def _sort_key_edge_high_first(r: dict[str, Any]) -> tuple[float, bool]:
    """
    Highest win_probability first; ensemble_lock only breaks ties.
    (Sorting with lock first wrongly buried a 70% REB edge below a 50% locked PTS pick.)
    """
    wp = float(r.get("win_probability") or 0)
    return (wp, bool(r.get("ensemble_lock")))


def trigger_injury_update(verbose: bool = True) -> bool:
    """
    Run scrape_injuries.py to refresh injury_reports in Supabase.
    Returns True if the process exited cleanly.
    """
    script = os.path.join(_api_dir, "scrape_injuries.py")
    if not os.path.exists(script):
        if verbose:
            logger.warning("  [injuries] scrape_injuries.py not found at %s", script)
        return False
    if verbose:
        logger.info("  [injuries] Updating injury report (ESPN)...")
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
        cwd=_api_dir,
    )
    return result.returncode == 0


_last_injury_scrape_utc: datetime | None = None
INJURY_SCRAPE_COOLDOWN = timedelta(minutes=12)

# Injury rows older than this are ignored when deciding exclusions.
# Current simulation policy:
#   - Questionable = OUT (exclude)
#   - Probable     = IN  (include)
_INJURY_LOOKBACK_DAYS = 21

_UNAVAIL_INJURY_SQL = f"""
                    WITH latest AS (
                        SELECT DISTINCT ON (ir.player_id)
                               ir.player_id,
                               ir.status,
                               ir.report_date
                        FROM injury_reports ir
                        WHERE ir.report_date >= CURRENT_DATE - INTERVAL '{_INJURY_LOOKBACK_DAYS} days'
                        ORDER BY ir.player_id, ir.report_date DESC, ir.report_sequence DESC
                    )
                    SELECT LOWER(TRIM(BOTH FROM p.full_name)) AS name
                    FROM latest l
                    JOIN players p ON p.id = l.player_id
                    WHERE l.status IN ('Out', 'Doubtful', 'Questionable')
                    """


def trigger_injury_update_throttled(verbose: bool = True) -> bool:
    """Run injury scrape at most once per INJURY_SCRAPE_COOLDOWN (wall clock)."""
    global _last_injury_scrape_utc
    now = datetime.now(timezone.utc)
    if _last_injury_scrape_utc is not None:
        if (now - _last_injury_scrape_utc) < INJURY_SCRAPE_COOLDOWN:
            return True
    ok = trigger_injury_update(verbose=verbose)
    if ok:
        _last_injury_scrape_utc = now
    return ok


def trigger_injury_update_before_sim(verbose: bool = False) -> None:
    """Always pull fresh ESPN injuries before building the simulate list (each Run Simulations)."""
    trigger_injury_update(verbose=verbose)
    global _last_injury_scrape_utc
    _last_injury_scrape_utc = datetime.now(timezone.utc)


def trigger_schedule_odds_sync(verbose: bool = True) -> bool:
    """
    Run scripts/sync_nba_schedule.py: NBA scoreboard (today + next 2 days) and Odds API
    markets (spreads, totals) merged into games.closing_spread / games.closing_total,
    plus days_rest. Projections read those columns for pace and blowout context; without
    this sync, lines can stay stale indefinitely.
    """
    script = _scripts_dir / "sync_nba_schedule.py"
    if not script.exists():
        if verbose:
            logger.warning("  [schedule] sync_nba_schedule.py not found at %s", script)
        return False
    if verbose:
        logger.info("  [schedule] Syncing NBA schedule + Vegas spread/total into DB ...")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=_api_dir,
        text=True,
    )
    ok = result.returncode == 0
    if verbose:
        if ok:
            logger.info("  [schedule] sync_nba_schedule.py -> OK")
        else:
            logger.warning(
                "  [schedule] sync_nba_schedule.py -> FAILED (rc=%s)",
                result.returncode,
            )
    return ok


def _load_unavailable_names_set(session: Session) -> set[str]:
    return set(
        normalize_player_name((r.name or "").strip())
        for r in session.execute(text(_UNAVAIL_INJURY_SQL)).fetchall()
        if r.name
    )


def apply_late_scratch_guard(
    tuples_list: list[tuple[Any, ...]],
    injured_out_names: set[str] | None = None,
) -> list[tuple[Any, ...]]:
    """
    Second ESPN injury scrape right before sim threads start: removes players ruled Out/Doubtful
    after the initial list was built (late scratches).
    """
    trigger_injury_update(verbose=False)
    global _last_injury_scrape_utc
    _last_injury_scrape_utc = datetime.now(timezone.utc)
    with Session(engine) as session:
        unavail = _load_unavailable_names_set(session)
    if injured_out_names:
        unavail |= set(injured_out_names)
    out: list[tuple[Any, ...]] = []
    for t in tuples_list:
        player_name = str(t[0] or "").strip()
        if normalize_player_name(player_name) in unavail:
            logger.info("[Late Scratch Guard] Skipping %s - Ruled Out", player_name)
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Lifespan — load XGBoost model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_xgb_models()
    yield


app = FastAPI(title="ParlayPal API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    # Any localhost / loopback port (alternate Vite ports, Simple Browser, etc.)
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim_to_dict(s: PlayerSim) -> dict:
    """Convert a PlayerSim dataclass to a JSON-serialisable dict."""
    d = asdict(s)
    # Rename fields to match the API contract expected by the frontend
    d["heuristic_mean"] = d.pop("final_mean")
    d["ml_mean"]        = d.pop("xgb_mean") if d.get("xgb_available") else None
    d["win_probability"] = d["best_pct"]   # convenience alias
    return d


def _normalize_explanation_tags_for_db(val: Any) -> list[str]:
    """Clamp tags before INSERT (JSONB array of strings)."""
    if val is None:
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return []
    if not isinstance(val, (list, tuple)):
        return []
    out: list[str] = []
    for x in val[:5]:
        s = str(x).strip()[:120]
        if s:
            out.append(s)
    return out[:3]


def _coerce_explanation_tags_from_db(val: Any) -> list[str]:
    """Read JSONB / legacy NULL into list[str] for API rows (matches live cache)."""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()][:3]
    if isinstance(val, dict):
        return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()][:3]
        except Exception:
            return []
    return []


def _clip_prediction_field(val: Any, max_len: int) -> Any:
    if val is None or max_len <= 0:
        return val
    if isinstance(val, (datetime, date)):
        s = val.isoformat()
    else:
        s = str(val)
    s = s.strip()
    return s[:max_len]


def _sanitize_prediction_log_tuple(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Clip strings to match PredictionLog column limits (PostgreSQL VARCHAR errors otherwise)."""
    tags_raw: Any = row[16] if len(row) > 16 else []
    (
        game_date,
        player_name,
        team_abbr,
        opponent,
        stat,
        line,
        line_src,
        hmean,
        mlmean,
        op,
        up,
        bside,
        bpct,
        ev,
        verdict,
        elock,
    ) = row[:16]
    tags = _normalize_explanation_tags_for_db(tags_raw)
    return (
        _clip_prediction_field(game_date, 10),
        _clip_prediction_field(player_name, 100),
        _clip_prediction_field(team_abbr, 5),
        _clip_prediction_field(opponent, 5),
        _clip_prediction_field(stat, 5),
        line,
        _clip_prediction_field(line_src, 5),
        hmean,
        mlmean,
        op,
        up,
        _clip_prediction_field(bside, 6),
        bpct,
        ev,
        _clip_prediction_field(verdict, 30),
        bool(elock),
        tags,
    )


def _batch_insert_prediction_logs(rows: list[tuple[Any, ...]]) -> None:
    """
    Persist rows with one ORM flush (no raw psycopg2). Sanitizes VARCHAR limits
    so PostgreSQL never rejects the batch.
    """
    if not rows:
        return
    clean = [_sanitize_prediction_log_tuple(r) for r in rows]
    logs = [
        PredictionLog(
            game_date=r[0],
            player_name=r[1],
            team_abbr=r[2],
            opponent=r[3],
            stat=r[4],
            line=r[5],
            line_source=r[6],
            heuristic_mean=r[7],
            ml_mean=r[8],
            over_pct=r[9],
            under_pct=r[10],
            best_side=r[11],
            best_pct=r[12],
            ev_per_110=r[13],
            verdict=r[14],
            ensemble_lock=r[15],
            explanation_tags=r[16],
        )
        for r in clean
    ]
    with Session(engine) as session:
        session.add_all(logs)
        session.commit()
    gd = str(clean[0][0]).strip()[:10] if clean else ""
    if len(gd) == 10:
        try:
            _snapshot_prediction_log_to_disk(gd)
        except Exception:
            pass


def _snapshot_prediction_log_to_disk(game_date: str) -> None:
    """Write a timestamped JSON snapshot of prediction_log for this slate (prediction_backups/)."""
    backup_root = os.path.join(_api_dir, "prediction_backups")
    os.makedirs(backup_root, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(backup_root, f"{game_date}_{stamp}.json")
    cols = (
        "id, logged_at, game_date, player_name, team_abbr, opponent, stat, line, "
        "line_source, best_side, over_pct, under_pct, verdict, hit, actual_value, "
        "ensemble_lock, heuristic_mean, ml_mean, ev_per_110, best_pct, explanation_tags"
    )
    with Session(engine) as session:
        rows = session.execute(
            text(f"SELECT {cols} FROM prediction_log WHERE TRIM(game_date) = TRIM(:gd) ORDER BY id ASC"),
            {"gd": game_date},
        ).fetchall()

    def _json_val(v: Any) -> Any:
        if v is None:
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if isinstance(v, (bool, int, float, str)):
            return v
        return str(v)

    out_rows: list[dict[str, Any]] = []
    for r in rows:
        m = r._mapping
        out_rows.append({k: _json_val(m[k]) for k in m.keys()})

    payload: dict[str, Any] = {
        "game_date": game_date,
        "exported_at_utc": stamp,
        "n_rows": len(out_rows),
        "rows": out_rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _fetch_live_odds_parallel(
    stats: tuple[str, ...],
    include_today: bool,
    *,
    use_odds_cache: bool = True,
) -> dict[str, dict]:
    if not stats:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, len(stats))) as pool:
        futs = {
            pool.submit(_fetch_live_odds, st, use_odds_cache, include_today): st
            for st in stats
        }
    out: dict[str, dict] = {}
    for fut in as_completed(futs):
        st = futs[fut]
        out[st] = fut.result()
    return out


def _log_game_win_predictions(summaries: list[dict[str, Any]], game_date: str) -> None:
    """
    Log model moneyline picks (home vs away) so POST /api/update-history can grade them
    once games.home_score / away_score exist (same table as player box scores).
    """
    rows: list[tuple[Any, ...]] = []
    for gs in summaries:
        home = str(gs.get("home_db_abbr") or gs.get("home_abbr") or "")[:5].strip()
        away = str(gs.get("away_db_abbr") or gs.get("away_abbr") or "")[:5].strip()
        if not home or not away:
            continue
        ph = float(gs["p_home_win"])
        pa = float(gs["p_away_win"])
        best_side = "OVER" if ph >= pa else "UNDER"
        best_pct = max(ph, pa)
        matchup = str(gs.get("matchup") or f"{away} @ {home}")[:40]
        verdict = f"Sim ML H{ph:.0f}% A{pa:.0f}%"
        if len(verdict) > 30:
            verdict = verdict[:30]
        rows.append(
            (
                game_date,
                f"{matchup} (sim ML)"[:100],
                home,
                away,
                "win",
                0.5,
                "SIM",
                round(ph / 100.0, 4),
                None,
                ph,
                pa,
                best_side,
                round(best_pct, 2),
                0.0,
                verdict,
                False,
                [],
            )
        )
    _batch_insert_prediction_logs(rows)


def _resolve_win_predictions_from_games(session: Session, limit: int) -> int:
    """Grade stat='win' rows when games has final scores and abbrs match log."""
    rows = session.execute(
        text(
            """
            SELECT pl.id, pl.best_side, g.home_score, g.away_score
            FROM prediction_log pl
            JOIN games g
              ON g.game_date = CAST(NULLIF(TRIM(pl.game_date), '') AS DATE)
            JOIN teams th ON th.id = g.home_team_id
            JOIN teams ta ON ta.id = g.away_team_id
            WHERE pl.stat = 'win'
              AND pl.hit IS NULL
              AND pl.actual_value IS NULL
              AND pl.game_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
              AND UPPER(TRIM(th.abbreviation)) = UPPER(TRIM(pl.team_abbr))
              AND UPPER(TRIM(ta.abbreviation)) = UPPER(TRIM(pl.opponent))
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
            ORDER BY pl.logged_at ASC
            LIMIT :lim
            """
        ),
        {"lim": max(1, min(limit, 2000))},
    ).fetchall()
    n = 0
    for r in rows:
        hs = int(r.home_score)
        aws = int(r.away_score)
        side = str(r.best_side or "").upper()
        if hs > aws:
            actual_val = 1.0
            hit = side == "OVER"
        elif hs < aws:
            actual_val = 0.0
            hit = side == "UNDER"
        else:
            actual_val = 0.5
            hit = None
        session.execute(
            text(
                """
                UPDATE prediction_log
                SET actual_value = :av, hit = :hit
                WHERE id = :id
                """
            ),
            {"av": actual_val, "hit": hit, "id": int(r.id)},
        )
        n += 1
    return n


# Player prop stats (order = mixed top-10 round-robin in /api/projections-combined and UI "All").
_PROJECTION_STATS: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk", "fg3")
_PROP_STAT_KEYS = frozenset(_PROJECTION_STATS)
_HISTORY_STATS: tuple[str, ...] = _PROJECTION_STATS + ("win",)


def _log_db_row_to_projection_dict(r: Any) -> dict[str, Any]:
    """Shape matches cached /api/projections rows (PlayerSim JSON) for the UI."""
    ml = r.ml_mean
    xgb_ok = ml is not None
    h = float(r.heuristic_mean)
    return {
        "player_name": r.player_name,
        "team_abbr": r.team_abbr,
        "opponent": r.opponent,
        "stat": str(r.stat or "").lower(),
        "line": float(r.line),
        "heuristic_mean": h,
        "std_dev": 4.0,
        "over_pct": float(r.over_pct),
        "under_pct": float(r.under_pct),
        "best_side": r.best_side,
        "best_pct": float(r.best_pct),
        "win_probability": float(r.best_pct),
        "ev_per_110": float(r.ev_per_110),
        "verdict": r.verdict,
        "line_source": r.line_source,
        "bump_pts": 0.0,
        "def_mult": 1.0,
        "pace_mult": 1.0,
        "conf_boost": 0.0,
        "is_rust": False,
        "gp": 10,
        "over_odds": 0,
        "under_odds": 0,
        "ml_mean": float(ml) if xgb_ok else None,
        "xgb_mean": float(ml) if xgb_ok else 0.0,
        "xgb_over_pct": float(r.over_pct),
        "xgb_under_pct": float(r.under_pct),
        "xgb_best_side": str(r.best_side or ""),
        "xgb_best_pct": float(r.best_pct),
        "xgb_ev_per_110": float(r.ev_per_110),
        "xgb_available": xgb_ok,
        "ensemble_lock": bool(r.ensemble_lock),
        "sim_note": "",
        "explanation_tags": _coerce_explanation_tags_from_db(
            getattr(r, "explanation_tags", None)
        ),
    }


def _results_from_prediction_log(game_date: str, stat: str) -> tuple[list[dict[str, Any]], Optional[str]]:
    """
    Rebuild projection rows from prediction_log when the in-memory cache is empty
    (e.g. uvicorn restarted). Lets Date=Today work after an earlier Run Simulations.
    """
    st = (stat or "").strip().lower()
    if st not in _PROP_STAT_KEYS:
        return [], None
    q = text(
        """
        SELECT DISTINCT ON (player_name, LOWER(TRIM(stat)))
            player_name, team_abbr, opponent, stat, line, line_source,
            heuristic_mean, ml_mean, over_pct, under_pct, best_side, best_pct,
            ev_per_110, verdict, ensemble_lock, logged_at, explanation_tags
        FROM prediction_log
        WHERE TRIM(game_date) = TRIM(:gd)
          AND LOWER(TRIM(stat)) = :st
        ORDER BY player_name, LOWER(TRIM(stat)), logged_at DESC NULLS LAST
        """
    )
    try:
        with Session(engine) as session:
            rows = session.execute(q, {"gd": game_date, "st": st}).fetchall()
    except Exception:
        return [], None
    if not rows:
        return [], None
    out = [_log_db_row_to_projection_dict(r) for r in rows]
    latest: datetime | None = None
    for r in rows:
        la = getattr(r, "logged_at", None)
        if la is not None and (latest is None or la > latest):
            latest = la
    ts = latest.isoformat() if latest else None
    return out, ts


def _normalize_prop_stat_param(stat: str) -> str:
    """Accept FG3 / 3pm / etc.; avoid Query regex 422 from case or alias mismatch."""
    s = (stat or "").strip().lower()
    if s in ("3pm", "3pt", "three", "threes", "fg3m"):
        s = "fg3"
    if s not in _PROP_STAT_KEYS:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid stat. Use pts, reb, ast, stl, blk, or fg3 "
                "(threes aliases: 3pm, 3pt, threes, fg3m)."
            ),
        )
    return s


def _normalize_optional_history_stat(stat: Optional[str]) -> Optional[str]:
    if stat is None:
        return None
    s = (stat or "").strip().lower()
    if s in ("3pm", "3pt", "three", "threes", "fg3m"):
        s = "fg3"
    allowed = frozenset(_HISTORY_STATS)
    if s not in allowed:
        raise HTTPException(
            status_code=400,
            detail="stat must be pts, reb, ast, stl, blk, fg3, win, or null for all",
        )
    return s


def _normalize_day_query(day: str) -> str:
    """Case-insensitive today|tomorrow; avoids strict regex 422 from proxies or clients."""
    d = (day or "").strip().lower()
    if d not in ("today", "tomorrow"):
        raise HTTPException(
            status_code=422,
            detail="Invalid 'day' query parameter. Use today or tomorrow.",
        )
    return d


def _mixed_top_props(side_rows: list[dict[str, Any]], *, final_n: int, per_stat: int) -> list[dict[str, Any]]:
    """
    Build top-N OVER or UNDER list with at least one slot per cached stat when possible,
    so the board is not dominated by a single prop type (e.g. all PTS).
    """
    by_stat: dict[str, list[dict[str, Any]]] = {s: [] for s in _PROJECTION_STATS}
    for r in side_rows:
        st = str(r.get("stat") or "").lower()
        if st in by_stat:
            by_stat[st].append(r)
    for lst in by_stat.values():
        lst.sort(key=_sort_key_edge_high_first, reverse=True)

    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("player_name") or ""),
            str(row.get("stat") or ""),
            str(row.get("best_side") or ""),
            str(row.get("line") or ""),
        )

    for st in _PROJECTION_STATS:
        for r in by_stat[st][:per_stat]:
            k = _key(r)
            if k not in seen:
                seen.add(k)
                picked.append(r)
            if len(picked) >= final_n:
                break
        if len(picked) >= final_n:
            break

    if len(picked) < final_n:
        rest: list[dict[str, Any]] = []
        for st in _PROJECTION_STATS:
            for r in by_stat[st][per_stat:]:
                k = _key(r)
                if k not in seen:
                    rest.append(r)
        rest.sort(key=_sort_key_edge_high_first, reverse=True)
        for r in rest:
            if len(picked) >= final_n:
                break
            k = _key(r)
            if k not in seen:
                seen.add(k)
                picked.append(r)

    picked.sort(key=_sort_key_edge_high_first, reverse=True)
    return picked[:final_n]


def _log_predictions(sims: list[PlayerSim], game_date: str) -> None:
    """Persist simulation results to prediction_log (batch insert, one row per PlayerSim.stat)."""
    rows = [
        (
            game_date,
            s.player_name,
            s.team_abbr,
            s.opponent,
            s.stat.lower(),
            s.line,
            s.line_source,
            s.final_mean,
            s.xgb_mean if s.xgb_available else None,
            s.over_pct,
            s.under_pct,
            s.best_side,
            s.best_pct,
            s.ev_per_110,
            s.verdict,
            s.ensemble_lock,
            list(s.explanation_tags or []),
        )
        for s in sims
    ]
    _batch_insert_prediction_logs(rows)


def _run_full_pipeline(
    stat: str,
    target_date,
    top_scorers_n: int = 4,
    n_sims: int | None = None,
    *,
    use_odds_cache: bool = True,
) -> dict:
    """
    Execute the live data pipeline end-to-end and return a results dict.
    Mirrors the logic in parlay_builder.main() but returns data instead of
    printing to stdout.
    """
    season = current_season()
    injured_out_names = get_injured_players()

    trigger_schedule_odds_sync(verbose=False)
    # Align "today" / "tomorrow" with NBA calendar in Eastern Time (ESPN slates).
    ny_today = datetime.now(_NY_TZ).date()
    ny_tomorrow = ny_today + timedelta(days=1)

    game_date = target_date.strftime("%Y-%m-%d")
    target_str = target_date.strftime("%Y%m%d")

    # 1. Fetch ESPN slate for target date
    target_games = fetch_scoreboard(target_str)

    if not target_games:
        return {
            "game_date": game_date,
            "stat": stat,
            "results": [],
            "otb": [],
            "skipped": [],
            "message": f"No games found for {game_date}.",
        }

    # 2. Eligible games
    # For tomorrow slates, apply "OTB" filtering (back-to-back teams still playing today).
    # For today's slate, simulate all games without OTB filtering.
    otb_labels: list[str] = []

    if target_date == ny_tomorrow:
        today_games = fetch_scoreboard(ny_today.strftime("%Y%m%d"))
        tomorrow_games = target_games

        today_by_espn_id: dict[int, Any] = {}
        for g in today_games:
            today_by_espn_id[g.home_espn_id] = g
            today_by_espn_id[g.away_espn_id] = g

        eligible_games: list[Any] = []
        triggered = False

        for g in tomorrow_games:
            blocking = [
                today_by_espn_id[eid]
                for eid in (g.home_espn_id, g.away_espn_id)
                if eid in today_by_espn_id
            ]
            if not blocking:
                eligible_games.append(g)
                continue

            any_live = any(tg.status_state in ("pre", "in") for tg in blocking)
            if any_live:
                labels = [f"{tg.label} [{tg.status_name}]" for tg in blocking]
                otb_labels.append(f"{g.label}  --  blocking: {', '.join(labels)}")
            else:
                if not triggered:
                    trigger_db_update(verbose=False)
                    triggered = True
                eligible_games.append(g)
    else:
        eligible_games = target_games

    if not eligible_games:
        return {
            "game_date": game_date,
            "stat": stat,
            "results": [],
            "otb": otb_labels,
            "skipped": [],
            "message": "No eligible games (all OTB).",
        }

    # 2.5 Fresh ESPN injuries before each run so Out/Doubtful exclusions match tipoff reality.
    trigger_injury_update_before_sim(verbose=False)

    # 3. Collect players to simulate
    sims:    list[PlayerSim] = []
    skipped: list[str]       = []

    with Session(engine) as session:
        abbr_to_db = build_abbr_to_db_id(session)

        unavailable_names = _load_unavailable_names_set(session)
        unavailable_names |= injured_out_names

        to_simulate = []
        for g in eligible_games:
            for team_abbr, opp_abbr, is_home in [
                (g.home_db_abbr, g.away_db_abbr, True),
                (g.away_db_abbr, g.home_db_abbr, False),
            ]:
                db_id     = abbr_to_db.get(team_abbr.upper())
                opp_db_id = abbr_to_db.get(opp_abbr.upper())
                if db_id is None:
                    logger.warning(
                        "[WARN] db_id not found for team_abbr=%s (game=%s)",
                        team_abbr,
                        g.label,
                    )
                    continue
                top_players = get_top_scorers(session, db_id, season, stat, n=top_scorers_n)
                logger.debug(
                    "[DEBUG] top_scorers returned %s players for team_abbr=%s, db_id=%s, stat=%s",
                    len(top_players),
                    team_abbr,
                    db_id,
                    stat,
                )
                for _, name, _ in top_players:
                    if normalize_player_name(name) in unavailable_names:
                        logger.info("[Late Scratch Guard] Skipping %s - Ruled Out", name)
                        continue
                    to_simulate.append((name, team_abbr, opp_abbr, is_home, opp_db_id))

        logger.debug(
            "[DEBUG] eligible_games=%s to_simulate_count=%s",
            len(eligible_games),
            len(to_simulate),
        )

    to_simulate = apply_late_scratch_guard(to_simulate, injured_out_names=injured_out_names)
    logger.debug(
        "[DEBUG] after late-scratch guard: to_simulate_count=%s",
        len(to_simulate),
    )

    # 4. Live odds (file cache in scripts/odds_cache_<stat>.json — bypass with fresh_odds on refresh)
    include_today = target_date == ny_today
    live_odds = _fetch_live_odds(stat, use_cache=use_odds_cache, include_today=include_today)

    # 5. Simulate — each thread opens its own independent DB session
    logger.info("Starting simulations for %s players...", len(to_simulate))

    def _simulate_one(args):
        player_name, team_abbr, opp_abbr, is_home, opp_db_id = args
        with Session(engine) as thread_session:
            result = simulate_player(
                thread_session, player_name, opp_abbr, stat, season,
                live_odds=live_odds,
                is_home=is_home,
                opp_db_team_id=opp_db_id,
                tomorrow_date_obj=target_date,
                n_sims=n_sims,
            )
        return player_name, team_abbr, result

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_simulate_one, args): args for args in to_simulate}
        for future in as_completed(futures):
            player_name, team_abbr, result = future.result()
            if result is None:
                logger.warning(
                    "SKIPPED: %s (%s) - simulate_player returned None "
                    "(likely missing odds or name mismatch).",
                    player_name,
                    team_abbr,
                )
                skipped.append(f"{player_name} ({team_abbr})")
            else:
                logger.info(
                    "SUCCESS: %s (%s) - Line: %s",
                    player_name,
                    team_abbr,
                    result.line,
                )
                sims.append(result)

    # 6. Persist to prediction_log
    if sims:
        _log_predictions(sims, game_date)

    return {
        "game_date": game_date,
        "stat": stat,
        "results": [_sim_to_dict(s) for s in sims],
        "otb": otb_labels,
        "skipped": skipped,
    }


_UNIFIED_N_SIMS_DEFAULT = 5_000


def _run_unified_pipeline(
    target_date,
    n_sims: int | None = None,
    *,
    use_odds_cache: bool = True,
) -> dict[str, Any]:
    """
    One joint Monte Carlo run per game: all five prop types share correlated noise;
    caches every stat slice plus simulated team win % from summed top-scorer PTS.
    """
    effective_n = int(n_sims) if n_sims is not None else _UNIFIED_N_SIMS_DEFAULT
    effective_n = max(1000, min(effective_n, 50_000))

    injured_out_names = get_injured_players()

    trigger_schedule_odds_sync(verbose=False)

    season = current_season()
    ny_today = datetime.now(_NY_TZ).date()
    ny_tomorrow = ny_today + timedelta(days=1)
    game_date = target_date.strftime("%Y-%m-%d")
    target_str = target_date.strftime("%Y%m%d")

    target_games = fetch_scoreboard(target_str)
    if not target_games:
        empty_by_stat = {s: [] for s in _PROJECTION_STATS}
        return {
            "game_date": game_date,
            "stat": "unified",
            "results": [],
            "otb": [],
            "skipped": [],
            "games": [],
            "game_rows": [],
            "by_stat": empty_by_stat,
            "n_sims": effective_n,
            "message": f"No games found for {game_date}.",
        }

    otb_labels: list[str] = []
    if target_date == ny_tomorrow:
        today_games = fetch_scoreboard(ny_today.strftime("%Y%m%d"))
        tomorrow_games = target_games
        today_by_espn_id: dict[int, Any] = {}
        for g in today_games:
            today_by_espn_id[g.home_espn_id] = g
            today_by_espn_id[g.away_espn_id] = g
        eligible_games: list[Any] = []
        triggered = False
        for g in tomorrow_games:
            blocking = [
                today_by_espn_id[eid]
                for eid in (g.home_espn_id, g.away_espn_id)
                if eid in today_by_espn_id
            ]
            if not blocking:
                eligible_games.append(g)
                continue
            any_live = any(tg.status_state in ("pre", "in") for tg in blocking)
            if any_live:
                labels = [f"{tg.label} [{tg.status_name}]" for tg in blocking]
                otb_labels.append(f"{g.label}  --  blocking: {', '.join(labels)}")
            else:
                if not triggered:
                    trigger_db_update(verbose=False)
                    triggered = True
                eligible_games.append(g)
    else:
        eligible_games = target_games

    if not eligible_games:
        empty_by_stat = {s: [] for s in _PROJECTION_STATS}
        return {
            "game_date": game_date,
            "stat": "unified",
            "results": [],
            "otb": otb_labels,
            "skipped": [],
            "games": [],
            "game_rows": [],
            "by_stat": empty_by_stat,
            "n_sims": effective_n,
            "message": "No eligible games (all OTB).",
        }

    trigger_injury_update_before_sim(verbose=False)

    to_simulate: list[tuple[str, str, str, bool, int | None, str]] = []
    skipped: list[str] = []

    with Session(engine) as session:
        abbr_to_db = build_abbr_to_db_id(session)
        unavailable_names = _load_unavailable_names_set(session)
        unavailable_names |= injured_out_names

        for g in eligible_games:
            for team_abbr, opp_abbr, is_home in [
                (g.home_db_abbr, g.away_db_abbr, True),
                (g.away_db_abbr, g.home_db_abbr, False),
            ]:
                db_id = abbr_to_db.get(team_abbr.upper())
                opp_db_id = abbr_to_db.get(opp_abbr.upper())
                if db_id is None:
                    continue
                top_players = get_top_scorers(session, db_id, season, "pts", n=4)
                for _, name, _ in top_players:
                    if normalize_player_name(name) in unavailable_names:
                        logger.info("[Late Scratch Guard] Skipping %s - Ruled Out", name)
                        continue
                    to_simulate.append((name, team_abbr, opp_abbr, is_home, opp_db_id, g.event_id))

    to_simulate = apply_late_scratch_guard(to_simulate, injured_out_names=injured_out_names)

    include_today = target_date == ny_today
    live_odds_by_stat = _fetch_live_odds_parallel(
        _PROJECTION_STATS, include_today, use_odds_cache=use_odds_cache
    )

    by_game: dict[str, list[tuple[str, str, str, bool, int | None]]] = defaultdict(list)
    for name, team_abbr, opp_abbr, is_home, opp_db_id, eid in to_simulate:
        by_game[eid].append((name, team_abbr, opp_abbr, is_home, opp_db_id))

    games_by_eid = {g.event_id: g for g in eligible_games}
    all_player_sims: list[PlayerSim] = []
    game_summaries: list[dict[str, Any]] = []

    def _run_one_game(eid: str):
        g = games_by_eid.get(eid)
        if not g:
            return [], None
        plist = by_game.get(eid) or []
        with Session(engine) as session:
            return simulate_game_players_joint(
                session,
                g,
                plist,
                season,
                live_odds_by_stat,
                effective_n,
            )

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(_run_one_game, eid): eid for eid in by_game}
        for fut in as_completed(futures):
            psims, gsum = fut.result()
            all_player_sims.extend(psims)
            if gsum:
                game_summaries.append(gsum)

    game_summaries.sort(key=lambda x: str(x.get("matchup") or ""))

    if all_player_sims:
        _log_predictions(all_player_sims, game_date)

    if game_summaries:
        _log_game_win_predictions(game_summaries, game_date)

    by_stat_lists: dict[str, list[dict[str, Any]]] = {s: [] for s in _PROJECTION_STATS}
    for s in all_player_sims:
        d = _sim_to_dict(s)
        key = s.stat.lower()
        if key in by_stat_lists:
            by_stat_lists[key].append(d)

    for rows in by_stat_lists.values():
        rows.sort(key=lambda r: _sort_key_edge_high_first(r), reverse=True)

    game_rows: list[dict[str, Any]] = []
    for gs in game_summaries:
        ph = float(gs["p_home_win"])
        pa = float(gs["p_away_win"])
        phc = float(gs.get("p_home_cover", 50.0))
        pac = float(gs.get("p_away_cover", 50.0))
        home = str(gs.get("home_db_abbr") or gs["home_abbr"])
        away = str(gs.get("away_db_abbr") or gs["away_abbr"])
        disp_home = str(gs["home_abbr"])
        disp_away = str(gs["away_abbr"])
        spread = gs.get("closing_spread")
        spread_num = float(spread) if spread is not None else 0.0
        proj_total = float(gs.get("projected_total") or 0.0)
        home_proj = float(gs.get("home_proj_pts") or 0.0)
        away_proj = float(gs.get("away_proj_pts") or 0.0)
        home_cover_by = float(gs.get("home_cover_by") or 0.0)
        away_cover_by = float(gs.get("away_cover_by") or 0.0)
        game_rows.append(
            {
                "player_name": f"{disp_home} vs {disp_away}",
                "team_abbr": home,
                "opponent": away,
                "matchup": gs["matchup"],
                "stat": "gproj",
                "line": spread_num,
                "heuristic_mean": proj_total,
                "ml_mean": None,
                "win_probability": max(phc, pac),
                "over_pct": phc,
                "under_pct": pac,
                "best_side": "OVER" if phc >= pac else "UNDER",
                "best_pct": max(phc, pac),
                "ev_per_110": 0.0,
                "verdict": (
                    f"Proj {disp_home} {home_proj:.1f} - {disp_away} {away_proj:.1f} "
                    f"(Total {proj_total:.1f}); Cover: {disp_home} {home_cover_by:+.1f}, "
                    f"{disp_away} {away_cover_by:+.1f}; Win P({disp_home})={ph:.1f}% "
                    f"P({disp_away})={pa:.1f}%"
                ),
                "ensemble_lock": False,
            }
        )

    return {
        "game_date": game_date,
        "stat": "unified",
        "results": [_sim_to_dict(s) for s in all_player_sims],
        "otb": otb_labels,
        "skipped": skipped,
        "games": game_summaries,
        "game_rows": game_rows,
        "by_stat": by_stat_lists,
        "n_sims": effective_n,
    }


def _write_unified_caches(data: dict[str, Any], day: str) -> str:
    """Populate per-stat caches + simgames from a unified pipeline result."""
    ts = datetime.now(timezone.utc).isoformat()
    gd = data["game_date"]
    for stat in _PROJECTION_STATS:
        rows = data["by_stat"].get(stat, [])
        _cache_store[_cache_key(gd, stat)] = {
            "timestamp": ts,
            "game_date": gd,
            "stat": stat,
            "results": rows,
            "otb": data["otb"],
            "skipped": data["skipped"],
            "day": day,
        }
    _cache_store[_cache_key(gd, "simgames")] = {
        "timestamp": ts,
        "game_date": gd,
        "stat": "simgames",
        "results": data.get("game_rows") or [],
        "otb": data["otb"],
        "skipped": data["skipped"],
        "day": day,
    }
    return ts


def _fetch_merged_history_rows(
    session: Session,
    *,
    stat: str | None,
    ny: str,
    limit: int,
) -> list[Any]:
    """
    Merge two slices of prediction_log so Eastern *past* slates stay in the payload:

    1) Rows with game_date strictly before today's NY calendar date (up to past_lim).
    2) Rows with game_date on or after that date (up to recent_lim).

    A single ORDER BY game_date DESC LIMIT N can still omit past dates entirely when
    one future slate has more than N rows. Grading is not run here — use POST
    /api/update-history so GET stays fast and reliable.
    """
    ny_key = (ny or "")[:10]
    cap = max(1, min(int(limit), 5000))
    past_lim = min(cap, 4000)
    recent_lim = min(cap, 4000)

    cols = (
        "id, logged_at, game_date, player_name, team_abbr, opponent, stat, "
        "line, line_source, heuristic_mean, ml_mean, "
        "over_pct, under_pct, best_side, best_pct, ev_per_110, verdict, "
        "ensemble_lock, actual_value, hit, explanation_tags"
    )
    date_key = "LEFT(TRIM(game_date::text), 10)"

    q_past = [
        f"SELECT {cols} FROM prediction_log",
        f"WHERE {date_key} < :ny",
    ]
    q_recent = [
        f"SELECT {cols} FROM prediction_log",
        f"WHERE {date_key} >= :ny",
    ]
    params_p: dict[str, Any] = {"ny": ny_key, "lim": past_lim}
    params_r: dict[str, Any] = {"ny": ny_key, "lim": recent_lim}
    if stat:
        q_past.append("AND stat = :stat")
        q_recent.append("AND stat = :stat")
        params_p["stat"] = stat
        params_r["stat"] = stat

    q_past.append("ORDER BY game_date DESC NULLS LAST, logged_at DESC NULLS LAST LIMIT :lim")
    q_recent.append("ORDER BY game_date DESC NULLS LAST, logged_at DESC NULLS LAST LIMIT :lim")

    past_rows = session.execute(text("\n".join(q_past)), params_p).fetchall()
    recent_rows = session.execute(text("\n".join(q_recent)), params_r).fetchall()

    by_id: dict[int, Any] = {}
    for r in past_rows:
        by_id[int(r.id)] = r
    for r in recent_rows:
        by_id.setdefault(int(r.id), r)

    merged = list(by_id.values())

    def sort_key(row: Any) -> tuple[str, str]:
        gd_raw = row.game_date
        if hasattr(gd_raw, "isoformat"):
            gd = gd_raw.isoformat()[:10]
        else:
            gd = str(gd_raw or "")[:10]
        lt = row.logged_at.isoformat() if row.logged_at else ""
        return (gd, lt)

    merged.sort(key=sort_key, reverse=True)
    return merged[:cap]


# ESPN / odds feed vs NBA.com team codes (and DB `teams.abbreviation`).
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


def _norm_abbr_token(a: str) -> str:
    x = (a or "").strip().upper()
    return _TEAM_ABBR_ALIASES.get(x, x)


def _abbr_equiv(a: str, b: str) -> bool:
    return _norm_abbr_token(a) == _norm_abbr_token(b)


def _game_pair_matches(home_abbr: str, away_abbr: str, log_t1: str, log_t2: str) -> bool:
    return (
        (_abbr_equiv(home_abbr, log_t1) and _abbr_equiv(away_abbr, log_t2))
        or (_abbr_equiv(home_abbr, log_t2) and _abbr_equiv(away_abbr, log_t1))
    )


def _strip_diacritics(s: str) -> str:
    """ASCII fold for matching (Schröder vs Schroder, Dončić vs Doncic)."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _normalize_player_name_key(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s).strip()).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$", "", s, flags=re.I)
    s = s.strip()
    return _strip_diacritics(s)


def _prop_grade_hit(actual_value: float, line: float, best_side: str) -> bool | None:
    """
    OVER/UNDER vs line. Exact tie = push → None (not a loss). Strict > / < otherwise.
    """
    side = (best_side or "").upper()
    if abs(actual_value - line) < 1e-9:
        return None
    if side == "OVER":
        return actual_value > line
    if side == "UNDER":
        return actual_value < line
    return None


def _run_recent_games_sync() -> bool:
    """Pull a wide NBA window so slate games/box scores exist before we resolve props."""
    script = os.path.join(_api_dir, "update_recent_games.py")
    if not os.path.exists(script):
        return False
    env = os.environ.copy()
    env["LOOKBACK_HOURS"] = os.getenv("PARLAY_LOOKBACK_HOURS", "168")
    r = subprocess.run(
        [sys.executable, script],
        cwd=_api_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return r.returncode == 0


def _python_resolve_prediction_logs(
    session: Session,
    stat_filter: str | None,
    gdate_filter: str | None,
    *,
    batch_limit: int = 2500,
) -> int:
    """
    Match pending rows to box scores when SQL joins fail (unicode names, abbr aliases, date text).
    """
    prows = session.execute(text("SELECT id, full_name FROM players")).fetchall()
    by_key: dict[str, list[int]] = {}
    for pid, fn in prows:
        k = _normalize_player_name_key(str(fn or ""))
        if not k:
            continue
        by_key.setdefault(k, []).append(int(pid))

    n_done = 0
    pend_parts = [
        "SELECT id, game_date, player_name, team_abbr, opponent, stat, line, best_side",
        "FROM prediction_log",
        "WHERE hit IS NULL AND actual_value IS NULL",
    ]
    pparams: dict[str, Any] = {"lim": batch_limit}
    if stat_filter:
        pend_parts.append("AND stat = :stat")
        pparams["stat"] = stat_filter
    if gdate_filter:
        pend_parts.append("AND TRIM(game_date) = TRIM(:gd)")
        pparams["gd"] = gdate_filter
    pend_parts.append("ORDER BY logged_at ASC LIMIT :lim")
    pending = session.execute(text("\n".join(pend_parts)), pparams).fetchall()

    for row in pending:
        if str(row.stat or "").lower() == "win":
            continue
        pl_id = int(row.id)
        gd_raw = str(row.game_date or "").strip()[:10]
        if len(gd_raw) < 10:
            continue
        try:
            datetime.strptime(gd_raw, "%Y-%m-%d")
        except ValueError:
            continue

        slate = session.execute(
            text("""
                SELECT g.id, th.abbreviation AS h, ta.abbreviation AS a
                FROM games g
                JOIN teams th ON th.id = g.home_team_id
                JOIN teams ta ON ta.id = g.away_team_id
                WHERE g.game_date = CAST(:gd AS DATE)
            """),
            {"gd": gd_raw},
        ).fetchall()
        hits = [s for s in slate if _game_pair_matches(s.h, s.a, row.team_abbr, row.opponent)]
        if len(hits) != 1:
            continue
        gid = int(hits[0].id)

        pk = _normalize_player_name_key(str(row.player_name or ""))
        cand_pids = by_key.get(pk)
        if not cand_pids:
            # First initial + last name when unicode/format differs (e.g. Porziņģis).
            parts = pk.split()
            if len(parts) >= 2 and parts[0] and parts[-1]:
                first0, last = parts[0][0].lower(), parts[-1].lower()
                loose = []
                for k2, ids in by_key.items():
                    ps = k2.split()
                    if (
                        len(ps) >= 2
                        and ps[-1].lower() == last
                        and ps[0][0].lower() == first0
                    ):
                        loose.extend(ids)
                cand_pids = list(dict.fromkeys(loose))
                if len(cand_pids) > 4:
                    cand_pids = None
        if not cand_pids:
            continue

        pbs_row = None
        log_abbr = _norm_abbr_token(str(row.team_abbr or ""))
        for pid in cand_pids:
            r2 = None
            if log_abbr:
                r2 = session.execute(
                    text("""
                        SELECT COALESCE(pbs.pts, 0) AS pts, COALESCE(pbs.reb, 0) AS reb,
                               COALESCE(pbs.ast, 0) AS ast, COALESCE(pbs.stl, 0) AS stl,
                               COALESCE(pbs.blk, 0) AS blk, COALESCE(pbs.fg3m, 0) AS fg3m
                        FROM player_box_scores_traditional pbs
                        JOIN teams t ON t.id = pbs.team_id
                        WHERE pbs.game_id = :g AND pbs.player_id = :p
                          AND UPPER(TRIM(t.abbreviation)) = :abbr
                        LIMIT 1
                    """),
                    {"g": gid, "p": pid, "abbr": log_abbr},
                ).fetchone()
            if r2 is None:
                r2 = session.execute(
                    text("""
                        SELECT COALESCE(pts, 0) AS pts, COALESCE(reb, 0) AS reb,
                               COALESCE(ast, 0) AS ast, COALESCE(stl, 0) AS stl,
                               COALESCE(blk, 0) AS blk, COALESCE(fg3m, 0) AS fg3m
                        FROM player_box_scores_traditional
                        WHERE game_id = :g AND player_id = :p
                        LIMIT 1
                    """),
                    {"g": gid, "p": pid},
                ).fetchone()
            if r2 is not None:
                pbs_row = r2
                break
        if pbs_row is None:
            continue

        stat = str(row.stat or "").lower()
        if stat == "pts":
            actual_value = float(pbs_row.pts)
        elif stat == "reb":
            actual_value = float(pbs_row.reb)
        elif stat == "ast":
            actual_value = float(pbs_row.ast)
        elif stat == "stl":
            actual_value = float(pbs_row.stl)
        elif stat == "blk":
            actual_value = float(pbs_row.blk)
        elif stat == "fg3":
            actual_value = float(pbs_row.fg3m)
        else:
            continue

        line = float(row.line)
        best_side = str(row.best_side or "").upper()
        hit = _prop_grade_hit(actual_value, line, best_side)

        session.execute(
            text("""
                UPDATE prediction_log
                SET actual_value = :actual_value, hit = :hit
                WHERE id = :id
            """),
            {"actual_value": actual_value, "hit": hit, "id": pl_id},
        )
        n_done += 1

    return n_done


def _run_db_resolution_pipeline(
    session: Session,
    stat_filter: str | None,
    gdate_filter: str | None,
    limit: int,
) -> tuple[int, int]:
    """SQL + in-DB python resolution. Returns (total_updates, python_only_updates)."""
    _nm_player = (
        "LOWER(TRIM(regexp_replace(COALESCE(p.full_name, ''), "
        r"'[[:space:]]+(jr\.?|sr\.?|ii|iii|iv|v)$', '', 'i')))"
    )
    _nm_log = (
        "LOWER(TRIM(regexp_replace(COALESCE(pl.player_name, ''), "
        r"'[[:space:]]+(jr\.?|sr\.?|ii|iii|iv|v)$', '', 'i')))"
    )
    resolve_sql = text(
        """
                SELECT DISTINCT ON (pl.id)
                    pl.id,
                    pl.line,
                    pl.best_side,
                    pl.stat,
                    CASE pl.stat
                      WHEN 'pts' THEN COALESCE(pbs.pts, 0)
                      WHEN 'reb' THEN COALESCE(pbs.reb, 0)
                      WHEN 'ast' THEN COALESCE(pbs.ast, 0)
                      WHEN 'stl' THEN COALESCE(pbs.stl, 0)
                      WHEN 'blk' THEN COALESCE(pbs.blk, 0)
                      WHEN 'fg3' THEN COALESCE(pbs.fg3m, 0)
                    END AS actual_value
                FROM prediction_log pl
                JOIN players p
                  ON """
        + _nm_player
        + """ = """
        + _nm_log
        + """
                JOIN player_box_scores_traditional pbs
                  ON pbs.player_id = p.id
                JOIN games g
                  ON g.id = pbs.game_id
                 AND g.game_date = CAST(NULLIF(TRIM(pl.game_date), '') AS DATE)
                JOIN teams th
                  ON th.id = g.home_team_id
                JOIN teams ta
                  ON ta.id = g.away_team_id
                JOIN teams tpb
                  ON tpb.id = pbs.team_id
                WHERE pl.hit IS NULL
                  AND pl.actual_value IS NULL
                  AND pl.game_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND (:stat IS NULL OR pl.stat = :stat)
                  AND (:gdate IS NULL OR TRIM(pl.game_date) = TRIM(:gdate))
                  AND pbs.team_id IN (g.home_team_id, g.away_team_id)
                  AND (
                        (
                            UPPER(TRIM(pl.team_abbr)) IN (UPPER(TRIM(th.abbreviation)), UPPER(TRIM(ta.abbreviation)))
                            AND UPPER(TRIM(pl.opponent)) IN (UPPER(TRIM(th.abbreviation)), UPPER(TRIM(ta.abbreviation)))
                        )
                     OR UPPER(TRIM(tpb.abbreviation)) = UPPER(TRIM(pl.team_abbr))
                  )
                ORDER BY pl.id, g.id
                LIMIT :limit
            """
    )
    resolve_sql_fallback = text(
        """
                SELECT DISTINCT ON (pl.id)
                    pl.id,
                    pl.line,
                    pl.best_side,
                    pl.stat,
                    CASE pl.stat
                      WHEN 'pts' THEN COALESCE(pbs.pts, 0)
                      WHEN 'reb' THEN COALESCE(pbs.reb, 0)
                      WHEN 'ast' THEN COALESCE(pbs.ast, 0)
                      WHEN 'stl' THEN COALESCE(pbs.stl, 0)
                      WHEN 'blk' THEN COALESCE(pbs.blk, 0)
                      WHEN 'fg3' THEN COALESCE(pbs.fg3m, 0)
                    END AS actual_value
                FROM prediction_log pl
                JOIN players p
                  ON """
        + _nm_player
        + """ = """
        + _nm_log
        + """
                JOIN player_box_scores_traditional pbs
                  ON pbs.player_id = p.id
                JOIN games g
                  ON g.id = pbs.game_id
                 AND g.game_date = CAST(NULLIF(TRIM(pl.game_date), '') AS DATE)
                WHERE pl.hit IS NULL
                  AND pl.actual_value IS NULL
                  AND pl.game_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
                  AND (:stat IS NULL OR pl.stat = :stat)
                  AND (:gdate IS NULL OR TRIM(pl.game_date) = TRIM(:gdate))
                  AND pbs.team_id IN (g.home_team_id, g.away_team_id)
                  AND (
                        SELECT COUNT(DISTINCT pbs2.game_id)
                        FROM player_box_scores_traditional pbs2
                        JOIN games g2 ON g2.id = pbs2.game_id
                        WHERE pbs2.player_id = p.id
                          AND g2.game_date = CAST(NULLIF(TRIM(pl.game_date), '') AS DATE)
                  ) = 1
                ORDER BY pl.id, pbs.game_id
                LIMIT :limit
            """
    )
    params = {"stat": stat_filter, "gdate": gdate_filter, "limit": limit}
    updated = 0
    python_resolved = 0
    updated += _resolve_win_predictions_from_games(session, limit)
    session.flush()
    # Python first: diacritic-folded names + team-scoped box scores beat strict SQL text joins.
    for _ in range(30):
        n = _python_resolve_prediction_logs(
            session, stat_filter, gdate_filter, batch_limit=min(limit, 3000)
        )
        python_resolved += n
        updated += n
        session.flush()
        if n == 0:
            break
    for _ in range(40):
        progressed = False
        for stmt in (resolve_sql, resolve_sql_fallback):
            pending_rows = session.execute(stmt, params).fetchall()
            if not pending_rows:
                continue
            progressed = True
            for r in pending_rows:
                actual_value = float(r.actual_value)
                line = float(r.line)
                best_side = str(r.best_side or "").upper()
                hit = _prop_grade_hit(actual_value, line, best_side)
                session.execute(
                    text("""
                        UPDATE prediction_log
                        SET actual_value = :actual_value,
                            hit          = :hit
                        WHERE id = :id
                    """),
                    {"actual_value": actual_value, "hit": hit, "id": int(r.id)},
                )
                updated += 1
        session.flush()
        if not progressed:
            break
    for _ in range(20):
        n = _python_resolve_prediction_logs(
            session, stat_filter, gdate_filter, batch_limit=min(limit, 3000)
        )
        python_resolved += n
        updated += n
        session.flush()
        if n == 0:
            break
    return updated, python_resolved


def _nba_resolve_via_api(
    session: Session,
    stat_filter: str | None,
    gdate_filter: str | None,
    ny_cutoff: str,
    *,
    max_rows: int,
    sleep_s: float = 0.55,
) -> int:
    """Grade pending rows before ny_cutoff using NBA Stats API (persists to prediction_log)."""
    if os.getenv("PARLAY_DISABLE_NBA_RESOLVE", "").lower() in ("1", "true", "yes"):
        return 0
    try:
        from resolve_nba_live import clear_trad_cache, fetch_player_stat_from_nba
    except Exception:
        return 0

    clear_trad_cache()
    parts = [
        "SELECT id, game_date, player_name, team_abbr, opponent, stat, line, best_side",
        "FROM prediction_log",
        "WHERE hit IS NULL AND actual_value IS NULL",
        # Include today's NY slate: finished games grade via NBA API; in-progress box scores are skipped until Final.
        "AND game_date <= :ny",
        "AND game_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'",
    ]
    pparams: dict[str, Any] = {"ny": ny_cutoff[:10], "lim": max_rows}
    if stat_filter:
        parts.append("AND stat = :stat")
        pparams["stat"] = stat_filter
    if gdate_filter:
        parts.append("AND TRIM(game_date) = TRIM(:gd)")
        pparams["gd"] = gdate_filter
    # Newest slates first so yesterday grades before older pending rows.
    parts.append("ORDER BY game_date DESC NULLS LAST, logged_at ASC NULLS LAST LIMIT :lim")
    rows = session.execute(text("\n".join(parts)), pparams).fetchall()
    n = 0
    for row in rows:
        if str(row.stat or "").lower() == "win":
            continue
        val = fetch_player_stat_from_nba(
            str(row.game_date),
            str(row.team_abbr or ""),
            str(row.opponent or ""),
            str(row.player_name or ""),
            str(row.stat or ""),
            sleep_s=sleep_s,
        )
        if val is None:
            continue
        line = float(row.line)
        best_side = str(row.best_side or "").upper()
        if best_side not in ("OVER", "UNDER"):
            continue
        hit = _prop_grade_hit(float(val), line, best_side)
        session.execute(
            text("""
                UPDATE prediction_log
                SET actual_value = :av, hit = :hit
                WHERE id = :id
            """),
            {"av": float(val), "hit": hit, "id": int(row.id)},
        )
        n += 1
    return n


def _run_refresh_job(
    job_id: str,
    stat: str,
    day: str,
    n_sims: int,
    fresh_odds: bool,
) -> None:
    """Background wrapper for /api/refresh that preserves existing simulation pipeline."""
    global _refresh_running
    try:
        ny_today = datetime.now(_NY_TZ).date()
        target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
        data = _run_full_pipeline(
            stat,
            target_date,
            4,
            n_sims,
            use_odds_cache=not fresh_odds,
        )
        cache_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "game_date": data["game_date"],
            "stat": stat,
            "results": data["results"],
            "otb": data["otb"],
            "skipped": data["skipped"],
            "day": day,
        }
        _cache_store[_cache_key(data["game_date"], stat)] = cache_entry
        with _jobs_lock:
            JOBS[job_id] = {
                "status": "completed",
                "result": {
                    "status": "ok",
                    "timestamp": cache_entry["timestamp"],
                    "game_date": data["game_date"],
                    "stat": stat,
                    "n_simulated": len(data["results"]),
                    "n_skipped": len(data["skipped"]),
                    "otb": data["otb"],
                    "results": data["results"],
                },
            }
    except Exception as exc:
        with _jobs_lock:
            JOBS[job_id] = {"status": "failed", "error": str(exc), "result": None}
    finally:
        _refresh_running = False


from routers.health import router as health_router
from routers.history import router as history_router
from routers.jobs import router as jobs_router
from routers.simulations import router as simulations_router

app.include_router(simulations_router)
app.include_router(jobs_router)
app.include_router(history_router)
app.include_router(health_router)
