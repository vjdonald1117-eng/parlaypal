import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from api import (
    JOBS,
    N_SIMS,
    _NY_TZ,
    _PROJECTION_STATS,
    _UNIFIED_N_SIMS_DEFAULT,
    _cache_key,
    _cache_store,
    _jobs_lock,
    _mixed_top_props,
    _normalize_day_query,
    _normalize_prop_stat_param,
    _refresh_running,
    _results_from_prediction_log,
    _run_refresh_job,
    _run_unified_pipeline,
    _sort_key_edge_high_first,
    _write_unified_caches,
    engine,
    fetch_h2h_moneyline_board,
    trigger_schedule_odds_sync,
)
from schemas import ProjectionsResponse

router = APIRouter()


def _run_refresh_all_job(job_id: str, day: str, n_sims: int, fresh_odds: bool) -> None:
    """Background wrapper for /api/v1/refresh-all returning job-style progress."""
    import api as api_module

    try:
        ny_today = datetime.now(_NY_TZ).date()
        target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
        data = _run_unified_pipeline(
            target_date,
            n_sims=n_sims,
            use_odds_cache=not fresh_odds,
        )
        ts = _write_unified_caches(data, day)
        with _jobs_lock:
            JOBS[job_id] = {
                "status": "completed",
                "result": {
                    "status": "ok",
                    "day": day,
                    "timestamp": ts,
                    **data,
                },
            }
    except Exception as exc:
        with _jobs_lock:
            JOBS[job_id] = {"status": "failed", "error": str(exc), "result": None}
    finally:
        api_module._refresh_running = False


@router.post("/api/refresh")
async def refresh(
    background_tasks: BackgroundTasks,
    request: Request,
    day: str = Query(default="tomorrow"),
    n_sims: int = Query(default=N_SIMS, ge=1000, le=50000),
    fresh_odds: bool = Query(
        default=False,
        description="Bypass scripts/odds_cache_*.json (30m TTL) and refetch DK/FD lines from The Odds API.",
    ),
):
    stat = _normalize_prop_stat_param(request.query_params.get("stat") or "pts")
    day = _normalize_day_query(day)
    import api as api_module
    if api_module._refresh_running:
        raise HTTPException(status_code=409, detail="A refresh is already in progress.")

    api_module._refresh_running = True
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        JOBS[job_id] = {"status": "processing", "result": None}
    background_tasks.add_task(_run_refresh_job, job_id, stat, day, n_sims, fresh_odds)
    return {"job_id": job_id, "status": "processing"}


@router.post("/api/refresh-unified")
async def refresh_unified(
    day: str = Query(default="tomorrow"),
    n_sims: int = Query(default=_UNIFIED_N_SIMS_DEFAULT, ge=1000, le=50000),
    fresh_odds: bool = Query(
        default=False,
        description="Bypass local odds file cache; refetch all prop lines from The Odds API.",
    ),
):
    day = _normalize_day_query(day)
    import api as api_module
    if api_module._refresh_running:
        raise HTTPException(status_code=409, detail="A refresh is already in progress.")
    api_module._refresh_running = True
    try:
        ny_today = datetime.now(_NY_TZ).date()
        target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
        data = await run_in_threadpool(
            _run_unified_pipeline,
            target_date,
            n_sims=n_sims,
            use_odds_cache=not fresh_odds,
        )
        ts = _write_unified_caches(data, day)
    except Exception as exc:
        api_module._refresh_running = False
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        api_module._refresh_running = False

    return {
        "status": "ok",
        "day": day,
        "timestamp": ts,
        "game_date": data["game_date"],
        "n_sims": data["n_sims"],
        "stats": {s: len(data["by_stat"].get(s, [])) for s in _PROJECTION_STATS},
        "games": data["games"],
        "otb": data["otb"],
        "skipped": data["skipped"],
    }


@router.post("/api/refresh-all")
async def refresh_all(
    day: str = Query(default="today"),
    n_sims: int = Query(default=_UNIFIED_N_SIMS_DEFAULT, ge=1000, le=50000),
    fresh_odds: bool = Query(
        default=False,
        description="Bypass local odds file cache; refetch all prop lines from The Odds API.",
    ),
):
    day = _normalize_day_query(day)
    import api as api_module
    if api_module._refresh_running:
        raise HTTPException(status_code=409, detail="A refresh is already in progress.")
    api_module._refresh_running = True
    try:
        ny_today = datetime.now(_NY_TZ).date()
        target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
        data = await run_in_threadpool(
            _run_unified_pipeline,
            target_date,
            n_sims=n_sims,
            use_odds_cache=not fresh_odds,
        )
        ts = _write_unified_caches(data, day)
        results = {
            stat: {
                "timestamp": ts,
                "game_date": data["game_date"],
                "n_simulated": len(data["by_stat"].get(stat, [])),
                "n_skipped": len(data["skipped"]),
                "otb": data["otb"],
            }
            for stat in _PROJECTION_STATS
        }
    except Exception as exc:
        api_module._refresh_running = False
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        api_module._refresh_running = False

    return {"status": "ok", "day": day, "n_sims": data["n_sims"], "stats": results}


@router.post("/api/v1/refresh-all")
async def refresh_all_v1(
    background_tasks: BackgroundTasks,
    day: str = Query(default="today"),
    n_sims: int = Query(default=_UNIFIED_N_SIMS_DEFAULT, ge=1000, le=50000),
    fresh_odds: bool = Query(
        default=False,
        description="Bypass local odds file cache; refetch all prop lines from The Odds API.",
    ),
):
    day = _normalize_day_query(day)
    import api as api_module
    if api_module._refresh_running:
        raise HTTPException(status_code=409, detail="A refresh is already in progress.")

    api_module._refresh_running = True
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        JOBS[job_id] = {"status": "processing", "result": None}
    background_tasks.add_task(_run_refresh_all_job, job_id, day, n_sims, fresh_odds)
    return {"job_id": job_id, "status": "processing"}


@router.post("/api/sync-game-lines")
async def sync_game_lines():
    try:
        ok = await run_in_threadpool(trigger_schedule_odds_sync, False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "status": "ok" if ok else "partial",
        "schedule_sync_ok": ok,
        "detail": (
            "Spread/total written to games table (today + next 2 ET days)."
            if ok
            else "sync_nba_schedule.py exited non-zero; check API logs and ODDS_API_KEY."
        ),
    }


@router.get("/api/projections", response_model=ProjectionsResponse)
async def projections(
    request: Request,
    day: str = Query(default="tomorrow"),
    team: Optional[str] = Query(default=None),
):
    stat = _normalize_prop_stat_param(request.query_params.get("stat") or "pts")
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    entry = _cache_store.get(_cache_key(game_date, stat))

    results: list[dict[str, Any]]
    entry_ts: Any = None
    if entry and entry.get("results"):
        results = entry["results"]
        entry_ts = entry.get("timestamp")
    else:
        db_results, db_ts = _results_from_prediction_log(game_date, stat)
        if db_results:
            results = db_results
            entry_ts = db_ts
        else:
            return {
                "timestamp": None,
                "game_date": game_date,
                "stat": stat,
                "day": day,
                "n_results": 0,
                "results": [],
                "cache_miss": True,
                "message": "No simulation cache for this date yet. Click Run Simulations (same Today/Tomorrow) and wait—often several minutes. In Cursor use Run Task → Parlay Pal: Run (clean — one terminal) and watch lines prefixed [api] for uvicorn. API: POST /api/refresh-all?day=today|tomorrow",
            }

    if team:
        team_upper = team.upper()
        results = [r for r in results if r["team_abbr"].upper() == team_upper]

    results_sorted = sorted(results, key=_sort_key_edge_high_first, reverse=True)
    return {
        "timestamp": entry_ts,
        "game_date": entry.get("game_date") if entry else game_date,
        "stat": entry.get("stat") if entry else stat,
        "day": entry.get("day") if entry else day,
        "n_results": len(results_sorted),
        "results": results_sorted,
    }


@router.get("/api/projections-leaders")
async def projections_leaders(
    day: str = Query(default="today"),
    per_stat: int = Query(default=5, ge=1, le=15),
    max_total: int = Query(default=30, ge=5, le=60),
):
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    leaders: list[dict[str, Any]] = []
    for stat in _PROJECTION_STATS:
        entry = _cache_store.get(_cache_key(game_date, stat))
        if entry and entry.get("results"):
            raw_rows = entry["results"]
        else:
            raw_rows, _ = _results_from_prediction_log(game_date, stat)
        if not raw_rows:
            continue
        rs = sorted(raw_rows, key=_sort_key_edge_high_first, reverse=True)[:per_stat]
        for r in rs:
            row = dict(r)
            row["stat"] = stat
            leaders.append(row)
    leaders.sort(key=_sort_key_edge_high_first, reverse=True)
    return {
        "game_date": game_date,
        "day": day,
        "n": min(len(leaders), max_total),
        "leaders": leaders[:max_total],
    }


@router.get("/api/projections-combined")
async def projections_combined(day: str = Query(default="tomorrow")):
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    all_over: list[dict[str, Any]] = []
    all_under: list[dict[str, Any]] = []
    latest_ts: str | None = None
    cached_stats: list[str] = []
    for stat in _PROJECTION_STATS:
        entry = _cache_store.get(_cache_key(game_date, stat))
        if entry and entry.get("results"):
            stat_rows = entry["results"]
            ts = entry.get("timestamp")
        else:
            stat_rows, db_ts = _results_from_prediction_log(game_date, stat)
            ts = db_ts
        if not stat_rows:
            continue
        cached_stats.append(stat)
        if isinstance(ts, str) and (latest_ts is None or ts > latest_ts):
            latest_ts = ts
        for r in stat_rows:
            row = dict(r)
            row["stat"] = stat
            side = str(row.get("best_side") or "").upper()
            if side == "OVER":
                all_over.append(row)
            elif side == "UNDER":
                all_under.append(row)

    all_over.sort(key=_sort_key_edge_high_first, reverse=True)
    all_under.sort(key=_sort_key_edge_high_first, reverse=True)
    return {
        "timestamp": latest_ts,
        "game_date": game_date,
        "stat": "all",
        "day": day,
        "cached_stats": cached_stats,
        "overs": _mixed_top_props(all_over, final_n=10, per_stat=2),
        "unders": _mixed_top_props(all_under, final_n=10, per_stat=2),
    }


@router.get("/api/moneylines")
async def moneylines(day: str = Query(default="tomorrow")):
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    board = fetch_h2h_moneyline_board({game_date})
    overs = board.get("overs") or []
    unders = board.get("unders") or []
    warning: str | None = None
    if not overs and not unders:
        if not (os.getenv("ODDS_API_KEY") or "").strip():
            warning = "ODDS_API_KEY is not set in .env — book moneylines cannot load."
        else:
            warning = (
                f"No h2h lines returned for {game_date} (NY calendar). "
                "Games may not be listed yet at the odds provider, or names failed to match."
            )
    return {
        "timestamp": board.get("fetched_at"),
        "game_date": game_date,
        "stat": "ml",
        "day": day,
        "overs": overs,
        "unders": unders,
        "n_games": board.get("n_games") or 0,
        "warning": warning,
    }


@router.get("/api/sim-games")
async def sim_games(day: str = Query(default="tomorrow")):
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    with Session(engine) as session:
        rows = session.execute(
            text(
                """
                SELECT
                    g.id,
                    g.game_date,
                    g.simulated_total_points,
                    g.simulated_margin,
                    g.simulated_home_score,
                    g.simulated_away_score,
                    g.closing_spread,
                    g.closing_total,
                    th.abbreviation AS home_abbr,
                    ta.abbreviation AS away_abbr
                FROM games g
                JOIN teams th ON th.id = g.home_team_id
                JOIN teams ta ON ta.id = g.away_team_id
                WHERE g.game_date = CAST(:gd AS date)
                  AND g.prediction_engine = 'possession_sim_v1'
                ORDER BY g.id
                """
            ),
            {"gd": game_date},
        ).fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        projected_total = float(r.simulated_total_points) if r.simulated_total_points is not None else None
        margin = float(r.simulated_margin or 0.0)
        home_score = float(r.simulated_home_score or 0.0)
        away_score = float(r.simulated_away_score or 0.0)
        spread = float(r.closing_spread) if r.closing_spread is not None else None
        closing_total = float(r.closing_total) if r.closing_total is not None else None
        home = str(r.home_abbr)
        away = str(r.away_abbr)
        results.append(
            {
                "game_id": int(r.id),
                "player_name": f"{home} vs {away}",
                "team_abbr": home,
                "opponent": away,
                "matchup": f"{home} vs {away}",
                "stat": "gproj",
                "line": closing_total,
                "spread_line": spread,
                "heuristic_mean": projected_total,
                "ml_mean": None,
                "win_probability": 50.0,
                "over_pct": 50.0,
                "under_pct": 50.0,
                "best_side": "OVER" if margin >= 0 else "UNDER",
                "best_pct": 50.0,
                "ev_per_110": 0.0,
                "verdict": (
                    f"Sim {home} {home_score:.1f} - {away} {away_score:.1f}; "
                    f"Total {projected_total:.1f}" if projected_total is not None
                    else f"Sim {home} {home_score:.1f} - {away} {away_score:.1f}"
                ),
                "ensemble_lock": False,
            }
        )

    return {
        "game_date": game_date,
        "day": day,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


@router.get("/api/parlays-best")
async def parlays_best(day: str = Query(default="today")):
    day = _normalize_day_query(day)
    ny_today = datetime.now(_NY_TZ).date()
    target_date = ny_today if day == "today" else (ny_today + timedelta(days=1))
    game_date = target_date.strftime("%Y-%m-%d")
    decimal_odds = float(os.getenv("PARLAY_DEFAULT_DECIMAL_ODDS", "1.9091"))
    decimal_odds = max(1.01, decimal_odds)

    with Session(engine) as session:
        rows = session.execute(
            text(
                """
                SELECT
                    pp.game_id,
                    pp.player_id,
                    pp.prop_stat,
                    pp.vegas_line,
                    pp.over_probability,
                    p.full_name AS player_name,
                    th.abbreviation AS home_abbr,
                    ta.abbreviation AS away_abbr
                FROM public.player_props pp
                JOIN public.games g ON g.id = pp.game_id
                JOIN public.players p ON p.id = pp.player_id
                JOIN public.teams th ON th.id = g.home_team_id
                JOIN public.teams ta ON ta.id = g.away_team_id
                WHERE g.game_date = CAST(:gd AS date)
                  AND pp.prediction_engine = 'possession_sim_v1'
                  AND pp.prop_stat IN ('pts', 'reb', 'ast')
                  AND pp.over_probability IS NOT NULL
                """
            ),
            {"gd": game_date},
        ).fetchall()

    pool: list[dict[str, Any]] = []
    for r in rows:
        over_p = float(r.over_probability)
        over_p = min(0.999, max(0.001, over_p))
        under_p = 1.0 - over_p
        side = "OVER" if over_p >= under_p else "UNDER"
        sim_p = over_p if side == "OVER" else under_p
        ev = (sim_p * decimal_odds) - 1.0
        if ev <= 0:
            continue
        pool.append(
            {
                "game_id": int(r.game_id),
                "player_id": int(r.player_id),
                "player_name": str(r.player_name),
                "stat": str(r.prop_stat),
                "line": float(r.vegas_line) if r.vegas_line is not None else None,
                "side": side,
                "matchup": f"{r.away_abbr} @ {r.home_abbr}",
                "probability": sim_p,
                "decimal_odds": decimal_odds,
                "ev": ev,
            }
        )

    pool.sort(key=lambda x: float(x["ev"]), reverse=True)
    builder_pool = pool[:20]
    parlays: list[dict[str, Any]] = []
    from itertools import combinations
    for a, b, c in combinations(builder_pool, 3):
        if int(a["player_id"]) == int(b["player_id"]) or int(a["player_id"]) == int(c["player_id"]) or int(b["player_id"]) == int(c["player_id"]):
            continue
        p_win = float(a["probability"]) * float(b["probability"]) * float(c["probability"])
        dec = float(a["decimal_odds"]) * float(b["decimal_odds"]) * float(c["decimal_odds"])
        p_ev = (p_win * dec) - 1.0
        parlays.append(
            {
                "legs": [a, b, c],
                "parlay_probability": p_win,
                "parlay_decimal_odds": dec,
                "parlay_ev": p_ev,
            }
        )

    parlays.sort(key=lambda x: float(x["parlay_ev"]), reverse=True)
    return {
        "game_date": game_date,
        "day": day,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "builder_pool_size": len(builder_pool),
        "results": parlays[:5],
    }


@router.get("/api/calibration")
async def calibration(min_per_cell: int = Query(default=3, ge=1, le=500)):
    sql = text(
        """
        SELECT
          (FLOOR(
            CASE WHEN best_pct <= 1 THEN best_pct * 100.0 ELSE best_pct END / 5.0
          ) * 5)::int AS bucket_lo,
          UPPER(TRIM(COALESCE(best_side, ''))) AS side,
          COUNT(*)::bigint AS n,
          SUM(CASE WHEN hit THEN 1 ELSE 0 END)::bigint AS hits
        FROM prediction_log
        WHERE hit IS NOT NULL
          AND stat NOT IN ('win', 'unified')
          AND best_pct IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    )
    with Session(engine) as session:
        rows = session.execute(sql).fetchall()

    cells: list[dict[str, Any]] = []
    total_n = total_hits = 0
    for r in rows:
        n = int(r.n)
        hits = int(r.hits)
        total_n += n
        total_hits += hits
        if n < min_per_cell:
            continue
        mid = float(r.bucket_lo) + 2.5
        actual = round(100.0 * hits / n, 2) if n else 0.0
        cells.append(
            {
                "bucket_lo": int(r.bucket_lo),
                "bucket_hi": int(r.bucket_lo) + 5,
                "side": str(r.side or "").upper(),
                "n": n,
                "hits": hits,
                "predicted_mid_pct": mid,
                "actual_hit_pct": actual,
                "gap_pct": round(actual - mid, 2),
            }
        )

    overall = round(100.0 * total_hits / total_n, 2) if total_n else None
    return {
        "min_per_cell": min_per_cell,
        "cells": cells,
        "totals": {"n": total_n, "hits": total_hits, "overall_hit_pct": overall},
    }
