from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from api import (
    _NY_TZ,
    _coerce_explanation_tags_from_db,
    _fetch_merged_history_rows,
    _nba_resolve_via_api,
    _normalize_optional_history_stat,
    _run_db_resolution_pipeline,
    _run_recent_games_sync,
    engine,
    trigger_db_update,
)
from database import PredictionLog
from schemas import HistoryResponse, ResolveRequest

router = APIRouter()


@router.get("/api/history", response_model=HistoryResponse)
async def history(
    stat: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
    resolve_db: bool = Query(default=False),
    resolve_nba: bool = Query(default=False),
):
    stat = _normalize_optional_history_stat(stat)
    ny = datetime.now(_NY_TZ).date().isoformat()
    with Session(engine) as session:
        if resolve_db:
            try:
                _run_db_resolution_pipeline(session, stat, None, min(8000, max(limit * 2, 2000)))
                session.commit()
            except Exception:
                session.rollback()
        if resolve_nba:
            try:
                nba_cap = min(1500, max(500, limit // 3))
                _nba_resolve_via_api(session, stat, None, ny, max_rows=nba_cap, sleep_s=0.35)
                session.commit()
            except Exception:
                session.rollback()
        rows = _fetch_merged_history_rows(session, stat=stat, ny=ny, limit=limit)

    records = [
        {
            "id": r.id,
            "logged_at": r.logged_at.isoformat() if r.logged_at else None,
            "game_date": str(r.game_date)[:10] if r.game_date is not None else None,
            "player_name": r.player_name,
            "team_abbr": r.team_abbr,
            "opponent": r.opponent,
            "stat": r.stat,
            "line": r.line,
            "line_source": r.line_source,
            "heuristic_mean": r.heuristic_mean,
            "ml_mean": r.ml_mean,
            "over_pct": r.over_pct,
            "under_pct": r.under_pct,
            "best_side": r.best_side,
            "win_probability": r.best_pct,
            "ev_per_110": r.ev_per_110,
            "verdict": r.verdict,
            "ensemble_lock": r.ensemble_lock,
            "actual_value": r.actual_value,
            "hit": r.hit,
            "explanation_tags": _coerce_explanation_tags_from_db(getattr(r, "explanation_tags", None)),
        }
        for r in rows
    ]
    resolved = [r for r in records if r["hit"] is not None]
    accuracy = round(sum(1 for r in resolved if r["hit"]) / len(resolved) * 100, 1) if resolved else None
    lock_resolved = [r for r in resolved if r["ensemble_lock"]]
    lock_accuracy = round(sum(1 for r in lock_resolved if r["hit"]) / len(lock_resolved) * 100, 1) if lock_resolved else None
    return {
        "n_records": len(records),
        "n_resolved": len(resolved),
        "overall_accuracy": accuracy,
        "lock_accuracy": lock_accuracy,
        "records": records,
    }


@router.post("/api/resolve")
async def resolve_projections(payload: ResolveRequest):
    if not payload.items:
        raise HTTPException(status_code=400, detail="No items provided.")
    updated = 0
    with Session(engine) as session:
        ids = [item.id for item in payload.items]
        rows: list[PredictionLog] = session.query(PredictionLog).filter(PredictionLog.id.in_(ids)).all()
        row_by_id = {row.id: row for row in rows}
        for item in payload.items:
            row = row_by_id.get(item.id)
            if row is None:
                continue
            row.actual_value = float(item.actual_value)
            if row.best_side.upper() == "OVER":
                row.hit = row.actual_value > row.line
            elif row.best_side.upper() == "UNDER":
                row.hit = row.actual_value < row.line
            else:
                row.hit = None
            updated += 1
        session.commit()
    if updated == 0:
        raise HTTPException(status_code=404, detail="No matching prediction_log rows found for provided ids.")
    return {"status": "ok", "updated": updated}


@router.post("/api/update-history")
async def update_history(
    stat: str | None = Query(default=None),
    game_date: str | None = Query(default=None),
    limit: int = Query(default=15000, ge=1, le=20000),
    force_regrade: bool = Query(
        default=False,
        description="Clear hit/actual for this game_date (props only) then re-grade. Use after fixing matchers.",
    ),
):
    stat = _normalize_optional_history_stat(stat)
    if force_regrade and not (game_date and str(game_date).strip()):
        raise HTTPException(status_code=400, detail="force_regrade requires game_date=YYYY-MM-DD")
    sync_ok = False
    try:
        sync_ok = _run_recent_games_sync()
    except Exception:
        try:
            sync_ok = trigger_db_update(verbose=False)
        except Exception:
            pass

    stat_filter = stat
    gdate_filter = game_date
    ny = datetime.now(_NY_TZ).date().isoformat()
    with Session(engine) as session:
        if force_regrade and gdate_filter:
            session.execute(
                text(
                    """
                    UPDATE prediction_log
                    SET hit = NULL, actual_value = NULL
                    WHERE TRIM(game_date) = TRIM(:gd)
                      AND stat NOT IN ('win')
                      AND (:stat IS NULL OR stat = :stat)
                    """
                ),
                {"gd": str(gdate_filter).strip()[:10], "stat": stat_filter},
            )
            session.commit()
        updated, python_resolved = _run_db_resolution_pipeline(session, stat_filter, gdate_filter, limit)
        nba_resolved = _nba_resolve_via_api(
            session,
            stat_filter,
            gdate_filter,
            ny,
            max_rows=min(2500, limit),
            sleep_s=0.45,
        )
        updated += nba_resolved
        session.commit()
    return {
        "status": "ok",
        "updated": updated,
        "stat": stat_filter,
        "game_date": gdate_filter,
        "box_score_sync_ok": sync_ok,
        "python_resolved": python_resolved,
        "nba_resolved": nba_resolved,
    }
