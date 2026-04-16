"""Pydantic request/response models for the Parlay Pal API."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ResolveItem(BaseModel):
    id: int
    actual_value: float


class ResolveRequest(BaseModel):
    items: list[ResolveItem]


class ProjectionRow(BaseModel):
    """
    One player prop row from GET /api/projections (and related) `results[]`.
    Fields are optional so older caches and DB replays still validate; unknown keys are kept.
    """

    model_config = ConfigDict(extra="allow")

    player_name: Optional[str] = None
    team_abbr: Optional[str] = None
    opponent: Optional[str] = None
    stat: Optional[str] = None
    line: Optional[float] = None
    heuristic_mean: Optional[float] = None
    projected_mean: Optional[float] = None
    std_dev: Optional[float] = None
    over_pct: Optional[float] = None
    under_pct: Optional[float] = None
    best_side: Optional[str] = None
    best_pct: Optional[float] = None
    win_probability: Optional[float] = None
    win_pct: Optional[float] = None
    ev_per_110: Optional[float] = None
    ev: Optional[float] = None
    verdict: Optional[str] = None
    line_source: Optional[str] = None
    bump_pts: Optional[float] = None
    def_mult: Optional[float] = None
    pace_mult: Optional[float] = None
    conf_boost: Optional[float] = None
    is_rust: Optional[bool] = None
    gp: Optional[int] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    ml_mean: Optional[float] = None
    xgb_mean: Optional[float] = None
    xgb_over_pct: Optional[float] = None
    xgb_under_pct: Optional[float] = None
    xgb_best_side: Optional[str] = None
    xgb_best_pct: Optional[float] = None
    xgb_ev_per_110: Optional[float] = None
    xgb_available: Optional[bool] = None
    ensemble_lock: Optional[bool] = None
    sim_note: Optional[str] = None
    explanation_tags: list[str] = Field(
        default_factory=list,
        description="Short driver badges (pace, defense, totals, NMU, etc.).",
    )


class ProjectionsResponse(BaseModel):
    """Response body for GET /api/projections."""

    model_config = ConfigDict(extra="allow")

    timestamp: Optional[str] = None
    game_date: Optional[str] = None
    stat: Optional[str] = None
    day: Optional[str] = None
    n_results: int = 0
    results: list[ProjectionRow] = Field(default_factory=list)
    cache_miss: Optional[bool] = None
    message: Optional[str] = None


class HistoryRecord(BaseModel):
    """One row from GET /api/history `records[]` (prediction_log)."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    logged_at: Optional[str] = None
    game_date: Optional[str] = None
    player_name: Optional[str] = None
    team_abbr: Optional[str] = None
    opponent: Optional[str] = None
    stat: Optional[str] = None
    line: Optional[float] = None
    line_source: Optional[str] = None
    heuristic_mean: Optional[float] = None
    ml_mean: Optional[float] = None
    over_pct: Optional[float] = None
    under_pct: Optional[float] = None
    best_side: Optional[str] = None
    win_probability: Optional[float] = None
    ev_per_110: Optional[float] = None
    verdict: Optional[str] = None
    ensemble_lock: Optional[bool] = None
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    explanation_tags: list[str] = Field(
        default_factory=list,
        description="Same badge list as live /api/projections when logged from PlayerSim.",
    )


class HistoryResponse(BaseModel):
    """Response body for GET /api/history."""

    model_config = ConfigDict(extra="allow")

    n_records: int = 0
    n_resolved: int = 0
    overall_accuracy: Optional[float] = None
    lock_accuracy: Optional[float] = None
    records: list[HistoryRecord] = Field(default_factory=list)
