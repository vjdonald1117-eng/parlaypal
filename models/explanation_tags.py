"""
Explainability tags for projection rows — short UI badges derived only from
pipeline data (ProjectionResult + optional game market fields).
"""

from __future__ import annotations

from dataclasses import dataclass

from models.player_projections import ProjectionResult

# Vegas-style game total benchmarks (NBA; used when closing_total is present).
_HIGH_TOTAL_THRESHOLD = 227.5
_LOW_TOTAL_THRESHOLD = 214.5
# Wide spread → bench units / script volatility (uses abs(home spread) from DB).
_BLOWOUT_SPREAD_ABS = 10.0


@dataclass(frozen=True)
class ExplanationContext:
    """Inputs for tag generation — all optional fields beyond proj are market / slate context."""

    proj: ProjectionResult
    stat: str = "pts"
    closing_total: float | None = None
    closing_spread: float | None = None
    rest_days_before_game: int | None = None


def generate_explanation_tags(context: ExplanationContext) -> list[str]:
    """
    Return 1–3 short driver tags for UI badges, ranked by importance.
    Uses: defensive rating context, pace multipliers, NMU bumps, coach confidence,
    rust flag, and when provided closing total / spread from the games table.
    """
    proj = context.proj
    stat = (context.stat or "pts").strip().lower()
    bump = float(getattr(proj, f"bump_{stat}", 0.0) or 0.0)
    n_out = len(proj.injured_out)

    candidates: list[tuple[float, str]] = []

    if context.rest_days_before_game is not None:
        rd = context.rest_days_before_game
        if rd == 0:
            candidates.append((92.0, "B2B fatigue risk"))
        elif rd >= 2:
            candidates.append((55.0, "Extra rest"))

    if proj.is_rust:
        candidates.append((95.0, "Rust game"))

    if context.closing_total is not None:
        ct = float(context.closing_total)
        if ct >= _HIGH_TOTAL_THRESHOLD:
            candidates.append((78.0, "High game total"))
        elif ct <= _LOW_TOTAL_THRESHOLD:
            candidates.append((78.0, "Low game total"))

    if context.closing_spread is not None and abs(float(context.closing_spread)) >= _BLOWOUT_SPREAD_ABS:
        candidates.append((72.0, "Blowout script risk"))

    if proj.matchup:
        m = float(proj.matchup.multiplier)
        vs = float(proj.matchup.vs_league)
        if m >= 1.04 and vs >= 1.5:
            candidates.append((86.0, "Soft defense"))
        elif m <= 0.96 and vs <= -1.5:
            candidates.append((86.0, "Stingy defense"))
        elif m >= 1.04:
            candidates.append((80.0, "Softer matchup"))
        elif m <= 0.96:
            candidates.append((80.0, "Tough matchup"))

    pm = float(proj.pace_multiplier)
    if pm >= 1.02:
        candidates.append((70.0, "High pace"))
    elif pm <= 0.98:
        candidates.append((70.0, "Slow pace"))

    if n_out > 0:
        if bump >= 0.35:
            candidates.append((88.0, "Usage spike"))
        else:
            candidates.append((84.0, f"Thin roster ({n_out})"))

    cb = float(proj.confidence_boost)
    if cb >= 0.06:
        candidates.append((50.0, "Stable minutes"))
    elif cb <= -0.06:
        candidates.append((50.0, "Minutes volatility"))

    candidates.sort(key=lambda x: -x[0])
    out: list[str] = []
    seen: set[str] = set()
    for _, label in candidates:
        if label in seen:
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= 3:
            break

    return out
