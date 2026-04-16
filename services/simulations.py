"""
services/simulations.py
========================
Monte Carlo prop simulation, DB std-dev helpers, and single-player simulate_player.

Implementation moved from models/monte_carlo_sim.py and models/parlay_builder.py
without changing numerical behavior.
"""

import argparse
import logging
import sys

from core.logger import configure_app_logging

configure_app_logging()
logger = logging.getLogger(__name__)

import os
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import engine
from services.injuries import (
    get_injured_players,
    get_injured_players_cached,
    normalize_player_name,
)
from models.player_projections import (
    MIN_GP_DEFAULT,
    ProjectionResult,
    current_season,
    project_player,
)
from models.explanation_tags import ExplanationContext, generate_explanation_tags
from scripts.fetch_live_odds import (
    OddsLine,
    compute_ev_american,
    lookup_player_odds,
)
from models.rotation_coaching_features import (
    blowout_risk_from_spread,
    is_starter_for_team,
    starter_mean_scale,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SIMS                 = 5_000   # default target n (override via run_simulation n_sims)
MC_STAGE1_N            = 2_000   # first-stage batch before optional early exit
MC_EARLY_STOP_HIGH_PCT = 83.0
MC_EARLY_STOP_LOW_PCT  = 50.0
_RNG          = np.random.default_rng()   # module-level; avoids per-call construction
BREAKEVEN_PCT = 52.38   # -110 odds breakeven

# When projected mean sits below the line, widen simulation tails (blowouts, foul trouble, DNP risk).
UNDER_CONTEXT_LINE_RATIO = 0.985
UNDER_CONTEXT_STD_MULT = 1.08

# Asymmetric UNDER support: a share of trials simulate reduced minutes (foul trouble,
# blowout rest, minor injury) — skews the left tail beyond symmetric Normal tails.
GAME_SCRIPT_FAILURE_RATE = 0.08
GAME_SCRIPT_MINUTES_FACTOR_LO = 0.30
GAME_SCRIPT_MINUTES_FACTOR_HI = 0.75


def effective_std_for_prop(std_dev: float, sim_mean: float, line: float) -> float:
    """Apply slightly wider noise for under-side contexts (not symmetric inverse of over)."""
    if line <= 0 or std_dev <= 0:
        return std_dev
    if sim_mean < line * UNDER_CONTEXT_LINE_RATIO:
        return float(std_dev * UNDER_CONTEXT_STD_MULT)
    return float(std_dev)

# Maps --stat argument → attribute names on ProjectionResult
STAT_MAP = {
    "pts": ("final_pts", "baseline_pts", "bump_pts"),
    "reb": ("final_reb", "baseline_reb", "bump_reb"),
    "ast": ("final_ast", "baseline_ast", "bump_ast"),
    "stl": ("final_stl", "baseline_stl", "bump_stl"),
    "blk": ("final_blk", "baseline_blk", "bump_blk"),
    "fg3": ("final_fg3", "baseline_fg3", "bump_fg3"),
}

# player_box_scores_traditional column for SQL (stat key → column name)
STAT_SQL_COLUMN = {
    "fg3": "fg3m",
}

EDGE_TIERS = [
    (65.0, "VERY STRONG EDGE"),
    (60.0, "STRONG EDGE"),
    (55.0, "MODERATE EDGE"),
    (52.38, "SLIGHT LEAN"),
    (0.0, "NO EDGE"),
]

# ---------------------------------------------------------------------------
# 2K Simulation Context — manual playbook/matchup adjustments
# ---------------------------------------------------------------------------

# Fatigue penalty: applied ONLY to PTS (shooting efficiency degrades on
# back-to-backs; other stats are less directly tied to shot quality).
REST_PENALTY = {
    0: -0.05,   # back-to-back: -5% pts
    1:  0.00,   # one day rest: neutral
}
REST_2PLUS_BONUS = +0.02   # 2+ days rest: fresh legs, +2% pts

# Lockdown defender: applied to whichever stat is being simulated.
# --lockdown      elite defender is actively guarding this player: -5%
# --no-lockdown   favorable matchup, no lockdown scheme present:  +3%
LOCKDOWN_ON_ADJ  = -0.05
LOCKDOWN_OFF_ADJ = +0.03


@dataclass
class ContextAdjustments:
    """Holds all 2K simulation context inputs and the deltas they produce."""
    rest_days:        int | None    # None = not provided
    lockdown:         bool | None   # None = not provided, True = on, False = off
    stat:             str           # which stat is being simulated
    rest_pct:         float         # fractional adjustment from rest
    lockdown_pct:     float         # fractional adjustment from lockdown
    rest_delta:       float         # absolute mean change from rest
    lockdown_delta:   float         # absolute mean change from lockdown
    pre_context_mean: float         # mean before context (= proj.final_xxx)
    final_mean:       float         # mean after context (fed to numpy)

    @property
    def has_any_adjustment(self) -> bool:
        return self.rest_days is not None or self.lockdown is not None

    @property
    def total_delta(self) -> float:
        return round(self.rest_delta + self.lockdown_delta, 3)

    @property
    def rest_label(self) -> str:
        if self.rest_days == 0:
            return "0 days  (back-to-back)"
        if self.rest_days == 1:
            return "1 day   (normal rest)"
        if self.rest_days is not None:
            return f"{self.rest_days}+ days  (fresh legs)"
        return "not specified"

    @property
    def lockdown_label(self) -> str:
        if self.lockdown is True:
            return "ON   (elite defender guarding)"
        if self.lockdown is False:
            return "OFF  (favorable matchup, no lockdown)"
        return "not specified"


def apply_context_adjustments(
    pre_mean:  float,
    stat:      str,
    rest_days: int | None,
    lockdown:  bool | None,
) -> ContextAdjustments:
    """
    Compute rest and lockdown deltas and return a ContextAdjustments object.

    Rest penalty applies ONLY to PTS — it represents degraded shooting
    efficiency on no-rest nights, not reduced physical output overall.

    Lockdown adjustment applies to whichever stat is being simulated,
    since an elite defender affects every offensive action.
    """
    rest_pct = 0.0
    if stat == "pts" and rest_days is not None:
        if rest_days == 0:
            rest_pct = REST_PENALTY[0]
        elif rest_days == 1:
            rest_pct = REST_PENALTY[1]
        else:
            rest_pct = REST_2PLUS_BONUS

    lockdown_pct = 0.0
    if lockdown is True:
        lockdown_pct = LOCKDOWN_ON_ADJ
    elif lockdown is False:
        lockdown_pct = LOCKDOWN_OFF_ADJ

    rest_delta     = pre_mean * rest_pct
    lockdown_delta = pre_mean * lockdown_pct
    final_mean     = max(0.0, pre_mean + rest_delta + lockdown_delta)

    return ContextAdjustments(
        rest_days=rest_days,
        lockdown=lockdown,
        stat=stat,
        rest_pct=rest_pct,
        lockdown_pct=lockdown_pct,
        rest_delta=round(rest_delta, 3),
        lockdown_delta=round(lockdown_delta, 3),
        pre_context_mean=round(pre_mean, 3),
        final_mean=round(final_mean, 3),
    )


# ---------------------------------------------------------------------------
# DB — standard deviation
# ---------------------------------------------------------------------------

def get_stat_stddev(
    session: Session,
    player_id: int,
    stat: str,
    season: str,
) -> tuple[float, int] | None:
    """
    Return (std_dev, games_played) for *stat* over *season*.
    Falls back to the previous season if fewer than 3 games in current.
    Returns None if no data at all.
    """
    col = STAT_SQL_COLUMN.get(stat, stat)

    query = text(f"""
        SELECT ROUND(STDDEV(pbs.{col})::numeric, 3) AS std_dev,
               COUNT(*)                              AS gp
        FROM player_box_scores_traditional pbs
        JOIN games g ON pbs.game_id = g.id
        WHERE pbs.player_id  = :pid
          AND pbs.dnp_status = FALSE
          AND g.season       = :season
    """)

    for s in [season, f"{int(season[:4])-1}-{season[:4][2:]}"]:
        row = session.execute(query, {"pid": player_id, "season": s}).fetchone()
        if row and row.std_dev is not None and row.gp >= 3:
            return float(row.std_dev), int(row.gp)

    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _apply_game_script_failure_drawdown(sims: np.ndarray) -> None:
    """
    In-place: randomly mark ~GAME_SCRIPT_FAILURE_RATE of draws as "script failure"
    (early foul trouble, blowout bench, in-game injury) and scale those outcomes
    by a random minutes fraction in [GAME_SCRIPT_MINUTES_FACTOR_LO, HI], then
    clip so no stat is below 0.
    """
    n = int(sims.shape[0])
    if n <= 0:
        return
    k = min(n, max(0, int(round(n * GAME_SCRIPT_FAILURE_RATE))))
    if k == 0:
        return
    idx = _RNG.choice(n, size=k, replace=False)
    factors = _RNG.uniform(
        GAME_SCRIPT_MINUTES_FACTOR_LO,
        GAME_SCRIPT_MINUTES_FACTOR_HI,
        size=k,
    )
    sims[idx] *= factors
    np.clip(sims, 0.0, None, out=sims)


def summarize_prop_from_samples(sims: np.ndarray, line: float) -> dict:
    """
    OVER/UNDER stats from an existing draw vector (joint sims share RNG structure).
    """
    effective_n = int(sims.shape[0])
    if effective_n <= 0:
        raise ValueError("sims must be non-empty.")

    over_count = int(np.sum(sims > line))
    under_count = effective_n - over_count

    over_pct = over_count / effective_n * 100
    under_pct = under_count / effective_n * 100

    best_pct = max(over_pct, under_pct)
    best_side = "OVER" if over_pct >= under_pct else "UNDER"

    verdict = "NO EDGE"
    for threshold, label in EDGE_TIERS:
        if best_pct >= threshold:
            verdict = label
            break

    win_pct_dec = best_pct / 100
    ev_per_110 = round(win_pct_dec * 100 - (1 - win_pct_dec) * 110, 2)

    p10, p25, p50, p75, p90 = np.percentile(sims, [10, 25, 50, 75, 90])

    return {
        "over_count": over_count,
        "under_count": under_count,
        "over_pct": round(over_pct, 2),
        "under_pct": round(under_pct, 2),
        "best_side": best_side,
        "best_pct": round(best_pct, 2),
        "verdict": verdict,
        "ev_per_110": ev_per_110,
        "p10": round(float(p10), 1),
        "p25": round(float(p25), 1),
        "p50": round(float(p50), 1),
        "p75": round(float(p75), 1),
        "p90": round(float(p90), 1),
    }


def run_simulation(
    adj_mean: float,
    std_dev: float,
    line: float,
    n_sims: int | None = None,
) -> dict:
    """
    Draw samples from N(adj_mean, std_dev), clip at 0, apply a random
    GAME_SCRIPT_FAILURE_RATE minutes drawdown on a subset (left-tail / UNDER
    asymmetry), re-clip at 0, and return a results dict with raw sims plus metrics.

    Two-stage: run MC_STAGE1_N draws first; if more remain and best_pct is
    outside (MC_EARLY_STOP_LOW_PCT, MC_EARLY_STOP_HIGH_PCT), stop and set
    sim_note; otherwise complete to effective_n.
    """
    effective_n = int(n_sims) if n_sims is not None else N_SIMS
    if effective_n <= 0:
        raise ValueError("n_sims must be a positive integer.")

    n1 = min(MC_STAGE1_N, effective_n)
    sims1 = np.random.normal(adj_mean, std_dev, n1)
    np.clip(sims1, 0.0, None, out=sims1)
    _apply_game_script_failure_drawdown(sims1)
    out1 = summarize_prop_from_samples(sims1, line)

    sim_note = ""
    if n1 >= effective_n:
        out = out1
        out["sims"] = sims1
    else:
        bp = float(out1["best_pct"])
        if bp > MC_EARLY_STOP_HIGH_PCT or bp < MC_EARLY_STOP_LOW_PCT:
            out = out1
            out["sims"] = sims1
            sim_note = "Stopped at 2k"
        else:
            n2 = effective_n - n1
            sims2 = np.random.normal(adj_mean, std_dev, n2)
            np.clip(sims2, 0.0, None, out=sims2)
            _apply_game_script_failure_drawdown(sims2)
            sims = np.concatenate([sims1, sims2])
            out = summarize_prop_from_samples(sims, line)
            out["sims"] = sims

    out["sim_note"] = sim_note
    return out


# ---------------------------------------------------------------------------
# ASCII histogram
# ---------------------------------------------------------------------------

def make_histogram(
    sims:      "np.ndarray",
    line:      float,
    adj_mean:  float,
    stat_label: str,
    bar_width:  int = 40,
    n_bins:     int = 22,
) -> str:
    """
    Build an ASCII histogram of the simulation results, with:
      - A '|' marker at the line position
      - A '*' marker at the adjusted mean
    """
    lo = max(0.0, float(np.percentile(sims, 1)))
    hi = float(np.percentile(sims, 99))
    # Expand range slightly so line/mean markers always appear
    lo = min(lo, line - 1, adj_mean - 1)
    hi = max(hi, line + 1, adj_mean + 1)

    counts, edges = np.histogram(sims, bins=n_bins, range=(lo, hi))
    max_count = counts.max() if counts.max() > 0 else 1

    lines_out = []
    bin_width = (hi - lo) / n_bins

    for i, count in enumerate(counts):
        bin_lo = edges[i]
        bin_hi = edges[i + 1]
        bar_len = int(count / max_count * bar_width)
        bar = "#" * bar_len

        # Annotate if line falls inside this bin
        line_marker = ""
        if bin_lo <= line < bin_hi:
            line_marker = f"  <-- LINE ({line} {stat_label})"

        # Annotate if mean falls inside this bin
        mean_marker = ""
        if bin_lo <= adj_mean < bin_hi:
            mean_marker = f"  <-- MEAN ({adj_mean:.1f})"

        annotation = line_marker or mean_marker

        lines_out.append(
            f"  {bin_lo:5.1f} - {bin_hi:5.1f} | {bar:<{bar_width}s}{annotation}"
        )

    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(
    proj:     ProjectionResult,
    stat:     str,
    line:     float,
    std_dev:  float,
    std_gp:   int,
    results:  dict,
    sim_mean: float,
    context:  ContextAdjustments | None = None,
) -> None:
    _final_attr, base_attr, bump_attr = STAT_MAP[stat]
    baseline   = getattr(proj, base_attr)
    bump       = getattr(proj, bump_attr)
    stat_upper = stat.upper()

    sep  = "=" * 62
    dash = "-" * 62
    dot  = "." * 62

    opp_str = (
        f"vs  {proj.matchup.full_name} ({proj.matchup.abbreviation})"
        if proj.matchup else "no opponent specified"
    )

    logger.info(f"\n{sep}")
    logger.info(f"  Monte Carlo Simulation  --  {N_SIMS:,} trials")
    logger.info(sep)
    logger.info(f"  Player   : {proj.player.full_name} ({proj.player.team_abbr})")
    logger.info(f"  Matchup  : {opp_str}")
    logger.info(f"  Stat     : {stat_upper}")
    logger.info(f"  Season   : {proj.season}")
    logger.info(f"  Date     : {date.today()}")
    logger.info(dash)

    # --- Mean build-up: stats layer ---
    logger.info(f"  {'Stat':34s}  {'Value':>8}")
    logger.info(f"  {'Season baseline avg':34s}  {baseline:>8.2f}")

    if bump != 0:
        bsign = "+" if bump >= 0 else ""
        logger.info(f"  {'Next Man Up bump':34s}  {bsign}{bump:>7.2f}")

    if proj.matchup:
        m          = proj.matchup
        pre_proj   = getattr(proj, f"projected_{stat}")
        def_delta  = round(pre_proj * (m.multiplier - 1.0), 2)
        dsign      = "+" if def_delta >= 0 else ""
        logger.info(f"  {'Defensive mult ({:.3f})'.format(m.multiplier):34s}  {dsign}{def_delta:>7.2f}")

    pace_delta = getattr(proj, f"pace_delta_{stat}")
    if pace_delta != 0.0:
        psign = "+" if pace_delta >= 0 else ""
        pace_label = f"Pace mult ({proj.pace_multiplier:.3f})"
        logger.info(f"  {pace_label:34s}  {psign}{pace_delta:>7.2f}")

    conf_delta = getattr(proj, f"confidence_delta_{stat}")
    if conf_delta != 0.0:
        csign = "+" if conf_delta >= 0 else ""
        boost_pct = round(proj.confidence_boost * 100, 1)
        conf_label = f"Coach Confidence (+{boost_pct}%)"
        logger.info(f"  {conf_label:34s}  {csign}{conf_delta:>7.2f}")

    rust_delta = getattr(proj, f"rust_delta_{stat}")
    if proj.is_rust:
        rsign = "+" if rust_delta >= 0 else ""
        logger.info(f"  {'Rust penalty (-20%)':34s}  {rsign}{rust_delta:>7.2f}")

    # If context adjustments exist, show the intermediate mean then the 2K section.
    # Otherwise the stats-layer mean IS the simulation input.
    if context and context.has_any_adjustment:
        logger.info(dot)
        logger.info(f"  {'Post-stats mean':34s}  {context.pre_context_mean:>8.2f}")
        logger.info("")
        logger.info(f"  2K SIMULATION CONTEXT")

        # Rest row
        if context.rest_days is not None:
            if stat != "pts":
                # Rest only penalises pts; note it but show no delta
                logger.info(f"  {'Fatigue (' + context.rest_label + ')':34s}  "
                      f"{'N/A (PTS only)':>8s}")
            else:
                pct_str  = f"{context.rest_pct * 100:+.1f}%"
                dlt_sign = "+" if context.rest_delta >= 0 else ""
                logger.info(f"  {'Fatigue (' + context.rest_label + ')':34s}  "
                      f"{pct_str:>5s}   {dlt_sign}{context.rest_delta:>5.2f}")

        # Lockdown row
        if context.lockdown is not None:
            pct_str  = f"{context.lockdown_pct * 100:+.1f}%"
            dlt_sign = "+" if context.lockdown_delta >= 0 else ""
            logger.info(f"  {'Lockdown: ' + context.lockdown_label:34s}  "
                  f"{pct_str:>5s}   {dlt_sign}{context.lockdown_delta:>5.2f}")

        # Total context delta
        td_sign = "+" if context.total_delta >= 0 else ""
        logger.info(f"  {'Total context adjustment':34s}  "
              f"{'':>5s}   {td_sign}{context.total_delta:>5.2f}")
        logger.info(dot)
        logger.info(f"  {'Final simulation mean':34s}  {sim_mean:>8.2f}")
    else:
        logger.info(f"  {'Adjusted mean (simulation input)':34s}  {sim_mean:>8.2f}")

    logger.info(f"  {'Std dev ({} GP sample)'.format(std_gp):34s}  {std_dev:>8.2f}")
    logger.info(f"  {'Prop line':34s}  {line:>8.2f}")
    logger.info(dot)

    # --- Distribution chart ---
    logger.info(f"\n  Distribution of {N_SIMS:,} simulated {stat_upper} values:\n")
    logger.info(make_histogram(results["sims"], line, sim_mean, stat_upper))
    logger.info("")
    logger.info(dot)

    # --- Percentiles ---
    logger.info(
        f"  Percentiles:  "
        f"p10={results['p10']:.1f}  "
        f"p25={results['p25']:.1f}  "
        f"p50={results['p50']:.1f}  "
        f"p75={results['p75']:.1f}  "
        f"p90={results['p90']:.1f}"
    )
    logger.info(dot)

    # --- Over / Under ---
    over_bar  = "#" * int(results["over_pct"]  / 2)
    under_bar = "#" * int(results["under_pct"] / 2)

    logger.info(f"\n  OVER  {line} {stat_upper:<4s}  |  "
          f"{results['over_count']:>5,} trials  |  "
          f"{results['over_pct']:>5.1f}%  |  {over_bar}")
    logger.info(f"  UNDER {line} {stat_upper:<4s}  |  "
          f"{results['under_count']:>5,} trials  |  "
          f"{results['under_pct']:>5.1f}%  |  {under_bar}")
    logger.info(dot)

    # --- Verdict ---
    ev      = results["ev_per_110"]
    ev_sign = "+" if ev >= 0 else ""
    logger.info(f"\n  Lean       : {results['best_side']} {line} {stat_upper}")
    logger.info(f"  Win prob   : {results['best_pct']:.1f}%  "
          f"(breakeven at -110 odds = {BREAKEVEN_PCT}%)")
    logger.info(f"  EV / $110  : {ev_sign}{ev:.2f}")
    logger.info("")

    verdict    = results["verdict"]
    edge_above = round(results["best_pct"] - BREAKEVEN_PCT, 2)

    if verdict == "NO EDGE":
        logger.info(f"  VERDICT: {verdict}")
        logger.info(f"  Win probability ({results['best_pct']:.1f}%) is below the {BREAKEVEN_PCT}% breakeven.")
        logger.info( "  Skip this prop — no mathematical edge at standard -110.")
    elif verdict == "SLIGHT LEAN":
        logger.info(f"  VERDICT: {verdict} {results['best_side']}")
        logger.info(f"  Only {edge_above:.1f}% above breakeven. Proceed with caution.")
        logger.info( "  Small edge — only play if you have a strong read on the game.")
    elif verdict == "MODERATE EDGE":
        logger.info(f"  VERDICT: {verdict} — {results['best_side']} {line} {stat_upper}")
        logger.info(f"  {edge_above:.1f}% above breakeven. Solid value at -110.")
        logger.info( "  Worth playing as part of a diversified card.")
    else:
        logger.info(f"  VERDICT: {verdict} — {results['best_side']} {line} {stat_upper}")
        logger.info(f"  {edge_above:.1f}% above breakeven. Strong mathematical case.")
        logger.info( "  High-confidence play at standard odds.")

    logger.info(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo prop simulation for NBA player stats"
    )
    parser.add_argument("--player",   required=True, type=str,
                        help="Player full name, e.g. 'Brandon Ingram'")
    parser.add_argument("--stat",     default="pts", type=str,
                        choices=list(STAT_MAP.keys()),
                        help="Stat to simulate: pts | reb | ast  (default: pts)")
    parser.add_argument("--line",     required=True, type=float,
                        help="Vegas prop line, e.g. 24.5")
    parser.add_argument("--opponent", default=None,  type=str,
                        help="Opponent abbreviation or full name (optional)")
    parser.add_argument("--min-gp",   default=MIN_GP_DEFAULT, type=int,
                        help=f"Min GP for baseline (default: {MIN_GP_DEFAULT})")
    parser.add_argument("--season",   default=None,  type=str,
                        help="Season string e.g. '2025-26' (default: current)")

    # --- 2K Simulation Context ---
    parser.add_argument(
        "--rest",
        type=int,
        default=None,
        metavar="DAYS",
        help=(
            "Days of rest before the game. "
            "0 = back-to-back (-5%% PTS), "
            "1 = normal (neutral), "
            "2+ = fresh legs (+2%% PTS)."
        ),
    )
    parser.add_argument(
        "--lockdown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "--lockdown: opponent's best defender is guarding this player (-5%% mean). "
            "--no-lockdown: favorable matchup, no lockdown coverage (+3%% mean). "
            "Omit for no adjustment."
        ),
    )

    args   = parser.parse_args()
    season = args.season or current_season()
    stat   = args.stat.lower()

    injured_out = get_injured_players()
    if normalize_player_name(args.player) in injured_out:
        logger.warning(f"[Late Scratch Guard] Skipping {args.player.strip()} - Ruled Out")
        sys.exit(1)

    logger.info(f"\nSeason: {season}  |  Stat: {stat.upper()}  |  Line: {args.line}  |  Simulations: {N_SIMS:,}")

    with Session(engine) as session:
        # Step 1 — full adjusted projection (baseline + NMU + defensive matchup)
        proj = project_player(
            session,
            args.player,
            season=season,
            min_gp=args.min_gp,
            opponent=args.opponent,
        )
        if proj is None:
            sys.exit(1)

        # Step 2 — historical std dev for the chosen stat
        std_result = get_stat_stddev(session, proj.player.player_id, stat, proj.season)

    if std_result is None:
        logger.warning(
            f"  Not enough game data to compute std dev for "
            f"{args.player!r} {stat.upper()} — cannot simulate."
        )
        sys.exit(1)

    std_dev, std_gp = std_result

    # Step 3 — 2K context adjustments (fatigue + lockdown)
    pre_mean = getattr(proj, STAT_MAP[stat][0])
    context  = apply_context_adjustments(pre_mean, stat, args.rest, args.lockdown)
    sim_mean = context.final_mean if context.has_any_adjustment else pre_mean

    # Step 4 — Monte Carlo simulation
    results = run_simulation(sim_mean, std_dev, args.line)

    # Step 5 — print report
    print_report(
        proj, stat, args.line, std_dev, std_gp, results,
        sim_mean=sim_mean,
        context=context if context.has_any_adjustment else None,
    )

# ---------------------------------------------------------------------------
# Single-player simulation (orchestration; XGB helpers live in parlay_builder)
# ---------------------------------------------------------------------------


def simulate_player(
    session,
    player_name: str,
    opponent_abbr: str | None,
    stat: str,
    season: str,
    live_odds: "dict[str, OddsLine] | None" = None,
    is_home: bool = False,
    opp_db_team_id: "int | None" = None,
    tomorrow_date_obj: "date | None" = None,
    n_sims: int | None = None,
) -> "PlayerSim | None":
    """
    Run the full projection + Monte Carlo for one player.
    If live_odds is provided, the prop line is taken from the bookmaker.
    Otherwise the player's raw baseline average is used as the line.
    EV is computed against actual book odds when available, else at -110.
    """
    from models import parlay_builder as _pb

    injured_out = get_injured_players_cached()
    if normalize_player_name(player_name) in injured_out:
        logger.warning(f"[Late Scratch Guard] Skipping {player_name} - Ruled Out")
        return None

    PlayerSim = _pb.PlayerSim
    _build_xgb_input = _pb._build_xgb_input
    NMU_VEGAS_KNOWS_THRESHOLD = _pb.NMU_VEGAS_KNOWS_THRESHOLD
    _xgb_models = _pb._xgb_models

    proj = project_player(
        session,
        player_name,
        season=season,
        min_gp=MIN_GP_DEFAULT,
        opponent=opponent_abbr,
    )
    if proj is None:
        return None

    std_result = get_stat_stddev(
        session, proj.player.player_id, stat, proj.season
    )
    if std_result is None:
        return None

    std_dev, gp = std_result

    pre_mean = getattr(proj, STAT_MAP[stat][0])   # final_{stat} from proj
    context  = apply_context_adjustments(pre_mean, stat, None, None)
    sim_mean = context.final_mean

    # ------------------------------------------------------------------
    # Determine the line: live book line (preferred) or baseline avg
    # ------------------------------------------------------------------
    odds_entry: OddsLine | None = None
    if live_odds:
        odds_entry = lookup_player_odds(proj.player.full_name, live_odds)

    if odds_entry is not None:
        line        = odds_entry.line
        line_source = odds_entry.bookmaker   # "DK" or "FD"
        over_odds   = odds_entry.over_odds
        under_odds  = odds_entry.under_odds
    else:
        line        = getattr(proj, f"baseline_{stat}")
        line_source = "EST"
        over_odds   = 0
        under_odds  = 0

    if stat in ("stl", "blk", "fg3"):
        std_dev = max(std_dev, 0.08)
    if line_source == "EST":
        # Quarantine estimated lines to avoid false confidence from narrow variance.
        std_dev *= 1.75

    if std_dev <= 0 or line <= 0:
        return None

    tomorrow_obj = tomorrow_date_obj or (date.today() + timedelta(days=1))
    spr_row = session.execute(
        text(
            """
            SELECT g.closing_spread, g.closing_total
            FROM games g
            WHERE g.game_date = :gd
              AND (g.home_team_id = :tid OR g.away_team_id = :tid)
            ORDER BY g.id
            LIMIT 1
            """
        ),
        {"gd": tomorrow_obj.isoformat(), "tid": proj.player.team_id},
    ).fetchone()
    cs = None
    ct = None
    if spr_row and spr_row.closing_spread is not None:
        cs = float(spr_row.closing_spread)
    if spr_row and getattr(spr_row, "closing_total", None) is not None:
        ct = float(spr_row.closing_total)
    blow_r = blowout_risk_from_spread(cs)
    st_scale = starter_mean_scale(
        blow_r,
        is_starter_for_team(session, proj.player.player_id, proj.player.team_id, season),
    )
    sim_mean = sim_mean * st_scale

    std_eff_h = effective_std_for_prop(std_dev, sim_mean, line)
    results = run_simulation(sim_mean, std_eff_h, line, n_sims=n_sims)

    # ------------------------------------------------------------------
    # EV: use actual book odds when available; fall back to -110 result
    # ------------------------------------------------------------------
    if odds_entry is not None:
        bet_odds   = over_odds if results["best_side"] == "OVER" else under_odds
        ev_per_110 = compute_ev_american(results["best_pct"], bet_odds, stake=110.0)
    else:
        ev_per_110 = results["ev_per_110"]

    # Capture the defensive multiplier delta for the chosen stat
    def_mult   = proj.matchup.multiplier if proj.matchup else 1.0
    bump_val   = getattr(proj, f"bump_{stat}")
    conf_boost = proj.confidence_boost

    # ------------------------------------------------------------------
    # Vegas Knows rule:
    # If there is no live odds line (EST) AND the NMU bump is large,
    # the model may be inflating a player whose line Vegas pulled because
    # the teammate injury is too fresh / the market is not yet open.
    # Override the verdict to warn the user.
    # ------------------------------------------------------------------
    verdict = results["verdict"]
    if stat == "pts" and line_source == "EST" and bump_val >= NMU_VEGAS_KNOWS_THRESHOLD:
        verdict = "NO VEGAS LINE (Likely OTB)"

    # ------------------------------------------------------------------
    # XGBoost ensemble prediction (per-stat model when available)
    # ------------------------------------------------------------------
    xgb_available  = False
    xgb_mean       = sim_mean      # safe fallback
    xgb_results    = results       # reuse heuristic dict as fallback
    xgb_ev_per_110 = ev_per_110

    if stat in _xgb_models:
        xgb_feats = _build_xgb_input(
            session, stat, proj.player.player_id, opp_db_team_id, is_home, tomorrow_obj
        )
        if xgb_feats is not None:
            xgb_mean = float(_xgb_models[stat].predict(xgb_feats)[0]) * st_scale
            std_eff_x = effective_std_for_prop(std_dev, xgb_mean, line)
            xgb_results  = run_simulation(xgb_mean, std_eff_x, line, n_sims=n_sims)
            xgb_available = True

            if odds_entry is not None:
                xgb_bet_odds   = (
                    over_odds if xgb_results["best_side"] == "OVER" else under_odds
                )
                xgb_ev_per_110 = compute_ev_american(
                    xgb_results["best_pct"], xgb_bet_odds, stake=110.0
                )
            else:
                xgb_ev_per_110 = xgb_results["ev_per_110"]

    # Ensemble lock: both models independently agree on the same side with edge
    ensemble_lock = (
        xgb_available
        and results["best_side"] == xgb_results["best_side"]
        and results["best_pct"] > BREAKEVEN_PCT
        and xgb_results["best_pct"] > BREAKEVEN_PCT
    )

    h_note = str(results.get("sim_note") or "")
    x_note = str(xgb_results.get("sim_note") or "") if xgb_available else ""
    if h_note and x_note and h_note != x_note:
        sim_note = f"{h_note}; {x_note}"
    else:
        sim_note = h_note or x_note

    return PlayerSim(
        player_name  = proj.player.full_name,
        team_abbr    = proj.player.team_abbr,
        opponent     = proj.matchup.abbreviation if proj.matchup else (opponent_abbr or ""),
        stat         = stat.upper(),
        line         = round(line, 1),
        final_mean   = round(sim_mean, 2),
        std_dev      = round(std_eff_h, 2),
        over_pct     = results["over_pct"],
        under_pct    = results["under_pct"],
        best_side    = results["best_side"],
        best_pct     = results["best_pct"],
        ev_per_110   = round(ev_per_110, 2),
        verdict      = verdict,
        bump_pts     = round(bump_val, 2),
        def_mult     = round(def_mult, 3),
        pace_mult    = round(proj.pace_multiplier, 3),
        conf_boost   = round(conf_boost * 100, 1),
        is_rust      = proj.is_rust,
        gp           = gp,
        line_source  = line_source,
        over_odds    = over_odds,
        under_odds   = under_odds,
        # XGBoost / ensemble
        xgb_mean       = round(xgb_mean, 2),
        xgb_over_pct   = xgb_results["over_pct"],
        xgb_under_pct  = xgb_results["under_pct"],
        xgb_best_side  = xgb_results["best_side"],
        xgb_best_pct   = xgb_results["best_pct"],
        xgb_ev_per_110 = round(xgb_ev_per_110, 2),
        xgb_available  = xgb_available,
        ensemble_lock  = ensemble_lock,
        sim_note       = sim_note,
        explanation_tags=generate_explanation_tags(
            ExplanationContext(
                proj=proj,
                stat=stat,
                closing_total=ct,
                closing_spread=cs,
            )
        ),
    )


if __name__ == "__main__":
    main()
