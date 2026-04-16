"""
models/player_projections.py
==============================
Calculates a baseline PRA projection for a player from season box-score
averages, then applies five sequential adjustments:

  1. Next Man Up  — redistribute injured teammates' usage proportionally
  2. Defensive Matchup  — scale by opponent's pts-allowed vs league avg
  3. Pace  — scale by expected game pace vs league avg pace
  4. Coach Confidence  — boost if L5 avg minutes > season avg by >10%
  5. Rust  — penalise -20% if returning from a recent Out listing

Each layer's delta is stored individually so the display can show the
full adjustment stack clearly.

Usage
-----
    python models/player_projections.py
    python models/player_projections.py --player "Brandon Ingram" --opponent BOS
    python models/player_projections.py --player "Myles Turner" --opponent WAS
"""

import argparse
import os
from dataclasses import dataclass, field
from datetime import date, timedelta

from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import Session

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

import logging
from core.logger import configure_app_logging

configure_app_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB engine (single source of truth in database.py)
# ---------------------------------------------------------------------------
from database import engine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUMP_CAP               = 0.40   # max +40% from Next Man Up (pre-dampening)
NMU_ABS_CAP            = 4.5    # absolute ceiling on NMU bump (pts/reb/ast)
NMU_ROLE_THRESH        = 12.0   # below this pts avg → full NMU scalar (1.0)
NMU_STAR_THRESH        = 25.0   # at or above this pts avg → min NMU scalar (0.2)
DEFENSE_MULTIPLIER_CAP = 0.20   # defensive multiplier clamped ±20%
INJURY_LOOKBACK        = 3      # days back to consider a report "current"
MIN_GP_DEFAULT         = 5

PACE_FILTER_MIN        = 80.0   # exclude obvious garbage pace values
PACE_FILTER_MAX        = 120.0
PACE_CAP               = 0.10   # pace multiplier clamped ±10%

CONFIDENCE_THRESHOLD   = 0.10   # L5 avg min must exceed season avg by >10%
CONFIDENCE_CAP         = 0.15   # cap the boost at +15%

RUST_PENALTY           = -0.20  # -20% for returning from injury
RUST_LOOKBACK_MAX      = 7      # days back to look for an Out report
RUST_LOOKBACK_RECENT   = 2      # days considered "still currently Out"


def current_season() -> str:
    today = date.today()
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    player_id:    int
    full_name:    str
    team_id:      int
    team_abbr:    str
    avg_pts:      float
    avg_reb:      float
    avg_ast:      float
    avg_stl:      float
    avg_blk:      float
    avg_fg3:      float
    avg_min:      float
    games_played: int


@dataclass
class InjuredTeammate:
    player_id:   int
    full_name:   str
    avg_pts:     float
    avg_reb:     float
    avg_ast:     float
    avg_stl:     float
    avg_blk:     float
    avg_fg3:     float
    avg_min:     float
    injury_type: str | None
    report_date: date


@dataclass
class DefensiveMatchup:
    team_id:      int
    full_name:    str
    abbreviation: str
    papg:         float
    league_avg:   float
    multiplier:   float
    games_played: int

    @property
    def vs_league(self) -> float:
        return round(self.papg - self.league_avg, 2)

    @property
    def label(self) -> str:
        if self.multiplier >= 1.05:
            return "weak defense (favourable)"
        elif self.multiplier <= 0.95:
            return "strong defense (tough)"
        return "average defense"


@dataclass
class ProjectionResult:
    player:       PlayerStats
    season:       str
    report_date:  date

    # Stage 1 inputs
    baseline_pts: float
    baseline_reb: float
    baseline_ast: float
    baseline_stl: float
    baseline_blk: float
    baseline_fg3: float

    # Next Man Up
    injured_out:  list[InjuredTeammate] = field(default_factory=list)
    bump_pts:     float = 0.0
    bump_reb:     float = 0.0
    bump_ast:     float = 0.0
    bump_stl:     float = 0.0
    bump_blk:     float = 0.0
    bump_fg3:     float = 0.0
    target_share: float = 0.0

    # Stage 2: Defensive matchup
    matchup:      DefensiveMatchup | None = None

    # Stage 3: Pace
    pace_multiplier:   float = 1.0
    player_team_pace:  float | None = None
    opponent_pace:     float | None = None
    league_avg_pace:   float | None = None
    pace_delta_pts:    float = 0.0
    pace_delta_reb:    float = 0.0
    pace_delta_ast:    float = 0.0
    pace_delta_stl:    float = 0.0
    pace_delta_blk:    float = 0.0
    pace_delta_fg3:    float = 0.0

    # Stage 4: Coach Confidence
    confidence_boost:      float = 0.0
    l5_avg_min:            float | None = None
    confidence_delta_pts:  float = 0.0
    confidence_delta_reb:  float = 0.0
    confidence_delta_ast:  float = 0.0
    confidence_delta_stl:  float = 0.0
    confidence_delta_blk:  float = 0.0
    confidence_delta_fg3:  float = 0.0

    # Stage 5: Rust
    is_rust:         bool  = False
    rust_delta_pts:  float = 0.0
    rust_delta_reb:  float = 0.0
    rust_delta_ast:  float = 0.0
    rust_delta_stl:  float = 0.0
    rust_delta_blk:  float = 0.0
    rust_delta_fg3:  float = 0.0

    # Computed totals (set in __post_init__)
    projected_pts: float = 0.0   # after NMU only
    projected_reb: float = 0.0
    projected_ast: float = 0.0
    projected_stl: float = 0.0
    projected_blk: float = 0.0
    projected_fg3: float = 0.0
    final_pts:     float = 0.0   # after all five stages
    final_reb:     float = 0.0
    final_ast:     float = 0.0
    final_stl:     float = 0.0
    final_blk:     float = 0.0
    final_fg3:     float = 0.0

    def __post_init__(self):
        # Stage 1 — Next Man Up
        self.projected_pts = round(self.baseline_pts + self.bump_pts, 4)
        self.projected_reb = round(self.baseline_reb + self.bump_reb, 4)
        self.projected_ast = round(self.baseline_ast + self.bump_ast, 4)
        self.projected_stl = round(self.baseline_stl + self.bump_stl, 4)
        self.projected_blk = round(self.baseline_blk + self.bump_blk, 4)
        self.projected_fg3 = round(self.baseline_fg3 + self.bump_fg3, 4)

        pts = self.projected_pts
        reb = self.projected_reb
        ast = self.projected_ast
        stl = self.projected_stl
        blk = self.projected_blk
        fg3 = self.projected_fg3

        # Stage 2 — Defensive matchup
        if self.matchup:
            m    = self.matchup.multiplier
            pts  = round(pts * m, 4)
            reb  = round(reb * m, 4)
            ast  = round(ast * m, 4)
            stl  = round(stl * m, 4)
            blk  = round(blk * m, 4)
            fg3  = round(fg3 * m, 4)

        # Stage 3 — Pace
        if self.pace_multiplier != 1.0:
            self.pace_delta_pts = round(pts * (self.pace_multiplier - 1.0), 2)
            self.pace_delta_reb = round(reb * (self.pace_multiplier - 1.0), 2)
            self.pace_delta_ast = round(ast * (self.pace_multiplier - 1.0), 2)
            self.pace_delta_stl = round(stl * (self.pace_multiplier - 1.0), 2)
            self.pace_delta_blk = round(blk * (self.pace_multiplier - 1.0), 2)
            self.pace_delta_fg3 = round(fg3 * (self.pace_multiplier - 1.0), 2)
            pts += self.pace_delta_pts
            reb += self.pace_delta_reb
            ast += self.pace_delta_ast
            stl += self.pace_delta_stl
            blk += self.pace_delta_blk
            fg3 += self.pace_delta_fg3

        # Stage 4 — Coach Confidence
        if self.confidence_boost > 0:
            self.confidence_delta_pts = round(pts * self.confidence_boost, 2)
            self.confidence_delta_reb = round(reb * self.confidence_boost, 2)
            self.confidence_delta_ast = round(ast * self.confidence_boost, 2)
            self.confidence_delta_stl = round(stl * self.confidence_boost, 2)
            self.confidence_delta_blk = round(blk * self.confidence_boost, 2)
            self.confidence_delta_fg3 = round(fg3 * self.confidence_boost, 2)
            pts += self.confidence_delta_pts
            reb += self.confidence_delta_reb
            ast += self.confidence_delta_ast
            stl += self.confidence_delta_stl
            blk += self.confidence_delta_blk
            fg3 += self.confidence_delta_fg3

        # Stage 5 — Rust
        if self.is_rust:
            self.rust_delta_pts = round(pts * RUST_PENALTY, 2)
            self.rust_delta_reb = round(reb * RUST_PENALTY, 2)
            self.rust_delta_ast = round(ast * RUST_PENALTY, 2)
            self.rust_delta_stl = round(stl * RUST_PENALTY, 2)
            self.rust_delta_blk = round(blk * RUST_PENALTY, 2)
            self.rust_delta_fg3 = round(fg3 * RUST_PENALTY, 2)
            pts += self.rust_delta_pts
            reb += self.rust_delta_reb
            ast += self.rust_delta_ast
            stl += self.rust_delta_stl
            blk += self.rust_delta_blk
            fg3 += self.rust_delta_fg3

        self.final_pts = round(max(0.0, pts), 2)
        self.final_reb = round(max(0.0, reb), 2)
        self.final_ast = round(max(0.0, ast), 2)
        self.final_stl = round(max(0.0, stl), 2)
        self.final_blk = round(max(0.0, blk), 2)
        self.final_fg3 = round(max(0.0, fg3), 2)

    def display(self):
        sep  = "-" * 62
        dot  = "." * 62
        sign = lambda v: f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

        logger.info(f"\n{sep}")
        logger.info(f"  {self.player.full_name}  ({self.player.team_abbr})   {self.season}")
        logger.info(sep)
        logger.info(f"  {'':36s}  {'PTS':>6}  {'REB':>6}  {'AST':>6}")
        logger.info(f"  {'Baseline (season avg)':36s}  "
              f"{self.baseline_pts:>6.1f}  {self.baseline_reb:>6.1f}  {self.baseline_ast:>6.1f}")
        logger.info(f"  {'GP used'} {self.player.games_played:<4d}  "
              f"{'avg min':>8s}  {self.player.avg_min:>5.1f}")

        # NMU
        if self.injured_out:
            logger.info(f"\n  Teammates Out ({len(self.injured_out)}):")
            for p in self.injured_out:
                inj = f"  [{p.injury_type}]" if p.injury_type else ""
                logger.info(f"    - {p.full_name:<26s}"
                      f"  {p.avg_pts:>5.1f}pts"
                      f"  {p.avg_reb:>5.1f}reb"
                      f"  {p.avg_ast:>5.1f}ast{inj}")
            logger.info(f"\n  Target usage share : {self.target_share * 100:.1f}%")
            logger.info(f"  Next Man Up bump   : "
                  f"{sign(self.bump_pts)}pts  "
                  f"{sign(self.bump_reb)}reb  "
                  f"{sign(self.bump_ast)}ast")
        else:
            logger.info("\n  No Out teammates — no Next Man Up adjustment.")

        logger.info(dot)
        logger.info(f"  {'Pre-matchup projection':36s}  "
              f"{self.projected_pts:>6.1f}  "
              f"{self.projected_reb:>6.1f}  "
              f"{self.projected_ast:>6.1f}")

        # Stage 2 — Defensive matchup
        if self.matchup:
            m = self.matchup
            def_d_pts = round(self.projected_pts * (m.multiplier - 1.0), 2)
            def_d_reb = round(self.projected_reb * (m.multiplier - 1.0), 2)
            def_d_ast = round(self.projected_ast * (m.multiplier - 1.0), 2)
            vs = sign(m.vs_league)
            logger.info(f"\n  Opponent  : {m.full_name} ({m.abbreviation})")
            logger.info(f"  Opp pts allowed/g : {m.papg:.1f}  "
                  f"({m.games_played} GP, {vs} vs league)  [{m.label}]")
            logger.info(f"  Def. multiplier   : {m.multiplier:.3f}")
            logger.info(f"  {'Defensive adj':36s}  "
                  f"{sign(def_d_pts):>6s}  "
                  f"{sign(def_d_reb):>6s}  "
                  f"{sign(def_d_ast):>6s}")

        # Stage 3 — Pace
        if self.player_team_pace is not None:
            pace_label = (f"team={self.player_team_pace:.1f}  "
                          f"opp={self.opponent_pace:.1f}  "
                          f"league={self.league_avg_pace:.1f}  "
                          f"mult={self.pace_multiplier:.3f}")
            logger.info(f"\n  Pace ({pace_label})")
            if self.pace_multiplier != 1.0:
                logger.info(f"  {'Pace adj':36s}  "
                      f"{sign(self.pace_delta_pts):>6s}  "
                      f"{sign(self.pace_delta_reb):>6s}  "
                      f"{sign(self.pace_delta_ast):>6s}")
            else:
                logger.info(f"  Pace adj : neutral (within 1% of league avg)")
        else:
            logger.info(f"\n  Pace : no data or no opponent specified")

        # Stage 4 — Coach Confidence
        if self.l5_avg_min is not None:
            diff_pct = round((self.l5_avg_min / self.player.avg_min - 1.0) * 100, 1) \
                       if self.player.avg_min else 0.0
            if self.confidence_boost > 0:
                logger.info(f"\n  Coach Confidence  : L5={self.l5_avg_min:.1f}min  "
                      f"season={self.player.avg_min:.1f}min  "
                      f"(+{diff_pct:.1f}%)  boost=+{self.confidence_boost * 100:.1f}%")
                logger.info(f"  {'Confidence adj':36s}  "
                      f"{sign(self.confidence_delta_pts):>6s}  "
                      f"{sign(self.confidence_delta_reb):>6s}  "
                      f"{sign(self.confidence_delta_ast):>6s}")
            else:
                logger.info(f"\n  Coach Confidence  : L5={self.l5_avg_min:.1f}min  "
                      f"season={self.player.avg_min:.1f}min  "
                      f"({diff_pct:+.1f}%) — threshold not met, no boost")
        else:
            logger.info(f"\n  Coach Confidence  : insufficient L5 data")

        # Stage 5 — Rust
        if self.is_rust:
            logger.info(f"\n  RUST DETECTED  : player returning from recent Out listing")
            logger.info(f"  Minute restriction penalty : {RUST_PENALTY * 100:.0f}%")
            logger.info(f"  {'Rust adj':36s}  "
                  f"{sign(self.rust_delta_pts):>6s}  "
                  f"{sign(self.rust_delta_reb):>6s}  "
                  f"{sign(self.rust_delta_ast):>6s}")
        else:
            logger.info(f"\n  Rust          : none detected")

        logger.info(dot)
        logger.info(f"  {'FINAL PROJECTION':36s}  "
              f"{self.final_pts:>6.1f}  "
              f"{self.final_reb:>6.1f}  "
              f"{self.final_ast:>6.1f}")
        logger.info(sep)


# ---------------------------------------------------------------------------
# DB — player / team helpers
# ---------------------------------------------------------------------------

def get_player_id(session: Session, name: str) -> tuple[int, int] | None:
    row = session.execute(
        text("SELECT id, team_id FROM players "
             "WHERE LOWER(full_name) = LOWER(:name) AND is_active = TRUE LIMIT 1"),
        {"name": name},
    ).fetchone()
    return (row.id, row.team_id) if row else None


def get_player_season_stats(
    session: Session,
    player_id: int,
    season: str,
    min_gp: int = MIN_GP_DEFAULT,
) -> PlayerStats | None:
    query = text("""
        SELECT p.id, p.full_name, p.team_id, t.abbreviation,
               ROUND(AVG(pbs.pts)::numeric, 2)            AS avg_pts,
               ROUND(AVG(pbs.reb)::numeric, 2)            AS avg_reb,
               ROUND(AVG(pbs.ast)::numeric, 2)            AS avg_ast,
               ROUND(AVG(pbs.stl)::numeric, 2)            AS avg_stl,
               ROUND(AVG(pbs.blk)::numeric, 2)            AS avg_blk,
               ROUND(AVG(pbs.fg3m)::numeric, 2)           AS avg_fg3,
               ROUND(AVG(pbs.minutes_played)::numeric, 2) AS avg_min,
               COUNT(*)                                    AS gp
        FROM player_box_scores_traditional pbs
        JOIN games   g ON pbs.game_id   = g.id
        JOIN players p ON pbs.player_id = p.id
        JOIN teams   t ON p.team_id     = t.id
        WHERE pbs.player_id  = :pid
          AND pbs.dnp_status = FALSE
          AND g.season       = :season
        GROUP BY p.id, p.full_name, p.team_id, t.abbreviation
    """)
    for s in [season, f"{int(season[:4])-1}-{season[:4][2:]}"]:
        row = session.execute(query, {"pid": player_id, "season": s}).fetchone()
        if row and row.gp >= min_gp:
            if s != season:
                logger.info(f"  [note] Using {s} averages (only {row.gp} GP in {season}).")
            return PlayerStats(
                player_id=row.id, full_name=row.full_name,
                team_id=row.team_id, team_abbr=row.abbreviation,
                avg_pts=float(row.avg_pts), avg_reb=float(row.avg_reb),
                avg_ast=float(row.avg_ast),
                avg_stl=float(row.avg_stl or 0), avg_blk=float(row.avg_blk or 0),
                avg_fg3=float(row.avg_fg3 or 0),
                avg_min=float(row.avg_min or 0),
                games_played=row.gp,
            )
    return None


def get_injured_out_teammates(
    session: Session,
    team_id: int,
    exclude_player_id: int,
    season: str,
    lookback_days: int = INJURY_LOOKBACK,
) -> list[InjuredTeammate]:
    cutoff = date.today() - timedelta(days=lookback_days)
    rows = session.execute(
        text("""
            SELECT DISTINCT ON (ir.player_id)
                   ir.player_id, ir.injury_type, ir.report_date
            FROM injury_reports ir
            WHERE ir.team_id    = :team_id
              AND ir.player_id != :exclude
              AND ir.is_out     = TRUE
              AND ir.report_date >= :cutoff
            ORDER BY ir.player_id, ir.report_date DESC
        """),
        {"team_id": team_id, "exclude": exclude_player_id, "cutoff": cutoff},
    ).fetchall()
    if not rows:
        return []

    injured: list[InjuredTeammate] = []
    for ir_row in rows:
        stats_row = session.execute(
            text("""
                SELECT p.full_name,
                       ROUND(AVG(pbs.pts)::numeric, 2)            AS avg_pts,
                       ROUND(AVG(pbs.reb)::numeric, 2)            AS avg_reb,
                       ROUND(AVG(pbs.ast)::numeric, 2)            AS avg_ast,
                       ROUND(AVG(pbs.stl)::numeric, 2)            AS avg_stl,
                       ROUND(AVG(pbs.blk)::numeric, 2)            AS avg_blk,
                       ROUND(AVG(pbs.fg3m)::numeric, 2)           AS avg_fg3,
                       ROUND(AVG(pbs.minutes_played)::numeric, 2) AS avg_min,
                       COUNT(*) AS gp
                FROM player_box_scores_traditional pbs
                JOIN players p ON pbs.player_id = p.id
                JOIN games   g ON pbs.game_id   = g.id
                WHERE pbs.player_id  = :pid
                  AND pbs.dnp_status = FALSE
                  AND g.season       = :season
                GROUP BY p.full_name
            """),
            {"pid": ir_row.player_id, "season": season},
        ).fetchone()
        if not stats_row or stats_row.gp < 3:
            continue
        injured.append(InjuredTeammate(
            player_id=ir_row.player_id,
            full_name=stats_row.full_name,
            avg_pts=float(stats_row.avg_pts),
            avg_reb=float(stats_row.avg_reb),
            avg_ast=float(stats_row.avg_ast),
            avg_stl=float(stats_row.avg_stl or 0),
            avg_blk=float(stats_row.avg_blk or 0),
            avg_fg3=float(stats_row.avg_fg3 or 0),
            avg_min=float(stats_row.avg_min or 0),
            injury_type=ir_row.injury_type,
            report_date=ir_row.report_date,
        ))
    return injured


def get_healthy_rotation_minutes(
    session: Session,
    team_id: int,
    exclude_ids: list[int],
    season: str,
    min_avg_min: float = 10.0,
) -> dict[int, float]:
    if not exclude_ids:
        exclude_ids = [-1]
    rows = session.execute(
        text("""
            SELECT pbs.player_id,
                   ROUND(AVG(pbs.minutes_played)::numeric, 2) AS avg_min
            FROM player_box_scores_traditional pbs
            JOIN games g ON pbs.game_id = g.id
            WHERE pbs.team_id     = :team_id
              AND pbs.player_id  != ALL(:exclude)
              AND pbs.dnp_status = FALSE
              AND g.season        = :season
            GROUP BY pbs.player_id
            HAVING AVG(pbs.minutes_played) >= :min_min
        """),
        {"team_id": team_id, "exclude": exclude_ids,
         "season": season, "min_min": min_avg_min},
    ).fetchall()
    return {r.player_id: float(r.avg_min) for r in rows}


# ---------------------------------------------------------------------------
# DB — defensive matchup
# ---------------------------------------------------------------------------

def resolve_opponent(session: Session, query: str) -> tuple[int, str, str] | None:
    q = query.strip()
    for sql in [
        "SELECT id, abbreviation, full_name FROM teams WHERE UPPER(abbreviation) = UPPER(:q) LIMIT 1",
        "SELECT id, abbreviation, full_name FROM teams WHERE LOWER(full_name) = LOWER(:q) LIMIT 1",
        "SELECT id, abbreviation, full_name FROM teams WHERE LOWER(full_name) LIKE LOWER(:q) LIMIT 1",
    ]:
        row = session.execute(text(sql), {"q": q if "LIKE" not in sql else f"%{q}%"}).fetchone()
        if row:
            return row.id, row.abbreviation, row.full_name
    return None


def get_opponent_papg(session: Session, team_id: int, season: str) -> tuple[float, int] | None:
    row = session.execute(
        text("""
            SELECT ROUND(AVG(opp_score)::numeric, 2) AS papg, COUNT(*) AS gp
            FROM (
                SELECT away_score AS opp_score FROM games
                WHERE home_team_id = :tid AND season = :season AND away_score IS NOT NULL
                UNION ALL
                SELECT home_score AS opp_score FROM games
                WHERE away_team_id = :tid AND season = :season AND home_score IS NOT NULL
            ) g
        """),
        {"tid": team_id, "season": season},
    ).fetchone()
    if not row or not row.papg:
        return None
    return float(row.papg), int(row.gp)


def get_league_avg_ppg(session: Session, season: str) -> float | None:
    row = session.execute(
        text("""
            SELECT ROUND(AVG(score)::numeric, 2) AS avg_ppg
            FROM (
                SELECT home_score AS score FROM games
                WHERE season = :season AND home_score IS NOT NULL
                UNION ALL
                SELECT away_score AS score FROM games
                WHERE season = :season AND away_score IS NOT NULL
            ) g
        """),
        {"season": season},
    ).fetchone()
    return float(row.avg_ppg) if row and row.avg_ppg else None


def build_defensive_matchup(
    session: Session, opponent_query: str, season: str
) -> DefensiveMatchup | None:
    resolved = resolve_opponent(session, opponent_query)
    if resolved is None:
        logger.warning(f"  [warning] Opponent not found: {opponent_query!r}")
        return None
    opp_id, opp_abbr, opp_name = resolved

    papg_result = get_opponent_papg(session, opp_id, season)
    if papg_result is None:
        logger.warning(f"  [warning] No game data for {opp_name} in {season}")
        return None
    papg, gp = papg_result

    league_avg = get_league_avg_ppg(session, season)
    if not league_avg:
        logger.warning("  [warning] Could not compute league avg PPG")
        return None

    raw_mult = papg / league_avg
    clamped  = max(1 - DEFENSE_MULTIPLIER_CAP, min(1 + DEFENSE_MULTIPLIER_CAP, raw_mult))
    return DefensiveMatchup(
        team_id=opp_id, full_name=opp_name, abbreviation=opp_abbr,
        papg=papg, league_avg=league_avg,
        multiplier=round(clamped, 4), games_played=gp,
    )


# ---------------------------------------------------------------------------
# DB — pace
# ---------------------------------------------------------------------------

def get_team_pace(
    session: Session, team_id: int, season: str
) -> tuple[float, int] | None:
    """
    Average game-level pace for a team this season.
    Groups by game_id first to avoid inflating with per-player rows,
    and filters out garbage values (pace outside 80-120).
    """
    row = session.execute(
        text("""
            SELECT ROUND(AVG(game_pace)::numeric, 2) AS avg_pace,
                   COUNT(*)                           AS games
            FROM (
                SELECT pba.game_id, AVG(pba.pace) AS game_pace
                FROM player_box_scores_advanced pba
                JOIN games g ON pba.game_id = g.id
                WHERE pba.team_id = :tid
                  AND pba.pace BETWEEN :lo AND :hi
                  AND g.season   = :season
                GROUP BY pba.game_id
            ) gp
        """),
        {"tid": team_id, "season": season,
         "lo": PACE_FILTER_MIN, "hi": PACE_FILTER_MAX},
    ).fetchone()
    if row and row.avg_pace and row.games >= 3:
        return float(row.avg_pace), int(row.games)
    return None


def get_league_avg_pace(session: Session, season: str) -> float | None:
    """League-wide average game pace (one value per game, filtered)."""
    row = session.execute(
        text("""
            SELECT ROUND(AVG(game_pace)::numeric, 2) AS avg_pace
            FROM (
                SELECT pba.game_id, AVG(pba.pace) AS game_pace
                FROM player_box_scores_advanced pba
                JOIN games g ON pba.game_id = g.id
                WHERE pba.pace BETWEEN :lo AND :hi
                  AND g.season = :season
                GROUP BY pba.game_id
            ) gp
        """),
        {"season": season, "lo": PACE_FILTER_MIN, "hi": PACE_FILTER_MAX},
    ).fetchone()
    return float(row.avg_pace) if row and row.avg_pace else None


# ---------------------------------------------------------------------------
# DB — coach confidence
# ---------------------------------------------------------------------------

def get_coach_confidence(
    session: Session, player_id: int, season: str, season_avg_min: float
) -> tuple[float | None, float]:
    """
    Returns (l5_avg_min, confidence_boost).
    boost > 0 only when L5 avg minutes exceeds season avg by > CONFIDENCE_THRESHOLD.
    """
    row = session.execute(
        text("""
            SELECT ROUND(AVG(minutes_played)::numeric, 2) AS l5_avg,
                   COUNT(*)                               AS games
            FROM (
                SELECT pbs.minutes_played
                FROM player_box_scores_traditional pbs
                JOIN games g ON pbs.game_id = g.id
                WHERE pbs.player_id        = :pid
                  AND pbs.dnp_status       = FALSE
                  AND pbs.minutes_played   IS NOT NULL
                  AND g.season             = :season
                ORDER BY g.game_date DESC
                LIMIT 5
            ) recent
        """),
        {"pid": player_id, "season": season},
    ).fetchone()

    if not row or not row.l5_avg or row.games < 3 or not season_avg_min:
        return None, 0.0

    l5_avg = float(row.l5_avg)
    ratio  = l5_avg / season_avg_min
    if ratio <= (1.0 + CONFIDENCE_THRESHOLD):
        return l5_avg, 0.0

    boost = min(ratio - 1.0, CONFIDENCE_CAP)
    return l5_avg, round(boost, 4)


# ---------------------------------------------------------------------------
# DB — rust detection
# ---------------------------------------------------------------------------

def check_rust_status(session: Session, player_id: int) -> bool:
    """
    Return True when the player was listed as Out between 3–7 days ago
    but has NO Out report in the last 2 days — i.e. returning from injury
    and likely on a minute restriction.
    """
    had_recent_out = session.execute(
        text("""
            SELECT COUNT(*) FROM injury_reports
            WHERE player_id   = :pid
              AND is_out       = TRUE
              AND report_date >= CURRENT_DATE - :max_days * INTERVAL '1 day'
              AND report_date <  CURRENT_DATE - :recent_days * INTERVAL '1 day'
        """),
        {"pid": player_id,
         "max_days": RUST_LOOKBACK_MAX,
         "recent_days": RUST_LOOKBACK_RECENT},
    ).scalar()

    if not had_recent_out:
        return False

    still_out = session.execute(
        text("""
            SELECT COUNT(*) FROM injury_reports
            WHERE player_id   = :pid
              AND is_out       = TRUE
              AND report_date >= CURRENT_DATE - :recent_days * INTERVAL '1 day'
        """),
        {"pid": player_id, "recent_days": RUST_LOOKBACK_RECENT},
    ).scalar()

    return still_out == 0


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------

def project_player(
    session: Session,
    player_name: str,
    season: str | None = None,
    min_gp: int = MIN_GP_DEFAULT,
    opponent: str | None = None,
) -> "ProjectionResult | None":
    if season is None:
        season = current_season()

    result = get_player_id(session, player_name)
    if result is None:
        logger.warning(f"  Player not found: {player_name!r}")
        return None
    player_id, team_id = result

    stats = get_player_season_stats(session, player_id, season, min_gp)
    if stats is None:
        logger.warning(f"  Not enough data for {player_name!r} (need >= {min_gp} GP).")
        return None

    # --- Stage 1: NMU ---
    injured      = get_injured_out_teammates(session, team_id, player_id, season)
    injured_ids  = [p.player_id for p in injured] + [player_id]
    healthy_mins = get_healthy_rotation_minutes(session, team_id, injured_ids, season)

    bump_pts = bump_reb = bump_ast = bump_stl = bump_blk = bump_fg3 = 0.0
    target_share = 0.0
    if injured and healthy_mins:
        total_min    = sum(healthy_mins.values()) + stats.avg_min
        target_share = stats.avg_min / total_min if total_min > 0 else 0.0

        raw_bump_pts = min(sum(p.avg_pts for p in injured) * target_share, stats.avg_pts * BUMP_CAP)
        raw_bump_reb = min(sum(p.avg_reb for p in injured) * target_share, stats.avg_reb * BUMP_CAP)
        raw_bump_ast = min(sum(p.avg_ast for p in injured) * target_share, stats.avg_ast * BUMP_CAP)
        raw_bump_stl = min(sum(p.avg_stl for p in injured) * target_share, max(stats.avg_stl, 0.01) * BUMP_CAP)
        raw_bump_blk = min(sum(p.avg_blk for p in injured) * target_share, max(stats.avg_blk, 0.01) * BUMP_CAP)
        raw_bump_fg3 = min(sum(p.avg_fg3 for p in injured) * target_share, max(stats.avg_fg3, 0.01) * BUMP_CAP)

        # Dampen NMU: scale inversely to player's pts baseline so role players
        # get the full bump while high-usage stars get far less (they're already
        # near max usage). Linear interpolation: 1.0 at ≤12 pts, 0.2 at ≥25 pts.
        _span   = NMU_STAR_THRESH - NMU_ROLE_THRESH          # 13.0
        _t      = max(0.0, min(1.0, (stats.avg_pts - NMU_ROLE_THRESH) / _span))
        nmu_scalar = 1.0 - _t * 0.8    # 1.0 → 0.2 as baseline rises

        bump_pts = min(raw_bump_pts * nmu_scalar, NMU_ABS_CAP)
        bump_reb = min(raw_bump_reb * nmu_scalar, NMU_ABS_CAP)
        bump_ast = min(raw_bump_ast * nmu_scalar, NMU_ABS_CAP)
        bump_stl = min(raw_bump_stl * nmu_scalar, NMU_ABS_CAP)
        bump_blk = min(raw_bump_blk * nmu_scalar, NMU_ABS_CAP)
        bump_fg3 = min(raw_bump_fg3 * nmu_scalar, NMU_ABS_CAP)

    # --- Stage 2: Defensive matchup ---
    matchup = build_defensive_matchup(session, opponent, season) if opponent else None

    # --- Stage 3: Pace ---
    pace_multiplier = 1.0
    player_team_pace = opponent_pace = league_avg_pace = None

    if matchup:
        league_pace_val  = get_league_avg_pace(session, season)
        team_pace_result = get_team_pace(session, team_id, season)
        opp_pace_result  = get_team_pace(session, matchup.team_id, season)

        if league_pace_val and team_pace_result and opp_pace_result:
            player_team_pace = team_pace_result[0]
            opponent_pace    = opp_pace_result[0]
            league_avg_pace  = league_pace_val

            expected = (player_team_pace + opponent_pace) / 2.0
            raw_mult = expected / league_avg_pace
            pace_multiplier = round(
                max(1 - PACE_CAP, min(1 + PACE_CAP, raw_mult)), 4
            )

    # --- Stage 4: Coach Confidence ---
    l5_avg_min, confidence_boost = get_coach_confidence(
        session, player_id, season, stats.avg_min
    )

    # --- Stage 5: Rust ---
    is_rust = check_rust_status(session, player_id)

    return ProjectionResult(
        player=stats, season=season, report_date=date.today(),
        baseline_pts=round(stats.avg_pts, 2),
        baseline_reb=round(stats.avg_reb, 2),
        baseline_ast=round(stats.avg_ast, 2),
        baseline_stl=round(stats.avg_stl, 2),
        baseline_blk=round(stats.avg_blk, 2),
        baseline_fg3=round(stats.avg_fg3, 2),
        injured_out=injured,
        bump_pts=round(bump_pts, 2),
        bump_reb=round(bump_reb, 2),
        bump_ast=round(bump_ast, 2),
        bump_stl=round(bump_stl, 2),
        bump_blk=round(bump_blk, 2),
        bump_fg3=round(bump_fg3, 2),
        target_share=round(target_share, 4),
        matchup=matchup,
        pace_multiplier=pace_multiplier,
        player_team_pace=player_team_pace,
        opponent_pace=opponent_pace,
        league_avg_pace=league_avg_pace,
        confidence_boost=confidence_boost,
        l5_avg_min=l5_avg_min,
        is_rust=is_rust,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NBA player PRA projection")
    parser.add_argument("--player",   type=str, default=None)
    parser.add_argument("--opponent", type=str, default=None,
                        help="Team abbreviation or full name")
    parser.add_argument("--min-gp",   type=int, default=MIN_GP_DEFAULT)
    parser.add_argument("--season",   type=str, default=None)
    args = parser.parse_args()

    season = args.season or current_season()
    logger.info(f"Season: {season}  |  Report date: {date.today()}")

    with Session(engine) as session:
        if args.player:
            proj = project_player(session, args.player, season, args.min_gp, args.opponent)
            if proj:
                proj.display()
        else:
            for player_name, opp in [
                ("Brandon Ingram", "Golden State Warriors"),
                ("Myles Turner",   "Washington Wizards"),
            ]:
                proj = project_player(session, player_name, season, args.min_gp, opponent=opp)
                if proj:
                    proj.display()


if __name__ == "__main__":
    main()
