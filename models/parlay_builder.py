"""
models/parlay_builder.py
=========================
Live-Data Parlay Builder for tomorrow's NBA slate.

Pipeline
--------
  1. Fetch today's and tomorrow's game slates from ESPN NBA scoreboard API.
  2. For each tomorrow game, check if either team is also playing today.
  3. If today's game is NOT finished (Scheduled/In Progress):
       -> Mark that game OTB (On The Board - Waiting for Final Buzzer) and skip.
  4. If today's game IS finished:
       -> Run update_recent_games.py via subprocess to sync box score into DB.
  5. Query the top 4 active scorers for each eligible team tomorrow.
  6. For each player, run the full 5-layer projection
       (Baseline -> Next Man Up -> Defense -> Pace -> Coach Confidence -> Rust)
       with the prop line set to the player's season baseline average.
  7. Run 10,000-trial Monte Carlo simulation for each player.
  8. Print a color-coded dashboard: Top 5 OVERs and Top 5 UNDERs ranked by EV.

Usage
-----
  python models/parlay_builder.py
  python models/parlay_builder.py --stat reb
  python models/parlay_builder.py --top 8
  python models/parlay_builder.py --no-color
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

# Force UTF-8 output so player names with accents (e.g. Vucevic, Doncic) don't
# crash the cp1252 Windows terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv
from sqlalchemy.orm import Session

_models_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = str(Path(__file__).resolve().parents[1])
load_dotenv(os.path.join(_repo_dir, ".env"))

import logging
from core.logger import configure_app_logging

configure_app_logging()
logger = logging.getLogger(__name__)

from models.player_projections import (
    ProjectionResult,
    project_player,
    engine,
    current_season,
    MIN_GP_DEFAULT,
)
from models.explanation_tags import ExplanationContext, generate_explanation_tags
from services.simulations import (
    get_stat_stddev,
    apply_context_adjustments,
    run_simulation,
    summarize_prop_from_samples,
    effective_std_for_prop,
    STAT_MAP,
    STAT_SQL_COLUMN,
    N_SIMS,
    BREAKEVEN_PCT,
    MC_STAGE1_N,
    MC_EARLY_STOP_HIGH_PCT,
    MC_EARLY_STOP_LOW_PCT,
    simulate_player,
)
from scripts.fetch_live_odds import (
    fetch_live_odds as _fetch_live_odds,
    lookup_player_odds,
    compute_ev_american,
    OddsLine,
)
from models.rotation_coaching_features import (
    blowout_risk_from_spread,
    fetch_rotation_features_live,
    is_starter_for_team,
    starter_mean_scale,
)

# ---------------------------------------------------------------------------
# XGBoost ensemble models per stat (loaded once at startup)
# ---------------------------------------------------------------------------
_XGB_SUPPORTED_STATS = ("pts", "reb", "ast", "stl", "blk", "fg3")
_XGB_MODEL_PATHS = {
    st: os.path.join(_models_dir, f"xgboost_{st}_model.json") for st in _XGB_SUPPORTED_STATS
}
_xgb_models: dict[str, "xgb.XGBRegressor"] = {}
_xgb_feature_names_by_stat: dict[str, list[str]] = {}


def _load_xgb_models() -> None:
    """Load trained XGBoost models for any stats that have files on disk."""
    global _xgb_models, _xgb_feature_names_by_stat
    _xgb_models = {}
    _xgb_feature_names_by_stat = {}
    for st, model_path in _XGB_MODEL_PATHS.items():
        if not os.path.exists(model_path):
            continue
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        feat_names = list(model.feature_names_in_)
        _xgb_models[st] = model
        _xgb_feature_names_by_stat[st] = feat_names
        ohe_count = sum(1 for f in feat_names if f.startswith("opp_"))
        logger.info(
            f"  [XGB:{st}] loaded ({len(feat_names)} features, "
            f"{ohe_count} opponent OHE columns)"
        )
    if not _xgb_models:
        logger.warning("  [XGB] No prop model files found — ML columns disabled.")


def _build_xgb_input(
    session: "Session",
    stat: str,
    player_id: int,
    opp_db_team_id: "int | None",
    is_home: bool,
    tomorrow: date,
) -> "pd.DataFrame | None":
    """
    Query last-5 game log and build the feature DataFrame the XGBoost
    model expects.  Returns None if insufficient data.
    """
    model = _xgb_models.get(stat)
    feature_names = _xgb_feature_names_by_stat.get(stat)
    if model is None or not feature_names:
        return None

    stat_col = STAT_SQL_COLUMN.get(stat, stat)
    if stat == "fg3":
        stat_col = "fg3m"
    rows = session.execute(
        text(f"""
            SELECT pbs.{stat_col} AS stat_value, pbs.minutes_played, g.game_date
            FROM player_box_scores_traditional pbs
            JOIN games g ON g.id = pbs.game_id
            WHERE pbs.player_id  = :pid
              AND pbs.dnp_status = FALSE
              AND pbs.minutes_played >= 5
            ORDER BY g.game_date DESC
            LIMIT 5
        """),
        {"pid": player_id},
    ).fetchall()

    if not rows:
        return None

    last_5_avg_stat = float(np.mean([r.stat_value for r in rows]))
    last_5_avg_min = float(np.mean([r.minutes_played for r in rows]))

    last_game_date = max(r.game_date for r in rows)
    if isinstance(last_game_date, str):
        from datetime import datetime as _dt
        last_game_date = _dt.strptime(last_game_date, "%Y-%m-%d").date()
    days_rest = float((tomorrow - last_game_date).days)

    # Build the full feature row using model's expected column order
    row: dict[str, float] = {name: 0.0 for name in feature_names}
    row[f"last_5_avg_{stat}"] = last_5_avg_stat
    row["last_5_avg_min"] = last_5_avg_min
    row["is_home"]        = float(is_home)
    row["days_rest"]      = days_rest

    # Optional game-context features (if model was trained with them).
    ctx = session.execute(
        text(
            """
            SELECT g.closing_spread, g.closing_total
            FROM players p
            JOIN games g
              ON g.game_date = :gd
             AND (g.home_team_id = p.team_id OR g.away_team_id = p.team_id)
            WHERE p.id = :pid
            LIMIT 1
            """
        ),
        {"pid": player_id, "gd": tomorrow.isoformat()},
    ).fetchone()
    if ctx:
        spread = float(ctx.closing_spread) if ctx.closing_spread is not None else 0.0
        total = float(ctx.closing_total) if ctx.closing_total is not None else 0.0
        row["closing_spread"] = spread
        row["closing_total"] = total
        row["is_potential_blowout"] = 1.0 if abs(spread) > 10.0 else 0.0

    if opp_db_team_id is not None:
        ohe_col = f"opp_{opp_db_team_id}"
        if ohe_col in row:
            row[ohe_col] = 1.0
        # If team is unknown (expansion/trade edge case), all OHE cols stay 0 —
        # model's handle_unknown="ignore" was trained for exactly this scenario.

    rot_keys = (
        "historical_minute_overlap",
        "usage_to_sub_ratio",
        "coach_id",
        "bench_reliance_factor",
    )
    if any(k in feature_names for k in rot_keys):
        rot = fetch_rotation_features_live(session, player_id, opp_db_team_id, tomorrow)
        for k, v in rot.items():
            if k in row:
                row[k] = float(v)

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[40m"


_color_enabled = True   # toggled by --no-color

def col(code: str, text: str) -> str:
    return f"{code}{text}{C.RESET}" if _color_enabled else text

def green(t):   return col(C.GREEN,   t)
def red(t):     return col(C.RED,     t)
def yellow(t):  return col(C.YELLOW,  t)
def cyan(t):    return col(C.CYAN,    t)
def magenta(t): return col(C.MAGENTA, t)
def bold(t):    return col(C.BOLD,    t)
def dim(t):     return col(C.DIM,     t)


# ---------------------------------------------------------------------------
# ESPN abbreviation → DB abbreviation normalization
# ESPN uses shorter/different codes for some teams.
# ---------------------------------------------------------------------------
ESPN_ABBR_TO_DB: dict[str, str] = {
    "SA":    "SAS",   # San Antonio Spurs
    "UTAH":  "UTA",   # Utah Jazz
    "GS":    "GSW",   # Golden State Warriors
    "NY":    "NYK",   # New York Knicks
    "NO":    "NOP",   # New Orleans Pelicans
    "WSH":   "WAS",   # Washington Wizards (ESPN sometimes uses WSH)
    "BKLYN": "BKN",   # Brooklyn Nets
}

def normalize_abbr(abbr: str) -> str:
    """Map an ESPN team abbreviation to the DB abbreviation."""
    return ESPN_ABBR_TO_DB.get(abbr.upper(), abbr.upper())


# ---------------------------------------------------------------------------
# ESPN API helpers
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


@dataclass
class ESPNGame:
    event_id:     str
    home_abbr:    str
    away_abbr:    str
    home_name:    str
    away_name:    str
    home_espn_id: str
    away_espn_id: str
    status_name:  str    # STATUS_SCHEDULED | STATUS_IN_PROGRESS | STATUS_FINAL
    status_state: str    # pre | in | post
    is_complete:  bool
    game_date:    str    # YYYY-MM-DD

    @property
    def espn_team_ids(self) -> set[str]:
        return {self.home_espn_id, self.away_espn_id}

    @property
    def home_db_abbr(self) -> str:
        return normalize_abbr(self.home_abbr)

    @property
    def away_db_abbr(self) -> str:
        return normalize_abbr(self.away_abbr)

    @property
    def label(self) -> str:
        return f"{self.away_abbr} @ {self.home_abbr}"


def fetch_scoreboard(date_str: str) -> list[ESPNGame]:
    """Fetch ESPN scoreboard for a given date (YYYYMMDD format)."""
    params = {"dates": date_str} if date_str else {}
    try:
        resp = requests.get(ESPN_SCOREBOARD, headers=HEADERS,
                            params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(
            "  [ESPN] Failed to fetch scoreboard for %s",
            date_str,
            exc_info=exc,
        )
        return []

    events = resp.json().get("events", [])
    games: list[ESPNGame] = []
    for ev in events:
        comp     = ev["competitions"][0]
        status   = comp["status"]["type"]
        teams    = {c["homeAway"]: c for c in comp["competitors"]}
        home     = teams.get("home", {})
        away     = teams.get("away", {})
        home_t   = home.get("team", {})
        away_t   = away.get("team", {})

        # Parse game date from ISO timestamp e.g. "2026-03-29T17:00Z"
        raw_date = ev.get("date", "")[:10]  # "2026-03-29"

        games.append(ESPNGame(
            event_id     = ev["id"],
            home_abbr    = home_t.get("abbreviation", ""),
            away_abbr    = away_t.get("abbreviation", ""),
            home_name    = home_t.get("displayName", ""),
            away_name    = away_t.get("displayName", ""),
            home_espn_id = str(home_t.get("id", "")),
            away_espn_id = str(away_t.get("id", "")),
            status_name  = status.get("name", ""),
            status_state = status.get("state", ""),
            is_complete  = bool(status.get("completed", False)),
            game_date    = raw_date,
        ))
    return games


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

from sqlalchemy import text   # noqa: E402  (after path setup)


def build_abbr_to_db_id(session: Session) -> dict[str, int]:
    """Maps uppercase team abbreviation -> DB team id."""
    rows = session.execute(
        text("SELECT id, abbreviation FROM teams")
    ).fetchall()
    return {r.abbreviation.upper(): r.id for r in rows}


def get_top_scorers(
    session: Session,
    team_db_id: int,
    season: str,
    stat: str = "pts",
    n: int = 4,
) -> list[tuple[int, str, float]]:
    """
    Return the top-n active scorers for a team this season.
    Each entry is (player_id, full_name, season_avg).
    Falls back to previous season if current has < 3 GP.
    """
    col = STAT_SQL_COLUMN.get(stat, stat)
    for s in [season, f"{int(season[:4])-1}-{season[:4][2:]}"]:
        rows = session.execute(
            text(f"""
                SELECT p.id, p.full_name,
                       ROUND(AVG(pbs.{col})::numeric, 2) AS avg_stat,
                       COUNT(*) AS gp
                FROM player_box_scores_traditional pbs
                JOIN players p ON pbs.player_id = p.id
                JOIN games   g ON pbs.game_id   = g.id
                WHERE p.team_id      = :tid
                  AND p.is_active    = TRUE
                  AND pbs.dnp_status = FALSE
                  AND g.season       = :season
                GROUP BY p.id, p.full_name
                HAVING COUNT(*) >= 5
                ORDER BY avg_stat DESC
                LIMIT :n
            """),
            {"tid": team_db_id, "season": s, "n": n},
        ).fetchall()
        if rows:
            return [(r.id, r.full_name, float(r.avg_stat)) for r in rows]
    return []


# ---------------------------------------------------------------------------
# Live trigger — run update_recent_games.py
# ---------------------------------------------------------------------------

def trigger_db_update(verbose: bool = True) -> bool:
    """
    Run update_recent_games.py via subprocess.
    Returns True if the process exited cleanly (rc=0).
    """
    script = os.path.join(_repo_dir, "update_recent_games.py")
    if not os.path.exists(script):
        logger.warning(f"  [trigger] update_recent_games.py not found at {script}")
        return False

    if verbose:
        logger.info(f"  {cyan('[DB SYNC]')} Running update_recent_games.py ...")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # let output stream to terminal
        text=True,
        cwd=_repo_dir,
    )
    success = result.returncode == 0
    if verbose:
        status = green("OK") if success else red(f"FAILED (rc={result.returncode})")
        logger.info(f"  {cyan('[DB SYNC]')} update_recent_games.py -> {status}")
    return success


# ---------------------------------------------------------------------------
# Simulation runner for a single player
# ---------------------------------------------------------------------------


@dataclass
class PlayerSim:
    player_name:  str
    team_abbr:    str
    opponent:     str
    stat:         str
    line:         float          # prop line (live if available, else baseline avg)
    final_mean:   float          # heuristic projected mean after all 5 layers
    std_dev:      float
    over_pct:     float
    under_pct:    float
    best_side:    str
    best_pct:     float
    ev_per_110:   float
    verdict:      str
    # extra context for display
    bump_pts:     float          # NMU delta
    def_mult:     float          # defensive multiplier (1.0 = none)
    pace_mult:    float
    conf_boost:   float
    is_rust:      bool
    gp:           int            # games used for std dev
    line_source:  str            # "DK" | "FD" | "EST"
    over_odds:    int            # American odds (0 if no live line)
    under_odds:   int            # American odds (0 if no live line)
    # --- XGBoost / Ensemble fields ---
    xgb_mean:       float = 0.0
    xgb_over_pct:   float = 0.0
    xgb_under_pct:  float = 0.0
    xgb_best_side:  str   = ""
    xgb_best_pct:   float = 0.0
    xgb_ev_per_110: float = 0.0
    xgb_available:  bool  = False
    ensemble_lock:  bool  = False
    sim_note:       str   = ""
    explanation_tags: list[str] = field(default_factory=list)


# Variance shares for joint draws (game noise + player night + stat idiosyncratic); sum = 1.
_JOINT_W_GAME = 0.12
_JOINT_W_PLAYER = 0.28
_JOINT_W_IDIO = 0.60


def _team_id_from_abbr(session: Session, abbr: str) -> int | None:
    row = session.execute(
        text("SELECT id FROM teams WHERE UPPER(TRIM(abbreviation)) = UPPER(TRIM(:abbr)) LIMIT 1"),
        {"abbr": abbr},
    ).fetchone()
    return int(row.id) if row and row.id is not None else None


def _topn_points_share(session: Session, team_id: int | None, season: str, n: int = 4) -> float:
    """
    Estimate fraction of team points scored by top-N players in completed games.
    Used to scale simulated top-scorer sums to realistic team totals.
    """
    n = max(1, min(int(n), 5))
    defaults = {1: 0.30, 2: 0.44, 3: 0.56, 4: 0.66, 5: 0.73}
    if team_id is None:
        return float(defaults[n])
    q = text(
        """
        WITH team_games AS (
            SELECT g.id AS game_id,
                   CASE WHEN g.home_team_id = :tid THEN g.home_score ELSE g.away_score END AS team_pts
            FROM games g
            WHERE g.season = :season
              AND (g.home_team_id = :tid OR g.away_team_id = :tid)
              AND g.home_score IS NOT NULL
              AND g.away_score IS NOT NULL
        ),
        topn AS (
            SELECT tg.game_id, SUM(x.pts) AS top_pts
            FROM team_games tg
            JOIN LATERAL (
                SELECT pbs.pts
                FROM player_box_scores_traditional pbs
                WHERE pbs.game_id = tg.game_id
                  AND pbs.team_id = :tid
                  AND pbs.dnp_status = FALSE
                ORDER BY pbs.pts DESC NULLS LAST
                LIMIT :n
            ) x ON TRUE
            GROUP BY tg.game_id
        )
        SELECT AVG(topn.top_pts / NULLIF(tg.team_pts, 0)) AS share
        FROM team_games tg
        JOIN topn ON topn.game_id = tg.game_id
        WHERE tg.team_pts > 0
        """
    )
    try:
        r = session.execute(q, {"tid": team_id, "season": season, "n": int(n)}).fetchone()
        share = float(r.share) if r and r.share is not None else float(defaults[n])
    except Exception:
        share = float(defaults[n])
    lo = {1: 0.20, 2: 0.32, 3: 0.45, 4: 0.56, 5: 0.63}[n]
    hi = {1: 0.45, 2: 0.60, 3: 0.70, 4: 0.78, 5: 0.84}[n]
    return float(min(hi, max(lo, share)))


def simulate_game_players_joint(
    session: Session,
    game: ESPNGame,
    players: list[tuple[str, str, str, bool, int | None]],
    season: str,
    live_odds_by_stat: dict[str, dict],
    n_sims: int,
) -> tuple[list[PlayerSim], dict[str, float | str | int]]:
    """
    One shared 10k-style draw per *game*: all listed players get the same game-level
    shock G, each player gets a player-night shock Zp, each (player, stat) gets
    independent noise. PTS/REB/AST/STL/BLK/FG3 lines are evaluated on the *same* index
    across stats for that player.

    Team win % uses the sum of simulated PTS for top scorers on each side (same draw
    index), so it is tied to the same joint simulation — not a separate model.
    """
    n = int(n_sims)
    if n <= 0:
        return [], {}

    gd = date.fromisoformat(game.game_date)
    spr = session.execute(
        text(
            """
            SELECT g.closing_spread, g.closing_total
            FROM games g
            JOIN teams th ON th.id = g.home_team_id
            JOIN teams ta ON ta.id = g.away_team_id
            WHERE g.game_date = :gd
              AND th.abbreviation = :ha
              AND ta.abbreviation = :aa
            LIMIT 1
            """
        ),
        {"gd": gd.isoformat(), "ha": game.home_db_abbr, "aa": game.away_db_abbr},
    ).fetchone()
    joint_cs = float(spr.closing_spread) if spr and spr.closing_spread is not None else None
    joint_ct = float(spr.closing_total) if spr and spr.closing_total is not None else None
    joint_br = blowout_risk_from_spread(joint_cs)

    a = float(np.sqrt(_JOINT_W_GAME))
    b = float(np.sqrt(_JOINT_W_PLAYER))
    c = float(np.sqrt(_JOINT_W_IDIO))

    G_game = np.random.randn(n)
    all_sims: list[PlayerSim] = []
    home_pts_acc = np.zeros(n)
    away_pts_acc = np.zeros(n)
    n_h_contrib = 0
    n_a_contrib = 0

    for player_name, team_abbr, opp_abbr, is_home, _opp_db_id in players:
        proj = project_player(
            session,
            player_name,
            season=season,
            min_gp=MIN_GP_DEFAULT,
            opponent=opp_abbr,
        )
        if proj is None:
            continue

        st_scale_joint = starter_mean_scale(
            joint_br,
            is_starter_for_team(session, proj.player.player_id, proj.player.team_id, season),
        )

        stat_payload: list[dict] = []
        for st in ("pts", "reb", "ast", "stl", "blk", "fg3"):
            std_result = get_stat_stddev(session, proj.player.player_id, st, proj.season)
            if std_result is None:
                continue
            std_dev, gp = std_result
            if st in ("stl", "blk", "fg3"):
                std_dev = max(std_dev, 0.08)

            pre_mean = getattr(proj, STAT_MAP[st][0])
            context = apply_context_adjustments(pre_mean, st, None, None)
            sim_mean = context.final_mean * st_scale_joint

            odds_dict = live_odds_by_stat.get(st)
            odds_entry = lookup_player_odds(proj.player.full_name, odds_dict) if odds_dict else None
            if odds_entry is not None:
                line = odds_entry.line
                line_source = odds_entry.bookmaker
                over_odds = odds_entry.over_odds
                under_odds = odds_entry.under_odds
            else:
                line = getattr(proj, f"baseline_{st}")
                line_source = "EST"
                over_odds = 0
                under_odds = 0

            if std_dev <= 0 or line <= 0:
                continue

            bump_val = getattr(proj, f"bump_{st}")
            if line_source == "EST":
                std_dev *= 1.75
            stat_payload.append(
                {
                    "st": st,
                    "sim_mean": sim_mean,
                    "std_dev": std_dev,
                    "line": line,
                    "line_source": line_source,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                    "bump_val": bump_val,
                    "gp": gp,
                }
            )

        if not stat_payload:
            continue

        Zp = np.random.randn(n)
        def_mult = proj.matchup.multiplier if proj.matchup else 1.0
        pace_mult = proj.pace_multiplier
        conf_boost = proj.confidence_boost
        is_rust = proj.is_rust
        opp_display = proj.matchup.abbreviation if proj.matchup else (opp_abbr or "")
        pts_draw: np.ndarray | None = None

        for sp in stat_payload:
            st = sp["st"]
            explain_tags = generate_explanation_tags(
                ExplanationContext(
                    proj=proj,
                    stat=st,
                    closing_total=joint_ct,
                    closing_spread=joint_cs,
                )
            )
            sim_mean = sp["sim_mean"]
            line = sp["line"]
            std_mult = effective_std_for_prop(sp["std_dev"], sim_mean, line)
            sim_note = ""

            if st == "pts":
                eps = np.random.randn(n)
                raw = sim_mean + std_mult * (a * G_game + b * Zp + c * eps)
                np.clip(raw, 0.0, None, out=raw)
                res_dict = summarize_prop_from_samples(raw, line)
            else:
                n1 = min(MC_STAGE1_N, n)
                eps1 = np.random.randn(n1)
                raw1 = sim_mean + std_mult * (a * G_game[:n1] + b * Zp[:n1] + c * eps1)
                np.clip(raw1, 0.0, None, out=raw1)
                res_dict = summarize_prop_from_samples(raw1, line)
                if n1 < n:
                    bp = float(res_dict["best_pct"])
                    if bp > MC_EARLY_STOP_HIGH_PCT or bp < MC_EARLY_STOP_LOW_PCT:
                        raw = raw1
                        sim_note = "Stopped at 2k"
                    else:
                        eps2 = np.random.randn(n - n1)
                        raw2 = sim_mean + std_mult * (
                            a * G_game[n1:] + b * Zp[n1:] + c * eps2
                        )
                        np.clip(raw2, 0.0, None, out=raw2)
                        raw = np.concatenate([raw1, raw2])
                        res_dict = summarize_prop_from_samples(raw, line)
                else:
                    raw = raw1

            if sp["line_source"] != "EST" and (sp["over_odds"] or sp["under_odds"]):
                bet_odds = sp["over_odds"] if res_dict["best_side"] == "OVER" else sp["under_odds"]
                ev_per_110 = compute_ev_american(res_dict["best_pct"], bet_odds, stake=110.0)
            else:
                ev_per_110 = res_dict["ev_per_110"]

            verdict = res_dict["verdict"]
            if st == "pts" and sp["line_source"] == "EST" and sp["bump_val"] >= NMU_VEGAS_KNOWS_THRESHOLD:
                verdict = "NO VEGAS LINE (Likely OTB)"

            if st == "pts":
                pts_draw = raw

            all_sims.append(
                PlayerSim(
                    player_name=proj.player.full_name,
                    team_abbr=proj.player.team_abbr,
                    opponent=opp_display,
                    stat=st.upper(),
                    line=round(sp["line"], 1),
                    final_mean=round(sp["sim_mean"], 2),
                    std_dev=round(sp["std_dev"], 2),
                    over_pct=res_dict["over_pct"],
                    under_pct=res_dict["under_pct"],
                    best_side=res_dict["best_side"],
                    best_pct=res_dict["best_pct"],
                    ev_per_110=round(ev_per_110, 2),
                    verdict=verdict,
                    bump_pts=round(sp["bump_val"], 2),
                    def_mult=round(def_mult, 3),
                    pace_mult=round(pace_mult, 3),
                    conf_boost=round(conf_boost * 100, 1),
                    is_rust=is_rust,
                    gp=sp["gp"],
                    line_source=sp["line_source"],
                    over_odds=sp["over_odds"],
                    under_odds=sp["under_odds"],
                    xgb_mean=round(sp["sim_mean"], 2),
                    xgb_over_pct=res_dict["over_pct"],
                    xgb_under_pct=res_dict["under_pct"],
                    xgb_best_side=res_dict["best_side"],
                    xgb_best_pct=res_dict["best_pct"],
                    xgb_ev_per_110=round(ev_per_110, 2),
                    xgb_available=False,
                    ensemble_lock=False,
                    sim_note=sim_note,
                    explanation_tags=list(explain_tags),
                )
            )

        if pts_draw is not None:
            if is_home:
                home_pts_acc += pts_draw
                n_h_contrib += 1
            else:
                away_pts_acc += pts_draw
                n_a_contrib += 1

    home_tid = _team_id_from_abbr(session, game.home_db_abbr)
    away_tid = _team_id_from_abbr(session, game.away_db_abbr)
    home_share = _topn_points_share(session, home_tid, season, n=n_h_contrib)
    away_share = _topn_points_share(session, away_tid, season, n=n_a_contrib)
    home_pts_team_sim = home_pts_acc / max(home_share, 1e-6)
    away_pts_team_sim = away_pts_acc / max(away_share, 1e-6)

    market_home_pts = market_away_pts = None
    if joint_ct is not None:
        spr_val = float(joint_cs or 0.0)
        market_home_pts = (float(joint_ct) - spr_val) / 2.0
        market_away_pts = (float(joint_ct) + spr_val) / 2.0

    if n_h_contrib > 0 and n_a_contrib > 0:
        home_wins = home_pts_team_sim > away_pts_team_sim
        p_home = float(home_wins.mean() * 100.0)
        p_away = round(100.0 - p_home, 2)
        p_home = round(p_home, 2)
        sim_home_mean = float(np.mean(home_pts_team_sim))
        sim_away_mean = float(np.mean(away_pts_team_sim))
        if market_home_pts is not None and market_away_pts is not None:
            if n_h_contrib < 3 or n_a_contrib < 3:
                # Too few scorers represented: trust market-implied team points.
                home_pts_mean = float(market_home_pts)
                away_pts_mean = float(market_away_pts)
            else:
                # Blend simulation and market anchor to avoid unrealistic tails.
                home_pts_mean = 0.65 * sim_home_mean + 0.35 * float(market_home_pts)
                away_pts_mean = 0.65 * sim_away_mean + 0.35 * float(market_away_pts)
        else:
            home_pts_mean = sim_home_mean
            away_pts_mean = sim_away_mean
        projected_margin = home_pts_mean - away_pts_mean
        projected_total = home_pts_mean + away_pts_mean
    else:
        p_home = p_away = 50.0
        if market_home_pts is not None and market_away_pts is not None:
            home_pts_mean = float(market_home_pts)
            away_pts_mean = float(market_away_pts)
            projected_margin = home_pts_mean - away_pts_mean
            projected_total = home_pts_mean + away_pts_mean
        else:
            home_pts_mean = away_pts_mean = 112.0
            projected_margin = 0.0
            projected_total = 224.0

    # Guardrails: team totals from partial player pools can drift unrealistically.
    # If projected total is outside plausible NBA range, anchor to market total when
    # available, otherwise to a neutral league baseline.
    if projected_total < 170.0 or projected_total > 260.0:
        anchor_total = float(joint_ct) if joint_ct is not None else 224.0
        if joint_cs is not None:
            # home expected margin from spread sign convention.
            anchor_margin = -float(joint_cs)
        else:
            anchor_margin = float(np.clip(projected_margin, -18.0, 18.0))
        home_pts_mean = (anchor_total + anchor_margin) / 2.0
        away_pts_mean = (anchor_total - anchor_margin) / 2.0
        projected_margin = home_pts_mean - away_pts_mean
        projected_total = home_pts_mean + away_pts_mean

    home_cover_by = away_cover_by = 0.0
    p_home_cover = p_away_cover = 50.0
    if joint_cs is not None:
        home_cover_by = projected_margin + float(joint_cs)
        away_cover_by = -home_cover_by
        if n_h_contrib > 0 and n_a_contrib > 0:
            home_cover_mask = (home_pts_team_sim - away_pts_team_sim + float(joint_cs)) > 0.0
            p_home_cover = float(home_cover_mask.mean() * 100.0)
            p_away_cover = float(100.0 - p_home_cover)

    summary: dict[str, float | str | int] = {
        "matchup": game.label,
        "home_abbr": game.home_abbr,
        "away_abbr": game.away_abbr,
        "home_db_abbr": game.home_db_abbr,
        "away_db_abbr": game.away_db_abbr,
        "p_home_win": p_home,
        "p_away_win": p_away,
        "home_pts_players": n_h_contrib,
        "away_pts_players": n_a_contrib,
        "home_proj_pts": round(home_pts_mean, 2),
        "away_proj_pts": round(away_pts_mean, 2),
        "projected_total": round(projected_total, 2),
        "closing_total": round(joint_ct, 2) if joint_ct is not None else None,
        "closing_spread": round(joint_cs, 2) if joint_cs is not None else None,
        "projected_margin": round(projected_margin, 2),
        "home_cover_by": round(home_cover_by, 2),
        "away_cover_by": round(away_cover_by, 2),
        "p_home_cover": round(p_home_cover, 2),
        "p_away_cover": round(p_away_cover, 2),
        "home_topn_share": round(home_share, 3),
        "away_topn_share": round(away_share, 3),
    }
    return all_sims, summary


# ---------------------------------------------------------------------------
# Dashboard printer
# ---------------------------------------------------------------------------

# NMU bump threshold above which "Vegas Knows" rule triggers (pts)
NMU_VEGAS_KNOWS_THRESHOLD = 3.0

VERDICT_COLORS = {
    "VERY STRONG EDGE":          C.GREEN  + C.BOLD,
    "STRONG EDGE":               C.GREEN,
    "MODERATE EDGE":             C.YELLOW,
    "SLIGHT LEAN":               C.CYAN,
    "NO EDGE":                   C.DIM,
    "NO VEGAS LINE (Likely OTB)": C.MAGENTA + C.BOLD,
}


def _verdict_col(verdict: str, text: str) -> str:
    code = VERDICT_COLORS.get(verdict, "")
    return col(code, text) if _color_enabled else text


def _edge_bar(pct: float, width: int = 20) -> str:
    filled = int((pct - 50) / 50 * width)  # 50% = empty, 100% = full
    filled = max(0, min(width, filled))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    if pct >= 60:
        return green(bar)
    elif pct >= 55:
        return yellow(bar)
    elif pct >= 52.38:
        return cyan(bar)
    else:
        return dim(bar)


def print_dashboard(
    overs:  list[PlayerSim],
    unders: list[PlayerSim],
    otb:    list[str],
    skipped_names: list[str],
    top_n:  int,
    stat:   str,
    tomorrow_date: str,
) -> None:
    # Detect whether any sim in this run has XGB available (pts mode)
    xgb_on = any(s.xgb_available for s in overs + unders)

    sep  = "=" * (110 if xgb_on else 74)
    dash = "-" * (110 if xgb_on else 74)
    dot  = "." * (110 if xgb_on else 74)

    logger.info(f"\n{bold(sep)}")
    logger.info(bold(f"  ParlayPal  --  NBA Prop Dashboard  --  {tomorrow_date}"))
    ml_note = "  |  Ensemble ML active" if xgb_on else ""
    logger.info(bold(
        f"  Stat: {stat.upper()}  |  {N_SIMS:,} simulations"
        f"  |  Line = Live (DK/FD) or baseline (EST){ml_note}"
    ))
    logger.info(bold(sep))

    if otb:
        logger.info(f"\n  {yellow(bold('OTB  (Waiting for Final Buzzer tonight)'))}:")
        for g in otb:
            logger.info(f"    {yellow('>>>')} {g}")

    # ---- Column header ----
    if xgb_on:
        hdr = (
            f"  {'#':>2}  {'Player':<22}  {'Matchup':>10}  "
            f"{'Line':<12}  {'H-Mean':>6}  {'H-Win%':>6}  {'H-EV':>7}  "
            f"{'ML-Mean':>7}  {'ML-Win%':>7}  {'ML-EV':>7}  "
            f"{'Verdict':<22}"
        )
    else:
        hdr = (
            f"  {'#':>2}  {'Player':<22}  {'Matchup':>10}  "
            f"{'Line':<12}  {'Mean':>5}  {'Win%':>6}  {'EV':>7}  {'Verdict':<20}  Adjustments"
        )

    logger.info(f"\n{bold(dash)}")
    logger.info(bold(f"  TOP {top_n} OVERs  --  {stat.upper()}  (ranked by Win Prob)"))
    logger.info(bold(dash))
    logger.info(dim(hdr))
    logger.info(dim(dot))

    for i, s in enumerate(overs[:top_n], 1):
        _print_sim_row(i, s, "OVER", xgb_on)

    logger.info(f"\n{bold(dash)}")
    logger.info(bold(f"  TOP {top_n} UNDERs  --  {stat.upper()}  (ranked by Win Prob)"))
    logger.info(bold(dash))
    logger.info(dim(hdr))
    logger.info(dim(dot))

    for i, s in enumerate(unders[:top_n], 1):
        _print_sim_row(i, s, "UNDER", xgb_on)

    logger.info(f"\n{bold(sep)}")

    # ---- Legend ----
    logger.info(dim("  ADJUSTMENTS KEY:"))
    logger.info(dim("    NMU=Next Man Up bump  DEF=defensive mult  PACE=pace mult"))
    logger.info(dim("    CC=coach confidence boost  [RUST]=rust penalty active"))
    if xgb_on:
        logger.info(dim("    H-=Heuristic (5-layer)  ML-=XGBoost model  "
                  "ENSEMBLE LOCK = both models agree on edge"))
    logger.info(bold(sep))

    if skipped_names:
        logger.info(dim(f"\n  Skipped ({len(skipped_names)} — insufficient data):"))
        logger.info(dim("    " + ", ".join(skipped_names)))

    logger.info("")


def _adj_tags(s: PlayerSim) -> str:
    tags = []
    if s.bump_pts != 0:
        sign = "+" if s.bump_pts >= 0 else ""
        tags.append(f"NMU:{sign}{s.bump_pts}")
    if s.def_mult != 1.0:
        tags.append(f"DEF:{s.def_mult:.3f}")
    if s.pace_mult != 1.0:
        tags.append(f"PACE:{s.pace_mult:.3f}")
    if s.conf_boost > 0:
        tags.append(f"CC:+{s.conf_boost:.1f}%")
    if s.is_rust:
        tags.append("[RUST]")
    return "  " + "  ".join(tags) if tags else "  --"


def _fmt_pct(pct: float) -> str:
    s = f"{pct:.1f}%"
    if pct >= 60:    return green(s)
    if pct >= 55:    return yellow(s)
    if pct >= 52.38: return cyan(s)
    return dim(s)


def _fmt_ev(ev: float) -> str:
    sign = "+" if ev >= 0 else ""
    s = f"{sign}{ev:.2f}"
    if ev >= 5:  return green(s)
    if ev >= 0:  return yellow(s)
    return red(s)


def _print_sim_row(rank: int, s: PlayerSim, side: str, xgb_on: bool = False) -> None:
    matchup = f"{s.team_abbr} vs {s.opponent}" if s.opponent else s.team_abbr

    # Line label: "O 24.5 DK", "U 7.5 FD", or "O 22.0(EST)"
    if s.line_source == "EST":
        line_label = f"{s.line:.1f}(EST)"
    else:
        line_label = f"{s.line:.1f} {s.line_source}"
    side_col = green(f"O {line_label}") if side == "OVER" else red(f"U {line_label}")

    if xgb_on:
        # ------ Ensemble / dual-model layout ------
        # Ensemble lock flag — printed in verdict column
        if s.ensemble_lock:
            lock_col  = magenta(bold("ENSEMBLE LOCK"))
            verdict_str = f"{lock_col:<22}"
        else:
            verdict_str = _verdict_col(s.verdict, f"{s.verdict:<22}")

        ml_mean_str = f"{s.xgb_mean:>7.1f}" if s.xgb_available else dim(f"{'N/A':>7}")
        ml_pct_str  = _fmt_pct(s.xgb_best_pct) if s.xgb_available else dim(f"{'N/A':>7}")
        ml_ev_str   = _fmt_ev(s.xgb_ev_per_110) if s.xgb_available else dim(f"{'N/A':>7}")

        adj_str = dim(_adj_tags(s))

        logger.info(
            f"  {rank:>2}. {s.player_name:<22}  {matchup:>10}  "
            f"{side_col}  {s.final_mean:>6.1f}  "
            f"{_fmt_pct(s.best_pct):>6}  {_fmt_ev(s.ev_per_110):>7}  "
            f"{ml_mean_str}  {ml_pct_str:>7}  {ml_ev_str:>7}  "
            f"{verdict_str}"
            f"{adj_str}"
        )
    else:
        # ------ Legacy single-model layout ------
        verdict_str = _verdict_col(s.verdict, f"{s.verdict:<20}")
        adj_str = dim(_adj_tags(s))
        logger.info(
            f"  {rank:>2}. {s.player_name:<22}  {matchup:>10}  "
            f"{side_col}  {s.final_mean:>5.1f}  "
            f"{_fmt_pct(s.best_pct):>6}  {_fmt_ev(s.ev_per_110):>7}  {verdict_str}"
            f"{adj_str}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _color_enabled

    parser = argparse.ArgumentParser(
        description="Live-data parlay builder for tomorrow's NBA slate"
    )
    parser.add_argument(
        "--stat", default="pts", choices=list(STAT_MAP.keys()),
        help="Stat to project and simulate (default: pts)"
    )
    parser.add_argument(
        "--top", default=5, type=int,
        help="Number of top plays to show per side (default: 5)"
    )
    parser.add_argument(
        "--min-gp", default=MIN_GP_DEFAULT, type=int,
        help=f"Min GP required for a player to be included (default: {MIN_GP_DEFAULT})"
    )
    parser.add_argument(
        "--scorers", default=4, type=int,
        help="Top-N scorers to analyze per team (default: 4)"
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output (use in environments that don't support it)"
    )
    args = parser.parse_args()

    _color_enabled = not args.no_color
    stat   = args.stat.lower()
    season = current_season()

    # Load XGBoost models (per-stat; gracefully skipped when files are missing)
    _load_xgb_models()

    # ------------------------------------------------------------------
    # Load manual exclude list (data/exclude.txt)
    # ------------------------------------------------------------------
    _exclude_path = os.path.join(_repo_dir, "data", "exclude.txt")
    exclude_set: set[str] = set()
    if os.path.exists(_exclude_path):
        with open(_exclude_path, encoding="utf-8") as _ef:
            for _line in _ef:
                name = _line.strip()
                if name and not name.startswith("#"):
                    exclude_set.add(name.lower())
        if exclude_set:
            logger.info(f"  {yellow('Exclude list')} loaded: "
                  f"{len(exclude_set)} player(s) will be skipped.")

    today_str    = date.today().strftime("%Y%m%d")
    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    tomorrow_display = (date.today() + timedelta(days=1)).strftime("%B %d, %Y")

    logger.info(f"\n{bold('ParlayPal  --  Live Data Parlay Builder')}")
    logger.info(f"  Season : {season}")
    logger.info(f"  Date   : {tomorrow_display}")
    logger.info(f"  Stat   : {stat.upper()}")
    logger.info("")

    # ------------------------------------------------------------------
    # 1. Fetch ESPN slates
    # ------------------------------------------------------------------
    logger.info(f"  Fetching ESPN scoreboard for today ({today_str}) ...")
    today_games    = fetch_scoreboard(today_str)
    logger.info(f"  Fetching ESPN scoreboard for tomorrow ({tomorrow_str}) ...")
    tomorrow_games = fetch_scoreboard(tomorrow_str)

    logger.info(f"\n  Today    : {len(today_games)} game(s)  |  "
          f"Tomorrow : {len(tomorrow_games)} game(s)")

    if not tomorrow_games:
        logger.info("  No games found for tomorrow. Nothing to do.")
        return

    # ESPN team ID -> today's game (for back-to-back check)
    today_by_espn_id: dict[str, ESPNGame] = {}
    for g in today_games:
        today_by_espn_id[g.home_espn_id] = g
        today_by_espn_id[g.away_espn_id] = g

    # ------------------------------------------------------------------
    # 2. Back-to-back / OTB check + optional DB sync
    # ------------------------------------------------------------------
    otb_labels:       list[str]  = []   # games to skip
    triggered_update: bool       = False
    eligible_games:   list[ESPNGame] = []

    for g in tomorrow_games:
        blocking_today: list[ESPNGame] = []
        for eid in (g.home_espn_id, g.away_espn_id):
            tg = today_by_espn_id.get(eid)
            if tg:
                blocking_today.append(tg)

        if not blocking_today:
            eligible_games.append(g)
            continue

        # Check if ALL blocking games are finished
        all_done = all(tg.is_complete for tg in blocking_today)
        any_live = any(
            tg.status_state in ("pre", "in") for tg in blocking_today
        )

        if any_live:
            blocking_strs = [
                f"{tg.label} [{tg.status_name}]" for tg in blocking_today
            ]
            otb_labels.append(
                f"{g.label}  --  blocking: {', '.join(blocking_strs)}"
            )
            logger.info(f"  {yellow('OTB')}  {g.label}  (back-to-back team not done yet)")
        else:
            # All today games are finished — trigger DB update once
            if not triggered_update:
                logger.info(f"\n  {cyan('[TRIGGER]')} Back-to-back team(s) finished today.")
                triggered_update = trigger_db_update(verbose=True)
                logger.info("")
            eligible_games.append(g)

    logger.info(f"\n  Eligible games : {len(eligible_games)}  |  OTB : {len(otb_labels)}")

    if not eligible_games:
        logger.info("  No eligible games to simulate. Exiting.")
        _print_otb_only(otb_labels, tomorrow_display)
        return

    # ------------------------------------------------------------------
    # 3. Build team lookup and collect players to simulate
    # ------------------------------------------------------------------
    with Session(engine) as session:
        abbr_to_db = build_abbr_to_db_id(session)

        # Collect (player_name, team_abbr, opp_abbr, is_home, opp_db_id) tuples
        to_simulate: list[tuple[str, str, str, bool, "int | None"]] = []
        no_db_teams: list[str] = []

        for g in eligible_games:
            matchup_pairs = [
                (g.home_db_abbr, g.away_db_abbr, True),   # home team vs away opponent
                (g.away_db_abbr, g.home_db_abbr, False),  # away team vs home opponent
            ]
            for team_abbr, opp_abbr, is_home in matchup_pairs:
                db_id     = abbr_to_db.get(team_abbr.upper())
                opp_db_id = abbr_to_db.get(opp_abbr.upper())
                if db_id is None:
                    if team_abbr not in no_db_teams:
                        no_db_teams.append(team_abbr)
                        logger.warning(f"  [warn] Team {team_abbr!r} not found in DB — skipping.")
                    continue

                scorers = get_top_scorers(
                    session, db_id, season, stat, n=args.scorers
                )
                if not scorers:
                    logger.warning(f"  [warn] No scorer data for {team_abbr} in {season} — skipping.")
                    continue

                for _, name, _ in scorers:
                    if name.lower() in exclude_set:
                        logger.info(f"  {dim('[EXCL]')}  {name} — in exclude list, skipping.")
                        continue
                    to_simulate.append((name, team_abbr, opp_abbr, is_home, opp_db_id))

        logger.info(f"\n  Players to simulate : {len(to_simulate)}")
        if not to_simulate:
            logger.info("  Nothing to simulate. Check DB data.")
            return

        # ------------------------------------------------------------------
        # 4. Fetch live odds (once, before sim loop)
        # ------------------------------------------------------------------
        logger.info(f"\n  Fetching live {stat.upper()} odds from The Odds API ...")
        live_odds = _fetch_live_odds(stat)
        if live_odds:
            logger.info(f"  {green(str(len(live_odds)))} live player line(s) loaded "
                  f"(DK priority).")
        else:
            logger.warning(f"  {yellow('No live lines found')} — using baseline estimates (EST).")

        # ------------------------------------------------------------------
        # 5. Run projections + simulations
        # ------------------------------------------------------------------
        logger.info(f"\n  Running {N_SIMS:,}-trial Monte Carlo for each player ...")
        logger.info(dim("  (Projection layers: Baseline -> NMU -> Defense -> Pace -> Confidence -> Rust)\n"))

        sims:     list[PlayerSim] = []
        skipped:  list[str]       = []
        tomorrow_obj = date.today() + timedelta(days=1)

        for player_name, team_abbr, opp_abbr, is_home, opp_db_id in to_simulate:
            result = simulate_player(
                session, player_name, opp_abbr, stat, season,
                live_odds=live_odds,
                is_home=is_home,
                opp_db_team_id=opp_db_id,
                tomorrow_date_obj=tomorrow_obj,
            )
            if result is None:
                skipped.append(f"{player_name} ({team_abbr})")
                logger.info(f"    {dim('SKIP')}  {player_name} — insufficient data")
            else:
                verdict_display = _verdict_col(result.verdict, result.verdict)
                side_display = (green if result.best_side == "OVER" else red)(result.best_side)
                src_tag = dim(f"[{result.line_source}]")
                lock_tag = magenta(" [LOCK]") if result.ensemble_lock else ""
                ml_tag   = (
                    dim(f"  ML={result.xgb_mean:<5.1f}")
                    if result.xgb_available else ""
                )
                logger.info(
                    f"    {cyan('SIM')}  {result.player_name:<24}  "
                    f"vs {opp_abbr or '---':<5}  "
                    f"line={result.line:<5.1f}{src_tag}  "
                    f"H={result.final_mean:<5.1f}{ml_tag}  "
                    f"{side_display} {result.best_pct:.1f}%  "
                    f"EV={result.ev_per_110:+.2f}  "
                    f"{verdict_display}{lock_tag}"
                )
                sims.append(result)

    if not sims:
        logger.info("\n  No simulations completed. Exiting.")
        return

    # ------------------------------------------------------------------
    # 6. Rank and display dashboard
    # ------------------------------------------------------------------
    overs  = sorted(
        [s for s in sims if s.best_side == "OVER"],
        key=lambda x: (x.best_pct, x.ev_per_110), reverse=True
    )
    unders = sorted(
        [s for s in sims if s.best_side == "UNDER"],
        key=lambda x: (x.best_pct, x.ev_per_110), reverse=True
    )

    print_dashboard(
        overs, unders, otb_labels, skipped,
        top_n=args.top,
        stat=stat,
        tomorrow_date=tomorrow_display,
    )


def _print_otb_only(otb_labels: list[str], tomorrow_display: str) -> None:
    sep = "=" * 74
    logger.info(f"\n{bold(sep)}")
    logger.info(bold(f"  ParlayPal  --  {tomorrow_display}  --  All games OTB"))
    logger.info(bold(sep))
    logger.info(f"\n  {yellow('All tomorrow games are blocked by unfinished games tonight.')}")
    logger.info("  Re-run after the final buzzer to get projections.\n")
    for g in otb_labels:
        logger.info(f"    {yellow('>>>')} {g}")
    logger.info(f"\n{bold(sep)}\n")


if __name__ == "__main__":
    main()
