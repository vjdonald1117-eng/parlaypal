"""
Hybrid player-prop predictor (XGBoost mean + Monte Carlo simulation).

For each leg:
1) Build contextual feature row from DB (recent form, home/away, rest, opponent).
2) Predict expected stat mean with xgboost_<stat>_model.json.
3) Run Monte Carlo (default 10,000 runs):
   - Normal for higher-volume stats (PTS/REB/AST)
   - Poisson for lower-count stats (STL/BLK/FG3)
4) Compute edge from simulated OVER probability vs sportsbook breakeven.
5) Output a confidence score (0-100) per leg.

Usage examples
--------------
Single leg:
  python models/hybrid_predictor.py --player "Jalen Brunson" --stat pts --line 27.5

Batch legs from JSON file:
  python models/hybrid_predictor.py --legs-file data/parlay_legs.json

Leg JSON format:
[
  {"player": "Jalen Brunson", "stat": "pts", "line": 27.5, "over_odds": -110},
  {"player": "Bam Adebayo", "stat": "reb", "line": 10.5}
]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


VALID_STATS = {"pts", "reb", "ast", "stl", "blk", "fg3"}
STAT_SQL_COL = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "fg3": "fg3m",
}
NORMAL_STATS = {"pts", "reb", "ast"}
BREAKEVEN_DEFAULT = 52.38  # -110 breakeven
_RNG = np.random.default_rng(42)

_models_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = str(Path(__file__).resolve().parents[1])
load_dotenv(os.path.join(_repo_dir, ".env"))

import logging
from core.logger import configure_app_logging

configure_app_logging()
logger = logging.getLogger(__name__)

from models.rotation_coaching_features import (
    blowout_risk_from_spread,
    fetch_rotation_features_live,
    is_starter_for_team,
    starter_mean_scale,
)


def current_season() -> str:
    today = datetime.now(ZoneInfo("America/New_York")).date()
    if today.month >= 10:
        return f"{today.year}-{str(today.year + 1)[2:]}"
    return f"{today.year - 1}-{str(today.year)[2:]}"


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in .env")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def breakeven_pct_from_american(odds: int) -> float:
    if odds < 0:
        return (abs(odds) / (abs(odds) + 100.0)) * 100.0
    return (100.0 / (odds + 100.0)) * 100.0


@dataclass
class LegInput:
    player: str
    stat: str
    line: float
    over_odds: int = -110
    opponent: str | None = None
    is_home: bool | None = None


def _resolve_player_row(session: Session, player_name: str):
    row = session.execute(
        text(
            """
            SELECT p.id, p.full_name, p.team_id, t.abbreviation AS team_abbr
            FROM players p
            JOIN teams t ON t.id = p.team_id
            WHERE LOWER(TRIM(p.full_name)) = LOWER(TRIM(:name))
            LIMIT 1
            """
        ),
        {"name": player_name},
    ).fetchone()
    return row


def _resolve_tonight_context(
    session: Session, *, team_id: int, game_date: date
) -> dict[str, Any] | None:
    row = session.execute(
        text(
            """
            SELECT
                g.id,
                g.home_team_id,
                g.away_team_id,
                g.home_days_rest,
                g.away_days_rest,
                g.closing_spread,
                g.closing_total,
                ht.abbreviation AS home_abbr,
                at.abbreviation AS away_abbr
            FROM games g
            JOIN teams ht ON ht.id = g.home_team_id
            JOIN teams at ON at.id = g.away_team_id
            WHERE g.game_date = :gd
              AND (g.status IS NULL OR g.status <> 'Final')
              AND (:tid = g.home_team_id OR :tid = g.away_team_id)
            ORDER BY g.id
            LIMIT 1
            """
        ),
        {"gd": game_date.isoformat(), "tid": team_id},
    ).fetchone()
    if not row:
        return None
    is_home = int(row.home_team_id) == int(team_id)
    opp_team_id = int(row.away_team_id if is_home else row.home_team_id)
    opp_abbr = str(row.away_abbr if is_home else row.home_abbr)
    days_rest = row.home_days_rest if is_home else row.away_days_rest
    cs_raw = row.closing_spread
    cs = float(cs_raw) if cs_raw is not None else 0.0
    ct_raw = row.closing_total
    ct = float(ct_raw) if ct_raw is not None else 220.0
    return {
        "game_id": int(row.id),
        "is_home": bool(is_home),
        "opponent_team_id": opp_team_id,
        "opponent_abbr": opp_abbr,
        "days_rest": float(days_rest) if days_rest is not None else 1.0,
        "closing_spread": cs,
        "closing_total": ct,
        "is_potential_blowout": 1.0 if abs(cs) > 10.0 else 0.0,
    }


def _player_recent_form(
    session: Session, *, player_id: int, stat: str, n_games: int = 5
) -> tuple[float, float] | None:
    col = STAT_SQL_COL[stat]
    rows = session.execute(
        text(
            f"""
            SELECT pbs.{col} AS stat_value, pbs.minutes_played
            FROM player_box_scores_traditional pbs
            JOIN games g ON g.id = pbs.game_id
            WHERE pbs.player_id = :pid
              AND pbs.dnp_status = FALSE
              AND pbs.minutes_played >= 5
            ORDER BY g.game_date DESC
            LIMIT :n
            """
        ),
        {"pid": player_id, "n": n_games},
    ).fetchall()
    if not rows:
        return None
    return float(np.mean([float(r.stat_value) for r in rows])), float(
        np.mean([float(r.minutes_played) for r in rows])
    )


def _get_stat_std(session: Session, *, player_id: int, stat: str, season: str) -> float:
    col = STAT_SQL_COL[stat]
    q = text(
        f"""
        SELECT STDDEV(pbs.{col}) AS std_dev, COUNT(*) AS gp
        FROM player_box_scores_traditional pbs
        JOIN games g ON g.id = pbs.game_id
        WHERE pbs.player_id = :pid
          AND pbs.dnp_status = FALSE
          AND g.season = :season
        """
    )
    cur = session.execute(q, {"pid": player_id, "season": season}).fetchone()
    if cur and cur.std_dev is not None and cur.gp >= 5:
        return float(cur.std_dev)
    prev = f"{int(season[:4]) - 1}-{season[:4][2:]}"
    old = session.execute(q, {"pid": player_id, "season": prev}).fetchone()
    if old and old.std_dev is not None and old.gp >= 5:
        return float(old.std_dev)
    return 1.75 if stat in {"stl", "blk", "fg3"} else 4.5


def _load_stat_model(stat: str) -> xgb.XGBRegressor:
    model_path = os.path.join(_models_dir, f"xgboost_{stat}_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file missing for stat={stat}: {model_path}\n"
            f"Train it with: python models/train_xgboost.py --stat {stat} --auto-build-data"
        )
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def _build_model_input(
    model: xgb.XGBRegressor,
    session: Session,
    *,
    stat: str,
    player_id: int,
    game_date: date,
    last_avg_stat: float,
    last_avg_min: float,
    is_home: bool,
    days_rest: float,
    opponent_team_id: int | None,
    ctx: dict[str, Any],
) -> pd.DataFrame:
    feat_names = list(model.feature_names_in_)
    row: dict[str, float] = {name: 0.0 for name in feat_names}
    row[f"last_5_avg_{stat}"] = float(last_avg_stat)
    row["last_5_avg_min"] = float(last_avg_min)
    row["is_home"] = 1.0 if is_home else 0.0
    row["days_rest"] = float(days_rest)
    if "closing_spread" in row:
        row["closing_spread"] = float(ctx.get("closing_spread", 0.0))
    if "closing_total" in row:
        row["closing_total"] = float(ctx.get("closing_total", 220.0))
    if "is_potential_blowout" in row:
        row["is_potential_blowout"] = float(ctx.get("is_potential_blowout", 0.0))
    if opponent_team_id is not None:
        ohe_col = f"opp_{int(opponent_team_id)}"
        if ohe_col in row:
            row[ohe_col] = 1.0
    rot_keys = (
        "historical_minute_overlap",
        "usage_to_sub_ratio",
        "coach_id",
        "bench_reliance_factor",
    )
    if any(k in feat_names for k in rot_keys):
        rot = fetch_rotation_features_live(session, player_id, opponent_team_id, game_date)
        for k, v in rot.items():
            if k in row:
                row[k] = float(v)
    return pd.DataFrame([row])


def _simulate_samples(stat: str, mean: float, std_dev: float, n_sims: int) -> np.ndarray:
    mean = max(0.0, float(mean))
    std_dev = max(0.1, float(std_dev))
    if stat in NORMAL_STATS:
        s = _RNG.normal(loc=mean, scale=std_dev, size=n_sims)
        return np.clip(s, 0.0, None)
    # Low-count box-score stats are better approximated as count processes.
    lam = max(0.05, mean)
    return _RNG.poisson(lam=lam, size=n_sims).astype(float)


def _confidence_score(over_pct: float, edge_pct: float, mean: float, std_dev: float) -> float:
    side_certainty = min(1.0, abs(over_pct - 50.0) / 25.0)  # 50->0, 75->1
    edge_strength = min(1.0, max(0.0, edge_pct) / 12.0)  # +12pp => strong
    cv = std_dev / max(0.5, mean)
    stability = max(0.0, 1.0 - min(1.0, cv / 2.0))
    score = 100.0 * (0.55 * side_certainty + 0.30 * edge_strength + 0.15 * stability)
    return round(max(0.0, min(100.0, score)), 1)


def evaluate_leg(
    session: Session,
    *,
    leg: LegInput,
    game_date: date,
    season: str,
    n_sims: int,
) -> dict[str, Any]:
    stat = leg.stat.lower().strip()
    if stat not in VALID_STATS:
        raise ValueError(f"Invalid stat: {leg.stat}. Use one of {sorted(VALID_STATS)}.")

    player = _resolve_player_row(session, leg.player)
    if not player:
        raise ValueError(f"Player not found: {leg.player}")

    ctx = _resolve_tonight_context(session, team_id=int(player.team_id), game_date=game_date)
    if not ctx:
        raise ValueError(f"No non-final game found for {player.full_name} on {game_date}.")

    recent = _player_recent_form(session, player_id=int(player.id), stat=stat, n_games=5)
    if not recent:
        raise ValueError(f"Insufficient recent games for {player.full_name} ({stat}).")
    last_avg_stat, last_avg_min = recent

    model = _load_stat_model(stat)
    x_in = _build_model_input(
        model,
        session,
        stat=stat,
        player_id=int(player.id),
        game_date=game_date,
        last_avg_stat=last_avg_stat,
        last_avg_min=last_avg_min,
        is_home=ctx["is_home"],
        days_rest=ctx["days_rest"],
        opponent_team_id=ctx["opponent_team_id"],
        ctx=ctx,
    )
    xgb_mean = float(model.predict(x_in)[0])
    mc_scale = starter_mean_scale(
        blowout_risk_from_spread(ctx.get("closing_spread")),
        is_starter_for_team(session, int(player.id), int(player.team_id), season),
    )
    xgb_mean *= mc_scale

    std_dev = _get_stat_std(session, player_id=int(player.id), stat=stat, season=season)
    samples = _simulate_samples(stat, xgb_mean, std_dev, n_sims)

    over_prob = float(np.mean(samples > float(leg.line)))
    over_pct = over_prob * 100.0
    under_pct = 100.0 - over_pct
    breakeven = breakeven_pct_from_american(int(leg.over_odds))
    edge_pct = over_pct - breakeven
    best_side = "OVER" if over_pct >= 50.0 else "UNDER"
    best_pct = max(over_pct, under_pct)
    confidence = _confidence_score(over_pct=over_pct, edge_pct=edge_pct, mean=xgb_mean, std_dev=std_dev)

    return {
        "player": player.full_name,
        "team_abbr": player.team_abbr,
        "opponent": ctx["opponent_abbr"],
        "game_date": game_date.isoformat(),
        "stat": stat,
        "line": float(leg.line),
        "over_odds": int(leg.over_odds),
        "distribution": "normal" if stat in NORMAL_STATS else "poisson",
        "xgb_context_mean": round(xgb_mean, 3),
        "std_dev_used": round(float(std_dev), 3),
        "over_pct": round(over_pct, 2),
        "under_pct": round(under_pct, 2),
        "edge_over_vs_breakeven_pct": round(edge_pct, 2),
        "best_side": best_side,
        "best_pct": round(best_pct, 2),
        "confidence_score": confidence,
        "n_sims": int(n_sims),
    }


def _parse_legs_file(path: str) -> list[LegInput]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("legs-file must contain a JSON array.")
    legs: list[LegInput] = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"legs-file row {i} is not an object.")
        legs.append(
            LegInput(
                player=str(item["player"]),
                stat=str(item["stat"]).lower(),
                line=float(item["line"]),
                over_odds=int(item.get("over_odds", -110)),
                opponent=item.get("opponent"),
                is_home=item.get("is_home"),
            )
        )
    return legs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid XGB + Monte Carlo player-prop predictor.")
    p.add_argument("--player", default="", help="Single-leg player full name.")
    p.add_argument("--stat", default="pts", choices=sorted(VALID_STATS), help="Single-leg stat.")
    p.add_argument("--line", type=float, default=None, help="Single-leg sportsbook line.")
    p.add_argument("--over-odds", type=int, default=-110, help="American odds for OVER (default -110).")
    p.add_argument("--legs-file", default="", help="JSON file containing multiple leg objects.")
    p.add_argument(
        "--game-date",
        default=None,
        help="Target slate date YYYY-MM-DD (default: today's NY date).",
    )
    p.add_argument("--season", default=None, help="Season string (default current).")
    p.add_argument("--n-sims", type=int, default=10_000, help="Monte Carlo samples per leg.")
    p.add_argument("--output-json", default="", help="Optional output file path for results JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    game_date = (
        datetime.strptime(args.game_date, "%Y-%m-%d").date()
        if args.game_date
        else datetime.now(ZoneInfo("America/New_York")).date()
    )
    season = args.season or current_season()

    legs: list[LegInput]
    if args.legs_file:
        legs = _parse_legs_file(args.legs_file)
    else:
        if not args.player or args.line is None:
            raise ValueError("Single-leg mode requires --player and --line.")
        legs = [
            LegInput(
                player=args.player,
                stat=args.stat,
                line=float(args.line),
                over_odds=int(args.over_odds),
            )
        ]

    engine = get_engine()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with Session(engine) as session:
        for leg in legs:
            try:
                res = evaluate_leg(
                    session,
                    leg=leg,
                    game_date=game_date,
                    season=season,
                    n_sims=int(args.n_sims),
                )
                results.append(res)
            except Exception as exc:
                errors.append(
                    {
                        "player": leg.player,
                        "stat": leg.stat,
                        "line": str(leg.line),
                        "error": str(exc),
                    }
                )

    logger.info(f"\nHybrid predictions for {game_date}  |  season={season}")
    logger.info("-" * 88)
    for r in results:
        logger.info(
            f"{r['player']:<22} {r['stat'].upper():<4} line {r['line']:<5} "
            f"OVER {r['over_pct']:>6.2f}%  EDGE {r['edge_over_vs_breakeven_pct']:>6.2f}pp  "
            f"CONF {r['confidence_score']:>5.1f}"
        )
    if errors:
        logger.info("\nSkipped legs:")
        for e in errors:
            logger.info(f"  - {e['player']} {e['stat']} {e['line']}: {e['error']}")

    payload = {"game_date": game_date.isoformat(), "season": season, "results": results, "errors": errors}
    if args.output_json:
        out = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()

