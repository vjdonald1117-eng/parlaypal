"""
Prepare possession-sim inputs from Supabase for tonight's slate.

Outputs JSON consumed by scripts/possession_simulator.py with:
  - projected starters per team
  - baseline rates (usage/ts/ast/tov/oreb/dreb)
  - XGBoost-adjusted modifiers (matchup + model signal)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import xgboost as xgb


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
_OUT_DEFAULT = os.path.join(_ROOT, "data", "simulation_prep_tonight.json")


def _current_season_for_date(ymd: str) -> str:
    d = datetime.strptime(ymd, "%Y-%m-%d").date()
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[2:]}"
    return f"{d.year - 1}-{str(d.year)[2:]}"


def _engine():
    load_dotenv(os.path.join(_ROOT, ".env"))
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def _load_xgb_pts():
    p = os.path.join(_MODELS, "xgboost_pts_model.json")
    if not os.path.exists(p):
        return None
    m = xgb.XGBRegressor()
    m.load_model(p)
    return m


def _today_ymd() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _games_for_date(session: Session, ymd: str):
    q = text(
        """
        SELECT g.id, g.game_date, g.home_team_id, g.away_team_id,
               g.home_days_rest, g.away_days_rest, g.closing_spread, g.closing_total,
               th.abbreviation AS home_abbr, ta.abbreviation AS away_abbr
        FROM games g
        JOIN teams th ON th.id = g.home_team_id
        JOIN teams ta ON ta.id = g.away_team_id
        WHERE g.game_date = CAST(:gd AS date)
          AND (g.status IS NULL OR LOWER(TRIM(g.status)) <> 'final')
        ORDER BY g.id
        """
    )
    return session.execute(q, {"gd": ymd}).fetchall()


def _team_def_rating(session: Session, team_id: int, season: str) -> float:
    q = text(
        """
        SELECT AVG(pba.def_rating) AS dr
        FROM player_box_scores_advanced pba
        JOIN games g ON g.id = pba.game_id
        WHERE pba.team_id = :tid AND g.season = :season
        """
    )
    r = session.execute(q, {"tid": team_id, "season": season}).fetchone()
    return float(r.dr) if r and r.dr is not None else 113.0


def _league_def_rating(session: Session, season: str) -> float:
    q = text(
        """
        SELECT AVG(pba.def_rating) AS dr
        FROM player_box_scores_advanced pba
        JOIN games g ON g.id = pba.game_id
        WHERE g.season = :season
        """
    )
    r = session.execute(q, {"season": season}).fetchone()
    return float(r.dr) if r and r.dr is not None else 113.0


def _projected_starters(session: Session, team_id: int, season: str):
    q = text(
        """
        SELECT p.id AS player_id, p.full_name, AVG(pbs.minutes_played) AS avg_min
        FROM player_box_scores_traditional pbs
        JOIN games g ON g.id = pbs.game_id
        JOIN players p ON p.id = pbs.player_id
        WHERE pbs.team_id = :tid
          AND g.season = :season
          AND pbs.dnp_status = FALSE
          AND pbs.minutes_played >= 5
        GROUP BY p.id, p.full_name
        ORDER BY avg_min DESC NULLS LAST
        LIMIT 5
        """
    )
    return session.execute(q, {"tid": team_id, "season": season}).fetchall()


def _team_last10_priors(session: Session, team_id: int, game_date_ymd: str) -> dict[str, float]:
    cols_q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'player_box_scores_advanced'
        """
    )
    adv_cols = {str(r.column_name).lower() for r in session.execute(cols_q).fetchall()}

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c.lower() in adv_cols:
                return c
        return None

    pace_col = _pick("pace")
    ortg_col = _pick("off_rating", "off_rtg", "offensive_rating")
    drtg_col = _pick("def_rating", "def_rtg", "defensive_rating")
    if not (pace_col and ortg_col and drtg_col):
        return {"pace": 99.5, "ortg": 113.0, "drtg": 113.0}

    q_league = text(
        """
        SELECT AVG(pba.__PACE__) AS lg_pace
        FROM player_box_scores_advanced pba
        JOIN games g ON g.id = pba.game_id
        WHERE g.game_date < CAST(:gd AS date)
          AND LOWER(TRIM(COALESCE(g.status, ''))) = 'final'
        """
        .replace("__PACE__", pace_col)
    )
    lg = session.execute(q_league, {"gd": game_date_ymd}).fetchone()
    league_pace_raw = float(lg.lg_pace or 100.0) if lg else 100.0

    q = text(
        """
        WITH per_game AS (
            SELECT
                pba.game_id,
                g.game_date,
                AVG(pba.__PACE__) AS pace_raw,
                AVG(pba.__ORTG__) AS ortg,
                AVG(pba.__DRTG__) AS drtg
            FROM player_box_scores_advanced pba
            JOIN games g ON g.id = pba.game_id
            WHERE pba.team_id = :tid
              AND g.game_date < CAST(:gd AS date)
              AND LOWER(TRIM(COALESCE(g.status, ''))) = 'final'
            GROUP BY pba.game_id, g.game_date
            ORDER BY g.game_date DESC
            LIMIT 10
        )
        SELECT AVG(pace_raw) AS pace_raw, AVG(ortg) AS ortg, AVG(drtg) AS drtg
        FROM per_game
        """
        .replace("__PACE__", pace_col)
        .replace("__ORTG__", ortg_col)
        .replace("__DRTG__", drtg_col)
    )
    r = session.execute(q, {"tid": team_id, "gd": game_date_ymd}).fetchone()
    if not r:
        return {"pace": 99.5, "ortg": 113.0, "drtg": 113.0}
    pace_raw = float(r.pace_raw or league_pace_raw or 100.0)
    pace_scaled = (pace_raw / max(1.0, league_pace_raw)) * 99.5
    return {
        "pace": float(min(108.0, max(88.0, pace_scaled))),
        "ortg": float(r.ortg or 113.0),
        "drtg": float(r.drtg or 113.0),
    }


def _baseline_rates(session: Session, player_id: int, season: str) -> dict[str, float]:
    def _to_unit_pct(v: float | None, fallback_pct: float) -> float:
        if v is None:
            return fallback_pct / 100.0
        x = float(v)
        # Support both DB encodings: 57.3 and 0.573
        return (x / 100.0) if x > 1.0 else x

    cols_q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'player_box_scores_advanced'
        """
    )
    adv_cols = {str(r.column_name).lower() for r in session.execute(cols_q).fetchall()}

    def _pick(*cands: str) -> str | None:
        for c in cands:
            if c.lower() in adv_cols:
                return c
        return None

    usg_col = _pick("usg_pct")
    ts_col = _pick("ts_pct")
    ast_col = _pick("ast_pct")
    tov_col = _pick("tov_pct", "to_ratio", "tov_ratio")
    oreb_col = _pick("oreb_pct")
    dreb_col = _pick("dreb_pct")
    if not (usg_col and ts_col and ast_col and tov_col and oreb_col and dreb_col):
        return {
            "usage_rate": 0.20,
            "true_shooting_pct": 0.56,
            "assist_rate": 0.15,
            "turnover_rate": 0.12,
            "offensive_rebound_pct": 0.06,
            "defensive_rebound_pct": 0.14,
            "avg_pts": 12.0,
            "avg_min": 26.0,
        }

    q = text(
        """
        SELECT
            AVG(pba.__USG__)  AS usage_rate,
            AVG(pba.__TS__)   AS true_shooting_pct,
            AVG(pba.__AST__)  AS assist_rate,
            AVG(pba.__TOV__)  AS turnover_rate,
            AVG(pba.__OREB__) AS offensive_rebound_pct,
            AVG(pba.__DREB__) AS defensive_rebound_pct,
            AVG(pbs.pts)      AS avg_pts,
            AVG(pbs.minutes_played) AS avg_min
        FROM player_box_scores_traditional pbs
        JOIN player_box_scores_advanced pba
          ON pba.game_id = pbs.game_id AND pba.player_id = pbs.player_id
        JOIN games g ON g.id = pbs.game_id
        WHERE pbs.player_id = :pid
          AND g.season = :season
          AND pbs.dnp_status = FALSE
          AND pbs.minutes_played >= 5
        """
        .replace("__USG__", usg_col)
        .replace("__TS__", ts_col)
        .replace("__AST__", ast_col)
        .replace("__TOV__", tov_col)
        .replace("__OREB__", oreb_col)
        .replace("__DREB__", dreb_col)
    )
    r = session.execute(q, {"pid": player_id, "season": season}).fetchone()
    if not r:
        return {
            "usage_rate": 0.20,
            "true_shooting_pct": 0.56,
            "assist_rate": 0.15,
            "turnover_rate": 0.12,
            "offensive_rebound_pct": 0.06,
            "defensive_rebound_pct": 0.14,
            "avg_pts": 12.0,
            "avg_min": 26.0,
        }
    return {
        "usage_rate": _to_unit_pct(r.usage_rate, 20.0),
        "true_shooting_pct": _to_unit_pct(r.true_shooting_pct, 56.0),
        "assist_rate": _to_unit_pct(r.assist_rate, 15.0),
        "turnover_rate": _to_unit_pct(r.turnover_rate, 12.0),
        "offensive_rebound_pct": _to_unit_pct(r.offensive_rebound_pct, 6.0),
        "defensive_rebound_pct": _to_unit_pct(r.defensive_rebound_pct, 14.0),
        "avg_pts": float(r.avg_pts or 12.0),
        "avg_min": float(r.avg_min or 26.0),
    }


def _xgb_pts_multiplier(
    model: xgb.XGBRegressor | None,
    *,
    baseline_pts: float,
    avg_min: float,
    is_home: bool,
    days_rest: float,
    opp_team_id: int,
) -> float:
    if model is None:
        return 1.0
    feats = {name: 0.0 for name in model.feature_names_in_}
    if "last_5_avg_pts" in feats:
        feats["last_5_avg_pts"] = baseline_pts
    if "last_5_avg_min" in feats:
        feats["last_5_avg_min"] = avg_min
    if "is_home" in feats:
        feats["is_home"] = 1.0 if is_home else 0.0
    if "days_rest" in feats:
        feats["days_rest"] = days_rest
    ohe = f"opp_{int(opp_team_id)}"
    if ohe in feats:
        feats[ohe] = 1.0
    x = np.array([[feats[k] for k in model.feature_names_in_]], dtype=float)
    pred = float(model.predict(x)[0])
    base = max(4.0, baseline_pts)
    mult = pred / base
    return float(min(1.25, max(0.85, mult)))


def build_prep(ymd: str, out_path: str) -> dict:
    season = _current_season_for_date(ymd)
    eng = _engine()
    model_pts = _load_xgb_pts()
    with Session(eng) as s:
        games = _games_for_date(s, ymd)
        lg_def = _league_def_rating(s, season)
        out_games = []
        for g in games:
            home_priors = _team_last10_priors(s, int(g.home_team_id), ymd)
            away_priors = _team_last10_priors(s, int(g.away_team_id), ymd)
            h_def = _team_def_rating(s, int(g.home_team_id), season)
            a_def = _team_def_rating(s, int(g.away_team_id), season)
            home_st = _projected_starters(s, int(g.home_team_id), season)
            away_st = _projected_starters(s, int(g.away_team_id), season)

            def _build_players(rows, is_home_team: bool):
                players = []
                opp_def = a_def if is_home_team else h_def
                opp_team_id = int(g.away_team_id if is_home_team else g.home_team_id)
                days_rest = float(g.home_days_rest or 1.0) if is_home_team else float(g.away_days_rest or 1.0)
                matchup_mult = float(min(1.10, max(0.90, lg_def / max(90.0, opp_def))))
                for r in rows:
                    b = _baseline_rates(s, int(r.player_id), season)
                    xgb_mult = _xgb_pts_multiplier(
                        model_pts,
                        baseline_pts=b["avg_pts"],
                        avg_min=b["avg_min"],
                        is_home=is_home_team,
                        days_rest=days_rest,
                        opp_team_id=opp_team_id,
                    )
                    combo = float(min(1.18, max(0.84, matchup_mult * (xgb_mult ** 0.5))))
                    players.append(
                        {
                            "player_id": int(r.player_id),
                            "name": str(r.full_name),
                            "usage_rate": float(min(0.45, max(0.06, b["usage_rate"] * (xgb_mult ** 0.35)))),
                            "true_shooting_pct": float(min(0.75, max(0.42, b["true_shooting_pct"] * combo))),
                            "assist_rate": float(min(0.55, max(0.04, b["assist_rate"] * (xgb_mult ** 0.2)))),
                            "turnover_rate": float(min(0.24, max(0.04, b["turnover_rate"] * (1.0 + (xgb_mult - 1.0) * 0.15)))),
                            "offensive_rebound_pct": float(min(0.20, max(0.01, b["offensive_rebound_pct"] * 1.02))),
                            "defensive_rebound_pct": float(min(0.32, max(0.05, b["defensive_rebound_pct"] * 1.01))),
                            "modifiers": {
                                "matchup_multiplier": matchup_mult,
                                "xgb_multiplier": xgb_mult,
                                "combined_multiplier": combo,
                            },
                        }
                    )
                return players

            out_games.append(
                {
                    "game_id": int(g.id),
                    "game_date": str(g.game_date),
                    "home_team_id": int(g.home_team_id),
                    "away_team_id": int(g.away_team_id),
                    "home_abbr": str(g.home_abbr),
                    "away_abbr": str(g.away_abbr),
                    "home_days_rest": float(g.home_days_rest or 1.0),
                    "away_days_rest": float(g.away_days_rest or 1.0),
                    "closing_spread": float(g.closing_spread) if g.closing_spread is not None else None,
                    "closing_total": float(g.closing_total) if g.closing_total is not None else None,
                    "home_pace": float(home_priors["pace"]),
                    "away_pace": float(away_priors["pace"]),
                    "home_ortg": float(home_priors["ortg"]),
                    "away_ortg": float(away_priors["ortg"]),
                    "home_drtg": float(home_priors["drtg"]),
                    "away_drtg": float(away_priors["drtg"]),
                    "home_starters": _build_players(home_st, True),
                    "away_starters": _build_players(away_st, False),
                }
            )

    payload = {"generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"), "game_date": ymd, "games": out_games}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare possession-sim inputs from Supabase + XGBoost modifiers.")
    ap.add_argument("--game-date", default=_today_ymd(), help="YYYY-MM-DD (default: NY today)")
    ap.add_argument("--output", default=_OUT_DEFAULT, help="Output JSON file path")
    args = ap.parse_args()
    payload = build_prep(args.game_date, os.path.abspath(args.output))
    print(
        f"[prep_simulation] wrote {len(payload['games'])} game(s) to {os.path.abspath(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

