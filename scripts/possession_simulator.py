"""
Bottom-up possession simulator (state-machine style).

This script simulates an NBA-like game using possession events only:
  - ball-handler selection
  - action selection (shoot / pass / turnover)
  - shot resolution
  - rebound resolution on misses

Final score is strictly the sum of simulated possession outcomes.
No normal distribution is used for scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


@dataclass(frozen=True)
class PlayerInput:
    name: str
    usage_rate: float
    true_shooting_pct: float
    assist_rate: float
    turnover_rate: float
    offensive_rebound_pct: float
    defensive_rebound_pct: float


class GameSimulator:
    def __init__(
        self,
        home_starters: list[dict[str, Any] | PlayerInput],
        away_starters: list[dict[str, Any] | PlayerInput],
        home_pace: float = 100.0,
        away_pace: float = 100.0,
        seed: int | None = None,
    ) -> None:
        if len(home_starters) != 5 or len(away_starters) != 5:
            raise ValueError("home_starters and away_starters must each contain exactly 5 players.")
        self.rng = random.Random(seed)
        self.home = [self._coerce_player(p) for p in home_starters]
        self.away = [self._coerce_player(p) for p in away_starters]
        self.home_pace = float(home_pace)
        self.away_pace = float(away_pace)
        self.avg_pace = (self.home_pace + self.away_pace) / 2.0
        # True pace target: expected possessions per team.
        self.target_possessions = max(70, int(round(self.avg_pace)))
        self.box_score: dict[str, dict[str, dict[str, int]]] = {"HOME": {}, "AWAY": {}}
        self.team_totals: dict[str, dict[str, int]] = {
            "HOME": {"PTS": 0, "AST": 0, "REB": 0, "ORB": 0, "DRB": 0, "TOV": 0, "FGM": 0, "FGA": 0},
            "AWAY": {"PTS": 0, "AST": 0, "REB": 0, "ORB": 0, "DRB": 0, "TOV": 0, "FGM": 0, "FGA": 0},
        }
        self._init_box_score()
        self._three_pt_prob = 0.37

    @staticmethod
    def _clip_rate(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(x)))

    def _coerce_player(self, src: dict[str, Any] | PlayerInput) -> dict[str, float | str]:
        if isinstance(src, PlayerInput):
            d = src.__dict__
        else:
            d = src
        return {
            "name": str(d["name"]),
            "usage_rate": self._clip_rate(float(d["usage_rate"]), 0.01, 0.80),
            "true_shooting_pct": self._clip_rate(float(d["true_shooting_pct"]), 0.30, 0.80),
            "assist_rate": self._clip_rate(float(d["assist_rate"]), 0.01, 0.70),
            "turnover_rate": self._clip_rate(float(d["turnover_rate"]), 0.01, 0.35),
            "offensive_rebound_pct": self._clip_rate(float(d["offensive_rebound_pct"]), 0.01, 0.20),
            "defensive_rebound_pct": self._clip_rate(float(d["defensive_rebound_pct"]), 0.05, 0.35),
        }

    def _init_box_score(self) -> None:
        for team_key, roster in (("HOME", self.home), ("AWAY", self.away)):
            for p in roster:
                name = str(p["name"])
                self.box_score[team_key][name] = {
                    "PTS": 0,
                    "AST": 0,
                    "REB": 0,
                    "ORB": 0,
                    "DRB": 0,
                    "TOV": 0,
                    "FGM": 0,
                    "FGA": 0,
                }

    def _add_stat(self, team_key: str, player_name: str, stat: str, val: int = 1) -> None:
        self.box_score[team_key][player_name][stat] += val
        if stat in self.team_totals[team_key]:
            self.team_totals[team_key][stat] += val
        if stat in ("ORB", "DRB"):
            self.team_totals[team_key]["REB"] += val
            self.box_score[team_key][player_name]["REB"] += val

    def determine_ball_handler(self, offense: list[dict[str, float | str]]) -> dict[str, float | str]:
        weights = [float(p["usage_rate"]) for p in offense]
        return self.rng.choices(offense, weights=weights, k=1)[0]

    def determine_action(self, handler: dict[str, float | str]) -> str:
        tov = float(handler["turnover_rate"])
        pass_rate = max(0.0, float(handler["assist_rate"]) * 0.75)
        shoot_rate = max(0.0, 1.0 - tov - pass_rate)
        total = tov + pass_rate + shoot_rate
        if total <= 0:
            return "shoot"
        r = self.rng.random() * total
        if r < tov:
            return "turnover"
        if r < tov + pass_rate:
            return "pass"
        return "shoot"

    def _pick_teammate_shooter(
        self, offense: list[dict[str, float | str]], passer_name: str
    ) -> dict[str, float | str]:
        candidates = [p for p in offense if str(p["name"]) != passer_name]
        weights = [float(p["usage_rate"]) for p in candidates]
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def _pick_passer(
        self, offense: list[dict[str, float | str]], shooter_name: str
    ) -> dict[str, float | str] | None:
        candidates = [p for p in offense if str(p["name"]) != shooter_name]
        if not candidates:
            return None
        weights = [float(p["assist_rate"]) for p in candidates]
        if sum(weights) <= 0:
            return None
        return self.rng.choices(candidates, weights=weights, k=1)[0]

    def resolve_shot(
        self,
        offense_key: str,
        offense: list[dict[str, float | str]],
        shooter: dict[str, float | str],
        assisted: bool,
    ) -> tuple[bool, dict[str, float | str] | None]:
        shooter_name = str(shooter["name"])
        self._add_stat(offense_key, shooter_name, "FGA", 1)
        made = self.rng.random() < float(shooter["true_shooting_pct"])
        assister = None
        if made:
            self._add_stat(offense_key, shooter_name, "FGM", 1)
            pts = 3 if self.rng.random() < self._three_pt_prob else 2
            self._add_stat(offense_key, shooter_name, "PTS", pts)
            if assisted:
                assister = self._pick_passer(offense, shooter_name)
                if assister is not None:
                    self._add_stat(offense_key, str(assister["name"]), "AST", 1)
        return made, assister

    def resolve_rebound(
        self,
        offense_key: str,
        defense_key: str,
        offense: list[dict[str, float | str]],
        defense: list[dict[str, float | str]],
    ) -> str:
        pool: list[tuple[str, dict[str, float | str], float, str]] = []
        for p in offense:
            pool.append((offense_key, p, float(p["offensive_rebound_pct"]), "ORB"))
        for p in defense:
            pool.append((defense_key, p, float(p["defensive_rebound_pct"]), "DRB"))
        weights = [max(0.0001, x[2]) for x in pool]
        team_key, rebounder, _, reb_type = self.rng.choices(pool, weights=weights, k=1)[0]
        self._add_stat(team_key, str(rebounder["name"]), reb_type, 1)
        return team_key

    def _simulate_one_possession(
        self,
        offense_key: str,
        defense_key: str,
        offense: list[dict[str, float | str]],
        defense: list[dict[str, float | str]],
    ) -> str:
        # A possession continues through offensive rebounds until turnover/make/defensive rebound.
        while True:
            handler = self.determine_ball_handler(offense)
            action = self.determine_action(handler)
            if action == "turnover":
                self._add_stat(offense_key, str(handler["name"]), "TOV", 1)
                return defense_key
            if action == "pass":
                shooter = self._pick_teammate_shooter(offense, str(handler["name"]))
                made, _assister = self.resolve_shot(offense_key, offense, shooter, assisted=True)
            else:
                made, _assister = self.resolve_shot(offense_key, offense, handler, assisted=False)
            if made:
                return defense_key
            reb_team = self.resolve_rebound(offense_key, defense_key, offense, defense)
            if reb_team == defense_key:
                return defense_key
            # Offensive rebound: same team keeps possession.

    def simulate_game(self) -> dict[str, Any]:
        home_possessions = 0
        away_possessions = 0
        offense_key = "HOME"
        while home_possessions < self.target_possessions or away_possessions < self.target_possessions:
            if offense_key == "HOME":
                if home_possessions >= self.target_possessions:
                    offense_key = "AWAY"
                    continue
                next_offense = self._simulate_one_possession("HOME", "AWAY", self.home, self.away)
                home_possessions += 1
                offense_key = next_offense
            else:
                if away_possessions >= self.target_possessions:
                    offense_key = "HOME"
                    continue
                next_offense = self._simulate_one_possession("AWAY", "HOME", self.away, self.home)
                away_possessions += 1
                offense_key = next_offense

        return {
            "target_possessions_per_team": self.target_possessions,
            "home_possessions": home_possessions,
            "away_possessions": away_possessions,
            "box_score": self.box_score,
            "team_totals": self.team_totals,
        }

    def print_box_score(self, result: dict[str, Any]) -> None:
        print("=" * 88)
        print(
            f"SIM FINAL  HOME {result['team_totals']['HOME']['PTS']} - "
            f"AWAY {result['team_totals']['AWAY']['PTS']}"
        )
        print(
            f"Possessions: HOME {result['home_possessions']} | AWAY {result['away_possessions']} "
            f"(target {result['target_possessions_per_team']})"
        )
        print("=" * 88)
        for team_key in ("HOME", "AWAY"):
            print(f"\n{team_key} BOX SCORE")
            print(f"{'Player':22s} {'PTS':>4s} {'FGM/FGA':>8s} {'AST':>4s} {'REB':>4s} {'ORB':>4s} {'DRB':>4s} {'TOV':>4s}")
            print("-" * 88)
            for player_name, s in self.box_score[team_key].items():
                fg = f"{s['FGM']}/{s['FGA']}"
                print(
                    f"{player_name[:22]:22s} {s['PTS']:4d} {fg:>8s} "
                    f"{s['AST']:4d} {s['REB']:4d} {s['ORB']:4d} {s['DRB']:4d} {s['TOV']:4d}"
                )
            tt = result["team_totals"][team_key]
            team_fg = f"{tt['FGM']}/{tt['FGA']}"
            print("-" * 88)
            print(
                f"{'TEAM TOTAL':22s} {tt['PTS']:4d} {team_fg:>8s} "
                f"{tt['AST']:4d} {tt['REB']:4d} {tt['ORB']:4d} {tt['DRB']:4d} {tt['TOV']:4d}"
            )


def _db_engine():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(root, ".env"))
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return create_engine(
        db_url,
        pool_pre_ping=True,
        connect_args={"sslmode": "require", "options": "-c timezone=utc"},
    )


def _ensure_games_sim_cols(conn) -> None:
    for col in (
        "simulated_home_score DOUBLE PRECISION",
        "simulated_away_score DOUBLE PRECISION",
        "simulated_total_points DOUBLE PRECISION",
        "simulated_margin DOUBLE PRECISION",
        "prediction_engine TEXT",
    ):
        conn.execute(text(f"ALTER TABLE public.games ADD COLUMN IF NOT EXISTS {col}"))


def _ensure_player_props_table(conn) -> None:
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS public.player_props (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL REFERENCES public.games(id) ON DELETE CASCADE,
                player_id INTEGER NOT NULL REFERENCES public.players(id) ON DELETE CASCADE,
                prop_stat VARCHAR(10) NOT NULL,
                vegas_line DOUBLE PRECISION,
                bookmaker TEXT NOT NULL DEFAULT 'draftkings',
                line_source TEXT NOT NULL DEFAULT 'EST',
                prediction_engine TEXT,
                predicted_value DOUBLE PRECISION,
                over_probability DOUBLE PRECISION,
                simulation_confidence DOUBLE PRECISION,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (game_id, player_id, prop_stat)
            )
            """
        )
    )
    conn.execute(
        text("ALTER TABLE public.player_props ADD COLUMN IF NOT EXISTS prediction_engine TEXT")
    )


def _simulate_one_game_many(game: dict[str, Any], n_sims: int, seed: int | None) -> dict[str, Any]:
    home_raw = game["home_starters"]
    away_raw = game["away_starters"]
    home_ortg = float(game.get("home_ortg") or 113.0)
    away_ortg = float(game.get("away_ortg") or 113.0)
    home_drtg = float(game.get("home_drtg") or 113.0)
    away_drtg = float(game.get("away_drtg") or 113.0)
    pace_h = float(game.get("home_pace") or 100.0)
    pace_a = float(game.get("away_pace") or 100.0)
    pace_h = max(85.0, min(110.0, pace_h))
    pace_a = max(85.0, min(110.0, pace_a))

    # Efficiency priors: boost offense vs weak defense, trim vs strong defense.
    def _ts_mult(off_ortg: float, opp_drtg: float) -> float:
        off_edge = (off_ortg - 113.0) / 120.0
        def_edge = (opp_drtg - 113.0) / 140.0
        return max(0.94, min(1.08, 1.0 + off_edge + def_edge))

    home_mult = _ts_mult(home_ortg, away_drtg)
    away_mult = _ts_mult(away_ortg, home_drtg)

    home: list[dict[str, Any]] = []
    for p in home_raw:
        q = dict(p)
        q["true_shooting_pct"] = max(0.38, min(0.82, float(q["true_shooting_pct"]) * home_mult))
        home.append(q)
    away: list[dict[str, Any]] = []
    for p in away_raw:
        q = dict(p)
        q["true_shooting_pct"] = max(0.38, min(0.82, float(q["true_shooting_pct"]) * away_mult))
        away.append(q)

    if len(home) != 5 or len(away) != 5:
        raise ValueError(f"game_id={game.get('game_id')} must have exactly 5 home and 5 away starters")

    per_player: dict[int, dict[str, list[float]]] = {}
    for p in home + away:
        per_player[int(p["player_id"])] = {"pts": [], "reb": [], "ast": []}

    home_totals: list[float] = []
    away_totals: list[float] = []

    for i in range(int(n_sims)):
        sim = GameSimulator(home, away, home_pace=pace_h, away_pace=pace_a, seed=(None if seed is None else seed + i))
        r = sim.simulate_game()
        hpts = float(r["team_totals"]["HOME"]["PTS"])
        apts = float(r["team_totals"]["AWAY"]["PTS"])
        home_totals.append(hpts)
        away_totals.append(apts)

        for p in home:
            name = str(p["name"])
            pid = int(p["player_id"])
            s = r["box_score"]["HOME"][name]
            per_player[pid]["pts"].append(float(s["PTS"]))
            per_player[pid]["reb"].append(float(s["REB"]))
            per_player[pid]["ast"].append(float(s["AST"]))
        for p in away:
            name = str(p["name"])
            pid = int(p["player_id"])
            s = r["box_score"]["AWAY"][name]
            per_player[pid]["pts"].append(float(s["PTS"]))
            per_player[pid]["reb"].append(float(s["REB"]))
            per_player[pid]["ast"].append(float(s["AST"]))

    return {
        "game_id": int(game["game_id"]),
        "home_mean": sum(home_totals) / len(home_totals),
        "away_mean": sum(away_totals) / len(away_totals),
        "player_samples": per_player,
    }


def _persist_outputs(conn, sim_res: dict[str, Any]) -> None:
    gid = int(sim_res["game_id"])
    home_mean = float(sim_res["home_mean"])
    away_mean = float(sim_res["away_mean"])
    conn.execute(
        text(
            """
            UPDATE games
            SET simulated_home_score = :h,
                simulated_away_score = :a,
                simulated_total_points = :t,
                simulated_margin = :m,
                prediction_engine = 'possession_sim_v1'
            WHERE id = :gid
            """
        ),
        {"h": home_mean, "a": away_mean, "t": home_mean + away_mean, "m": home_mean - away_mean, "gid": gid},
    )

    _ensure_player_props_table(conn)
    q_lines = text(
        """
        SELECT player_id, prop_stat, vegas_line
        FROM public.player_props
        WHERE game_id = :gid
          AND prop_stat IN ('pts','reb','ast')
        """
    )
    existing = conn.execute(q_lines, {"gid": gid}).fetchall()
    existing_map: dict[tuple[int, str], float | None] = {}
    for r in existing:
        existing_map[(int(r.player_id), str(r.prop_stat))] = (
            float(r.vegas_line) if r.vegas_line is not None else None
        )
    upd = text(
        """
        UPDATE public.player_props
        SET predicted_value = :pv,
            over_probability = :op,
            simulation_confidence = :sc,
            prediction_engine = 'possession_sim_v1',
            updated_at = NOW()
        WHERE game_id = :gid
          AND player_id = :pid
          AND prop_stat = :st
        """
    )
    upsert = text(
        """
        INSERT INTO public.player_props (
            game_id, player_id, prop_stat, vegas_line, bookmaker, line_source,
            predicted_value, over_probability, simulation_confidence, prediction_engine, updated_at
        ) VALUES (
            :gid, :pid, :st, :line, 'sim', 'EST',
            :pv, :op, :sc, 'possession_sim_v1', NOW()
        )
        ON CONFLICT (game_id, player_id, prop_stat) DO UPDATE SET
            predicted_value = EXCLUDED.predicted_value,
            over_probability = EXCLUDED.over_probability,
            simulation_confidence = EXCLUDED.simulation_confidence,
            prediction_engine = EXCLUDED.prediction_engine,
            updated_at = NOW()
        """
    )
    for pid, stats_map in sim_res["player_samples"].items():
        for st in ("pts", "reb", "ast"):
            vals = stats_map.get(st, [])
            if not vals:
                continue
            pv = float(sum(vals) / len(vals))
            line = existing_map.get((int(pid), st))
            if line is None:
                line = pv
            else:
                line = float(line)
            line_out = float(line)
            over = float(sum(1 for v in vals if v > line) / len(vals))
            conf = float(min(0.99, max(0.01, abs(over - 0.5) * 2.0)))
            conn.execute(upd, {"pv": pv, "op": over, "sc": conf, "gid": gid, "pid": int(pid), "st": st})
            conn.execute(
                upsert,
                {
                    "gid": gid,
                    "pid": int(pid),
                    "st": st,
                    "line": line_out,
                    "pv": pv,
                    "op": over,
                    "sc": conf,
                },
            )


def run_from_prep(prep_path: str, n_sims: int, seed: int | None) -> None:
    with open(prep_path, "r", encoding="utf-8") as f:
        prep = json.load(f)
    games = prep.get("games", [])
    if not games:
        print("[possession_simulator] No games in prep payload.")
        return
    eng = _db_engine()
    with eng.begin() as conn:
        _ensure_games_sim_cols(conn)
        for g in games:
            sim_res = _simulate_one_game_many(g, n_sims=n_sims, seed=seed)
            _persist_outputs(conn, sim_res)
            print(
                f"[possession_simulator] game_id={sim_res['game_id']} "
                f"HOME={sim_res['home_mean']:.1f} AWAY={sim_res['away_mean']:.1f}",
                flush=True,
            )


def _demo_run() -> None:
    home_demo = [
        {
            "name": "Home Guard 1",
            "usage_rate": 0.29,
            "true_shooting_pct": 0.60,
            "assist_rate": 0.30,
            "turnover_rate": 0.12,
            "offensive_rebound_pct": 0.03,
            "defensive_rebound_pct": 0.10,
        },
        {
            "name": "Home Guard 2",
            "usage_rate": 0.24,
            "true_shooting_pct": 0.58,
            "assist_rate": 0.18,
            "turnover_rate": 0.11,
            "offensive_rebound_pct": 0.04,
            "defensive_rebound_pct": 0.11,
        },
        {
            "name": "Home Wing",
            "usage_rate": 0.21,
            "true_shooting_pct": 0.57,
            "assist_rate": 0.14,
            "turnover_rate": 0.10,
            "offensive_rebound_pct": 0.05,
            "defensive_rebound_pct": 0.14,
        },
        {
            "name": "Home Forward",
            "usage_rate": 0.16,
            "true_shooting_pct": 0.56,
            "assist_rate": 0.10,
            "turnover_rate": 0.11,
            "offensive_rebound_pct": 0.09,
            "defensive_rebound_pct": 0.20,
        },
        {
            "name": "Home Center",
            "usage_rate": 0.10,
            "true_shooting_pct": 0.63,
            "assist_rate": 0.08,
            "turnover_rate": 0.14,
            "offensive_rebound_pct": 0.13,
            "defensive_rebound_pct": 0.26,
        },
    ]
    away_demo = [
        {
            "name": "Away Guard 1",
            "usage_rate": 0.28,
            "true_shooting_pct": 0.59,
            "assist_rate": 0.29,
            "turnover_rate": 0.12,
            "offensive_rebound_pct": 0.03,
            "defensive_rebound_pct": 0.10,
        },
        {
            "name": "Away Guard 2",
            "usage_rate": 0.22,
            "true_shooting_pct": 0.57,
            "assist_rate": 0.17,
            "turnover_rate": 0.11,
            "offensive_rebound_pct": 0.04,
            "defensive_rebound_pct": 0.11,
        },
        {
            "name": "Away Wing",
            "usage_rate": 0.23,
            "true_shooting_pct": 0.58,
            "assist_rate": 0.15,
            "turnover_rate": 0.10,
            "offensive_rebound_pct": 0.06,
            "defensive_rebound_pct": 0.15,
        },
        {
            "name": "Away Forward",
            "usage_rate": 0.15,
            "true_shooting_pct": 0.55,
            "assist_rate": 0.10,
            "turnover_rate": 0.11,
            "offensive_rebound_pct": 0.09,
            "defensive_rebound_pct": 0.20,
        },
        {
            "name": "Away Center",
            "usage_rate": 0.12,
            "true_shooting_pct": 0.62,
            "assist_rate": 0.08,
            "turnover_rate": 0.14,
            "offensive_rebound_pct": 0.13,
            "defensive_rebound_pct": 0.25,
        },
    ]

    sim = GameSimulator(home_demo, away_demo, home_pace=99.0, away_pace=101.0, seed=42)
    result = sim.simulate_game()
    sim.print_box_score(result)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bottom-up possession simulator.")
    ap.add_argument("--prep", default="", help="Path to prep_simulation JSON payload.")
    ap.add_argument("--n-sims", type=int, default=300, help="Monte Carlo runs per game for DB outputs.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed base.")
    args = ap.parse_args()
    if args.prep:
        run_from_prep(args.prep, n_sims=max(50, int(args.n_sims)), seed=args.seed)
    else:
        _demo_run()


if __name__ == "__main__":
    main()
