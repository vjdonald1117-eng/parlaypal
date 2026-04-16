"""
Rotation & coaching proxies for XGBoost prop training / live inference.

We do not have play-by-play subs or Defensive Win Shares in the DB. This module uses:
  - player_box_scores_advanced.def_rating  (lower = better defender in NBA convention)
  - usg_pct for usage / rotation volatility proxies
  - Team bench minute share in close games as bench_reliance_factor
  - coach_id: stable numeric surrogate = opponent_team_id (one staff bucket per opponent org)

historical_minute_overlap:
  Pregame estimate: min(target player's rolling avg minutes, opponent "lock" defender's
  rolling avg minutes), where the lock is the opponent rotation player (>=8 min in game)
  with best (minimum) prior rolling mean def_rating.

usage_to_sub_ratio:
  Proxy: opponent_team_mean_usg / (1 + std of rotation minutes volatility).
  Higher means star-heavy usage with volatile minute distribution (faster hook signal).

bench_reliance_factor:
  For the opponent, in prior *close* games (|margin| < 10), share of team minutes played
  by non-top-5 players (by season-to-date avg minutes before that game).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def load_advanced_box_columns(engine: Engine) -> pd.DataFrame:
    q = text(
        """
        SELECT
            pbs.game_id,
            pbs.player_id,
            pbs.team_id,
            pbs.minutes_played,
            pba.def_rating,
            pba.usg_pct
        FROM player_box_scores_traditional pbs
        LEFT JOIN player_box_scores_advanced pba
          ON pba.game_id = pbs.game_id AND pba.player_id = pbs.player_id
        JOIN games g ON g.id = pbs.game_id
        WHERE pbs.dnp_status = FALSE
          AND pbs.minutes_played >= 5
          AND g.home_score IS NOT NULL
          AND g.away_score IS NOT NULL
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(q, conn)
    return df


def load_game_margins(engine: Engine) -> pd.DataFrame:
    q = text(
        """
        SELECT id AS game_id, game_date, home_team_id, away_team_id,
               home_score, away_score,
               ABS(home_score - away_score) AS margin
        FROM games
        WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(q, conn)


def _prep_player_game_features(adv_df: pd.DataFrame) -> pd.DataFrame:
    """Per player-game rolling *prior* defensive rating and minutes."""
    df = adv_df.sort_values(["player_id", "game_id"]).copy()
    g = df.groupby("player_id", sort=False)
    df["roll_def_pre"] = g["def_rating"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=2).mean()
    )
    df["roll_min_pre"] = g["minutes_played"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=2).mean()
    )
    df["roll_usg_pre"] = g["usg_pct"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=2).mean()
    )
    return df


def _lock_features_by_game_team(df_prep: pd.DataFrame) -> pd.DataFrame:
    """
    For each (game_id, team_id) the lock defender row among rotation players
    (>=8 min in that game) with minimum roll_def_pre.
    """
    rows = []
    for (gid, tid), grp in df_prep.groupby(["game_id", "team_id"], sort=False):
        sub = grp[(grp["minutes_played"] >= 8) & grp["roll_def_pre"].notna()].copy()
        if sub.empty:
            rows.append(
                {
                    "game_id": gid,
                    "team_id": tid,
                    "lock_roll_min_pre": 20.0,
                    "lock_roll_def_pre": 112.0,
                    "lock_mean_usg_pre": 18.0,
                }
            )
            continue
        i = sub["roll_def_pre"].idxmin()
        r = sub.loc[i]
        rows.append(
            {
                "game_id": gid,
                "team_id": tid,
                "lock_roll_min_pre": float(r["roll_min_pre"] or 20.0),
                "lock_roll_def_pre": float(r["roll_def_pre"]),
                "lock_mean_usg_pre": float(r["roll_usg_pre"] or 18.0),
            }
        )
    return pd.DataFrame(rows)


def _team_minute_volatility(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team_id): std of rotation minutes (>=6) in that game."""
    vol_rows = []
    for (gid, tid), grp in df_prep.groupby(["game_id", "team_id"], sort=False):
        m = grp.loc[grp["minutes_played"] >= 6, "minutes_played"]
        vol_rows.append(
            {
                "game_id": gid,
                "team_id": tid,
                "min_volatility": float(m.std()) if len(m) > 1 else 4.0,
                "mean_usg_rot": float(
                    grp.loc[grp["minutes_played"] >= 12, "usg_pct"].mean() or 18.0
                ),
            }
        )
    return pd.DataFrame(vol_rows)


def _bench_reliance_by_game(
    df_prep: pd.DataFrame, margins: pd.DataFrame, close_margin: float = 10.0
) -> pd.DataFrame:
    """
    For each team appearance in a close game, compute bench minute share:
    1 - (sum of top 5 player minutes in that game / team regulation minutes sum).
    """
    mg = margins[margins["margin"] < close_margin][["game_id"]].drop_duplicates()
    df = df_prep.merge(mg, on="game_id", how="inner")
    out_rows = []
    for (gid, tid), grp in df.groupby(["game_id", "team_id"], sort=False):
        mins = grp["minutes_played"].sort_values(ascending=False).values
        if len(mins) == 0:
            continue
        top5 = float(np.sum(mins[:5]))
        total = float(np.sum(mins))
        if total <= 0:
            continue
        bench_share = max(0.0, min(1.0, 1.0 - top5 / max(total, 1e-6)))
        out_rows.append({"game_id": gid, "team_id": tid, "bench_share_game": bench_share})
    if not out_rows:
        return pd.DataFrame(columns=["team_id", "game_date", "bench_reliance_factor"])
    bg = pd.DataFrame(out_rows)
    # attach game_date for expanding mean per team
    bg = bg.merge(
        margins[["game_id", "game_date"]].drop_duplicates(),
        on="game_id",
        how="left",
    )
    bg = bg.sort_values(["team_id", "game_date"])
    bg["bench_reliance_factor"] = bg.groupby("team_id", sort=False)["bench_share_game"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return bg[["team_id", "game_date", "bench_reliance_factor"]].drop_duplicates()


def apply_rotation_coaching_features(
    featured_df: pd.DataFrame,
    engine: Engine,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """
    Mutates a copy of featured_df (must contain game_id, opponent_team_id,
    last_{w}_avg_min for rolling_window, game_date).
    """
    out = featured_df.copy()
    adv = load_advanced_box_columns(engine)
    margins = load_game_margins(engine)
    prep = _prep_player_game_features(adv)
    locks = _lock_features_by_game_team(prep)
    vol = _team_minute_volatility(prep)
    bench = _bench_reliance_by_game(prep, margins)

    min_col = f"last_{rolling_window}_avg_min"
    if min_col not in out.columns:
        raise KeyError(f"Expected column {min_col} on training frame")

    lk = locks.rename(
        columns={
            "team_id": "opponent_team_id",
            "lock_roll_min_pre": "_lock_min",
            "lock_roll_def_pre": "_lock_def",
            "lock_mean_usg_pre": "_lock_usg",
        }
    )
    out = out.merge(
        lk[["game_id", "opponent_team_id", "_lock_min", "_lock_def", "_lock_usg"]],
        on=["game_id", "opponent_team_id"],
        how="left",
    )
    out["_lock_min"] = out["_lock_min"].fillna(20.0)
    out["_lock_def"] = out["_lock_def"].fillna(112.0)

    out["historical_minute_overlap"] = np.minimum(
        pd.to_numeric(out[min_col], errors="coerce").fillna(24.0),
        out["_lock_min"],
    )

    vo = vol.rename(columns={"team_id": "opponent_team_id"})
    out = out.merge(
        vo[["game_id", "opponent_team_id", "min_volatility", "mean_usg_rot"]],
        on=["game_id", "opponent_team_id"],
        how="left",
    )
    out["min_volatility"] = out["min_volatility"].fillna(4.0)
    out["mean_usg_rot"] = out["mean_usg_rot"].fillna(18.0)
    out["usage_to_sub_ratio"] = out["mean_usg_rot"] / (1.0 + out["min_volatility"])

    out["coach_id"] = pd.to_numeric(out["opponent_team_id"], errors="coerce").fillna(0).astype(float)

    out = out.merge(
        bench,
        left_on=["opponent_team_id", "game_date"],
        right_on=["team_id", "game_date"],
        how="left",
    )
    out["bench_reliance_factor"] = out["bench_reliance_factor"].fillna(0.32)
    out.drop(columns=[c for c in out.columns if c in ("_lock_min", "_lock_def", "_lock_usg", "team_id")], errors="ignore", inplace=True)
    out.drop(columns=["min_volatility", "mean_usg_rot"], errors="ignore", inplace=True)

    return out


def fetch_rotation_features_live(
    session,
    player_id: int,
    opponent_team_id: int | None,
    game_date: object,
) -> dict[str, float]:
    """
    Best-effort live features (pregame) for parlay_builder._build_xgb_input.
    """
    defaults = {
        "historical_minute_overlap": 22.0,
        "usage_to_sub_ratio": 1.0,
        "coach_id": float(opponent_team_id or 0),
        "bench_reliance_factor": 0.32,
    }
    if opponent_team_id is None:
        return defaults

    from sqlalchemy import text as sql_text

    # Next game for opponent + recent close-game bench share
    q_bench = sql_text(
        """
        WITH close_games AS (
            SELECT g.id, g.game_date, g.home_team_id AS tid
            FROM games g
            WHERE ABS(g.home_score - g.away_score) < 10
              AND g.home_score IS NOT NULL
            UNION ALL
            SELECT g.id, g.game_date, g.away_team_id AS tid
            FROM games g
            WHERE ABS(g.home_score - g.away_score) < 10
              AND g.home_score IS NOT NULL
        ),
        opp_close AS (
            SELECT cg.id, cg.game_date
            FROM close_games cg
            WHERE cg.tid = :otid
            ORDER BY cg.game_date DESC
            LIMIT 25
        )
        SELECT AVG(x.bshare) AS br
        FROM (
            SELECT g.id,
                   CASE WHEN SUM(pbs.minutes_played) > 0 THEN
                     GREATEST(0.0, 1.0 - (
                       SELECT SUM(t.m) FROM (
                         SELECT pbs2.minutes_played AS m
                         FROM player_box_scores_traditional pbs2
                         WHERE pbs2.game_id = g.id AND pbs2.team_id = :otid
                           AND pbs2.dnp_status = FALSE
                         ORDER BY pbs2.minutes_played DESC NULLS LAST
                         LIMIT 5
                       ) t
                     ) / NULLIF(SUM(pbs.minutes_played), 0))
                   ELSE 0.32 END AS bshare
            FROM games g
            JOIN player_box_scores_traditional pbs ON pbs.game_id = g.id AND pbs.team_id = :otid
            WHERE g.id IN (SELECT id FROM opp_close)
              AND pbs.dnp_status = FALSE
            GROUP BY g.id
        ) x
        """
    )
    try:
        r = session.execute(q_bench, {"otid": opponent_team_id}).fetchone()
        if r and r.br is not None:
            defaults["bench_reliance_factor"] = float(max(0.05, min(0.85, float(r.br))))
    except Exception:
        pass

    # Lock defender overlap: min(player roll min, opp lock roll min) from last games
    q_lock = sql_text(
        """
        WITH opp_players AS (
            SELECT pbs.player_id,
                   AVG(pba.def_rating) AS avg_def,
                   AVG(pbs.minutes_played) AS avg_min
            FROM player_box_scores_traditional pbs
            JOIN player_box_scores_advanced pba
              ON pba.game_id = pbs.game_id AND pba.player_id = pbs.player_id
            JOIN games g ON g.id = pbs.game_id
            WHERE pbs.team_id = :otid
              AND pbs.dnp_status = FALSE
              AND pbs.minutes_played >= 8
              AND g.game_date < CAST(:gd AS date)
              AND g.home_score IS NOT NULL
            GROUP BY pbs.player_id
            HAVING COUNT(*) >= 3
            ORDER BY avg_def ASC NULLS LAST
            LIMIT 1
        ),
        pl_min AS (
            SELECT AVG(mins) AS pm FROM (
                SELECT pbs.minutes_played AS mins
                FROM player_box_scores_traditional pbs
                JOIN games g ON g.id = pbs.game_id
                WHERE pbs.player_id = :pid
                  AND pbs.dnp_status = FALSE
                  AND g.game_date < CAST(:gd AS date)
                  AND g.home_score IS NOT NULL
                ORDER BY g.game_date DESC
                LIMIT 10
            ) z
        )
        SELECT (SELECT pm FROM pl_min) AS pmin,
               (SELECT avg_min FROM opp_players LIMIT 1) AS lmin,
               (SELECT AVG(pba.usg_pct) FROM player_box_scores_traditional pbs
                JOIN player_box_scores_advanced pba
                  ON pba.game_id = pbs.game_id AND pba.player_id = pbs.player_id
                JOIN games g ON g.id = pbs.game_id
                WHERE pbs.team_id = :otid AND g.game_date < CAST(:gd AS date)
                  AND pbs.dnp_status = FALSE AND pbs.minutes_played >= 10
                  AND g.home_score IS NOT NULL) AS ousg,
               (SELECT STDDEV_POP(pbs.minutes_played) FROM player_box_scores_traditional pbs
                JOIN games g ON g.id = pbs.game_id
                WHERE pbs.team_id = :otid AND g.game_date < CAST(:gd AS date)
                  AND pbs.dnp_status = FALSE AND pbs.minutes_played >= 6
                  AND g.home_score IS NOT NULL) AS mvol
        """
    )
    try:
        gd = game_date.isoformat() if hasattr(game_date, "isoformat") else str(game_date)
        row = session.execute(q_lock, {"otid": opponent_team_id, "pid": player_id, "gd": gd}).fetchone()
        if row:
            pmin = float(row.pmin or 24.0)
            lmin = float(row.lmin or 22.0)
            defaults["historical_minute_overlap"] = float(min(pmin, lmin))
            ousg = float(row.ousg or 18.0)
            mvol = float(row.mvol or 4.0)
            defaults["usage_to_sub_ratio"] = float(ousg / (1.0 + max(0.5, mvol)))
    except Exception:
        pass

    defaults["coach_id"] = float(opponent_team_id)
    return defaults


def blowout_risk_from_spread(closing_spread_abs: float | None) -> float:
    """0..1 smooth ramp; ~0 below 8 pts spread, ~1 above 20."""
    if closing_spread_abs is None:
        return 0.0
    x = float(abs(closing_spread_abs))
    return float(1.0 / (1.0 + np.exp(-0.35 * (x - 12.0))))


def is_starter_for_team(session, player_id: int, team_id: int | None, season: str) -> bool:
    """Top 5 on team by average minutes in season = starters for MC bench rule."""
    if team_id is None:
        return True
    from sqlalchemy import text as sql_text

    q = sql_text(
        """
        WITH t AS (
            SELECT pbs.player_id, AVG(pbs.minutes_played) AS am
            FROM player_box_scores_traditional pbs
            JOIN games g ON g.id = pbs.game_id
            WHERE g.season = :season AND pbs.team_id = :tid
              AND pbs.dnp_status = FALSE
            GROUP BY pbs.player_id
        ),
        top5 AS (
            SELECT player_id FROM t ORDER BY am DESC NULLS LAST LIMIT 5
        )
        SELECT EXISTS (SELECT 1 FROM top5 WHERE player_id = :pid) AS is_starter
        """
    )
    try:
        r = session.execute(q, {"season": season, "tid": team_id, "pid": player_id}).fetchone()
        if r:
            return bool(r.is_starter)
    except Exception:
        pass
    return True


def starter_mean_scale(blowout_risk: float, is_starter: bool) -> float:
    """Apply up to 20% mean reduction for starters in high blowout-risk games."""
    if not is_starter or blowout_risk <= 0.01:
        return 1.0
    return float(1.0 - 0.20 * blowout_risk)
