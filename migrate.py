"""
NBA Analytics Engine  - Database Migration
Creates all core relational tables in Supabase via SQLAlchemy.

Schema requirements:
  - games: home/away identity explicit, days_rest stored per team per game
  - player_box_scores_traditional: minutes_played, dnp_status, dnp_reason, days_rest
  - injury_reports: chronological update tracking, data_source, status progression
  - Heavy indexing on player_id + game_id across all fact tables for 5yr data volumes
"""

import os
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, text,
    Column, Integer, String, Float, Boolean, Date, DateTime, Text, SmallInteger,
    ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base, relationship

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

# Supabase uses pgBouncer in transaction mode on port 6543.
# pool_pre_ping checks liveness on checkout; no prepared statements needed.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={
        "sslmode": "require",
        "options": "-c timezone=utc",
    },
    echo=True,
)

Base = declarative_base()


# ---------------------------------------------------------------------------
# teams
# ---------------------------------------------------------------------------
class Team(Base):
    __tablename__ = "teams"

    id           = Column(Integer, primary_key=True)       # Official NBA team ID
    abbreviation = Column(String(5),   nullable=False, unique=True)
    full_name    = Column(String(100), nullable=False)
    short_name   = Column(String(50))
    city         = Column(String(50))
    conference   = Column(String(10))                      # 'East' | 'West'
    division     = Column(String(20))
    created_at   = Column(DateTime(timezone=True), server_default=text("now()"))

    players    = relationship("Player",       back_populates="team")
    home_games = relationship("Game",         foreign_keys="Game.home_team_id", back_populates="home_team")
    away_games = relationship("Game",         foreign_keys="Game.away_team_id", back_populates="away_team")
    box_scores = relationship("TeamBoxScore", back_populates="team")


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------
class Player(Base):
    __tablename__ = "players"

    id             = Column(Integer, primary_key=True)     # Official NBA player ID
    team_id        = Column(Integer, ForeignKey("teams.id"), nullable=True)
    first_name     = Column(String(50),  nullable=False)
    last_name      = Column(String(50),  nullable=False)
    full_name      = Column(String(100), nullable=False)
    position       = Column(String(10))
    jersey_number  = Column(SmallInteger)
    height_inches  = Column(SmallInteger)
    weight_lbs     = Column(SmallInteger)
    birth_date     = Column(Date)
    country        = Column(String(60))
    draft_year     = Column(SmallInteger)
    draft_round    = Column(SmallInteger)
    draft_pick     = Column(SmallInteger)
    is_active      = Column(Boolean, default=True, nullable=False)
    created_at     = Column(DateTime(timezone=True), server_default=text("now()"))

    team            = relationship("Team",                       back_populates="players")
    trad_box_scores = relationship("PlayerBoxScoreTraditional",  back_populates="player")
    adv_box_scores  = relationship("PlayerBoxScoreAdvanced",     back_populates="player")
    injury_reports  = relationship("InjuryReport",               back_populates="player")

    __table_args__ = (
        Index("ix_players_team_id",    "team_id"),
        Index("ix_players_is_active",  "is_active"),
        Index("ix_players_full_name",  "full_name"),
    )


# ---------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------
class Game(Base):
    """
    home_team_id / away_team_id are the canonical source of truth for
    home/away identity  - every downstream join uses these to derive is_home.

    home_days_rest / away_days_rest are stored integers (populated by the
    data pipeline) so analysts never need to re-derive them at query time.
    NULL = season opener or first tracked game for that team.
    """
    __tablename__ = "games"

    id            = Column(Integer, primary_key=True)      # Official NBA game ID
    season        = Column(String(10), nullable=False)     # e.g. '2024-25'
    season_type   = Column(String(20), nullable=False)     # 'Regular Season' | 'Playoffs' | 'Pre Season'
    game_date     = Column(Date,       nullable=False)
    home_team_id  = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id  = Column(Integer, ForeignKey("teams.id"), nullable=False)
    home_score    = Column(SmallInteger)
    away_score    = Column(SmallInteger)
    status        = Column(String(20))                     # 'Final' | 'In Progress' | 'Scheduled'
    arena         = Column(String(100))
    attendance    = Column(Integer)

    # Days since each team's previous game  - NULL for season openers.
    # back-to-back = days_rest == 1; three-in-four = days_rest <= 3, etc.
    home_days_rest = Column(SmallInteger, nullable=True)
    away_days_rest = Column(SmallInteger, nullable=True)

    created_at    = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        UniqueConstraint("id", name="uq_games_id"),
        # Most common query patterns
        Index("ix_games_game_date",              "game_date"),
        Index("ix_games_season",                 "season"),
        Index("ix_games_home_team_id",           "home_team_id"),
        Index("ix_games_away_team_id",           "away_team_id"),
        Index("ix_games_season_date",            "season", "game_date"),
        Index("ix_games_home_team_date",         "home_team_id", "game_date"),
        Index("ix_games_away_team_date",         "away_team_id", "game_date"),
    )

    home_team       = relationship("Team", foreign_keys=[home_team_id], back_populates="home_games")
    away_team       = relationship("Team", foreign_keys=[away_team_id], back_populates="away_games")
    team_box_scores = relationship("TeamBoxScore",               back_populates="game")
    trad_box_scores = relationship("PlayerBoxScoreTraditional",  back_populates="game")
    adv_box_scores  = relationship("PlayerBoxScoreAdvanced",     back_populates="game")
    injury_reports  = relationship("InjuryReport",               back_populates="game")


# ---------------------------------------------------------------------------
# team_box_scores
# ---------------------------------------------------------------------------
class TeamBoxScore(Base):
    __tablename__ = "team_box_scores"

    id      = Column(Integer, primary_key=True, autoincrement=True)
    game_id = Column(Integer, ForeignKey("games.id"),  nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"),  nullable=False)

    # Explicit home/away flag  - mirrors games.home_team_id but stored here
    # so every box-score row is self-contained for analytics queries.
    is_home        = Column(Boolean, nullable=False)
    days_rest      = Column(SmallInteger, nullable=True)   # mirror of games.home/away_days_rest

    # Scoring by period
    pts    = Column(SmallInteger)
    pts_q1 = Column(SmallInteger)
    pts_q2 = Column(SmallInteger)
    pts_q3 = Column(SmallInteger)
    pts_q4 = Column(SmallInteger)
    pts_ot = Column(SmallInteger)

    # Shooting
    fgm      = Column(SmallInteger); fga      = Column(SmallInteger); fg_pct  = Column(Float)
    fg3m     = Column(SmallInteger); fg3a     = Column(SmallInteger); fg3_pct = Column(Float)
    ftm      = Column(SmallInteger); fta      = Column(SmallInteger); ft_pct  = Column(Float)

    # Rebounds / other
    oreb       = Column(SmallInteger)
    dreb       = Column(SmallInteger)
    reb        = Column(SmallInteger)
    ast        = Column(SmallInteger)
    stl        = Column(SmallInteger)
    blk        = Column(SmallInteger)
    tov        = Column(SmallInteger)
    pf         = Column(SmallInteger)
    plus_minus = Column(SmallInteger)

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        UniqueConstraint("game_id", "team_id", name="uq_team_box_game_team"),
        Index("ix_team_box_game_id",    "game_id"),
        Index("ix_team_box_team_id",    "team_id"),
        Index("ix_team_box_team_game",  "team_id", "game_id"),
    )

    game = relationship("Game", back_populates="team_box_scores")
    team = relationship("Team", back_populates="box_scores")


# ---------------------------------------------------------------------------
# player_box_scores_traditional
# ---------------------------------------------------------------------------
class PlayerBoxScoreTraditional(Base):
    """
    minutes_played   - decimal minutes (e.g. 34.5). NULL when DNP.
    dnp_status       - True if the player did not play.
    dnp_reason       - Freeform reason string when dnp_status=True.
                      Controlled vocabulary recommended:
                      'REST' | 'INJURY' | 'ILLNESS' | 'PERSONAL' |
                      'COACH_DECISION' | 'INACTIVE' | 'SUSPENSION'
    days_rest        - Days since this player's previous game appearance.
                      NULL for season debut. 1 = back-to-back.
                      Populated by the data pipeline, not derived at query time.
    """
    __tablename__ = "player_box_scores_traditional"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    game_id   = Column(Integer, ForeignKey("games.id"),    nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"),  nullable=False)
    team_id   = Column(Integer, ForeignKey("teams.id"),    nullable=False)

    # Availability
    dnp_status     = Column(Boolean,     nullable=False, default=False)
    dnp_reason     = Column(String(50),  nullable=True)   # see docstring above
    minutes_played = Column(Float,       nullable=True)   # NULL when DNP
    days_rest      = Column(SmallInteger, nullable=True)  # NULL = season debut

    # Shooting
    fgm    = Column(SmallInteger); fga    = Column(SmallInteger); fg_pct  = Column(Float)
    fg3m   = Column(SmallInteger); fg3a   = Column(SmallInteger); fg3_pct = Column(Float)
    ftm    = Column(SmallInteger); fta    = Column(SmallInteger); ft_pct  = Column(Float)

    # Counting stats
    oreb       = Column(SmallInteger)
    dreb       = Column(SmallInteger)
    reb        = Column(SmallInteger)
    ast        = Column(SmallInteger)
    stl        = Column(SmallInteger)
    blk        = Column(SmallInteger)
    tov        = Column(SmallInteger)
    pf         = Column(SmallInteger)
    pts        = Column(SmallInteger)
    plus_minus = Column(SmallInteger)
    fantasy_pts = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_trad_game_player"),
        # Primary lookup patterns for 5yr historical queries
        Index("ix_ptrad_player_id",         "player_id"),
        Index("ix_ptrad_game_id",            "game_id"),
        Index("ix_ptrad_team_id",            "team_id"),
        Index("ix_ptrad_player_game",        "player_id", "game_id"),
        Index("ix_ptrad_dnp_status",         "dnp_status"),
        Index("ix_ptrad_player_dnp",         "player_id", "dnp_status"),
        Index("ix_ptrad_days_rest",          "days_rest"),         # back-to-back analysis
    )

    game   = relationship("Game",   back_populates="trad_box_scores")
    player = relationship("Player", back_populates="trad_box_scores")


# ---------------------------------------------------------------------------
# player_box_scores_advanced
# ---------------------------------------------------------------------------
class PlayerBoxScoreAdvanced(Base):
    __tablename__ = "player_box_scores_advanced"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    game_id   = Column(Integer, ForeignKey("games.id"),    nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"),  nullable=False)
    team_id   = Column(Integer, ForeignKey("teams.id"),    nullable=False)

    minutes_played = Column(Float, nullable=True)

    # On/off efficiency
    off_rating = Column(Float)   # points scored per 100 possessions
    def_rating = Column(Float)   # points allowed per 100 possessions
    net_rating = Column(Float)   # off_rating - def_rating

    # Playmaking
    ast_pct    = Column(Float)   # % of teammate FGM assisted while on court
    ast_to_tov = Column(Float)   # assist-to-turnover ratio
    ast_ratio  = Column(Float)

    # Rebounding
    oreb_pct = Column(Float)
    dreb_pct = Column(Float)
    reb_pct  = Column(Float)

    # Shooting quality
    efg_pct  = Column(Float)   # effective FG% = (FGM + 0.5*3PM) / FGA
    ts_pct   = Column(Float)   # true shooting % = PTS / (2 * (FGA + 0.44*FTA))

    # Usage / pace context
    usg_pct  = Column(Float)   # % of team plays used while on court
    to_ratio = Column(Float)
    pace     = Column(Float)
    pie      = Column(Float)   # player impact estimate

    created_at = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_adv_game_player"),
        Index("ix_padv_player_id",    "player_id"),
        Index("ix_padv_game_id",      "game_id"),
        Index("ix_padv_team_id",      "team_id"),
        Index("ix_padv_player_game",  "player_id", "game_id"),
    )

    game   = relationship("Game",   back_populates="adv_box_scores")
    player = relationship("Player", back_populates="adv_box_scores")


# ---------------------------------------------------------------------------
# injury_reports
# ---------------------------------------------------------------------------
class InjuryReport(Base):
    """
    Designed for chronological tracking of status changes.

    One row = one status update for a player, not one injury.
    Multiple rows per player/game represent the status progression
    (e.g. Questionable Mon -> Doubtful Tue -> Out Wed for the same game).

    report_sequence  - integer ordering within the same (player_id, game_id)
                      pair. 1 = first report, 2 = first update, etc.
                      Allows easy "latest status" queries via MAX(report_sequence).

    data_source      - origin of the report:
                      'NBA_OFFICIAL' | 'ESPN' | 'ROTOWORLD' | 'BEAT_REPORTER' | 'TEAM'
    """
    __tablename__ = "injury_reports"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    player_id       = Column(Integer, ForeignKey("players.id"), nullable=False)
    team_id         = Column(Integer, ForeignKey("teams.id"),   nullable=False)
    game_id         = Column(Integer, ForeignKey("games.id"),   nullable=True)  # NULL = practice report

    report_date     = Column(Date,        nullable=False)
    report_sequence = Column(SmallInteger, nullable=False, default=1)
    # 'Out' | 'Doubtful' | 'Questionable' | 'Day-To-Day' | 'Available' | 'GTD'
    status          = Column(String(30),  nullable=False)
    injury_type     = Column(String(100), nullable=True)   # e.g. 'Left Knee Soreness'
    body_part       = Column(String(50),  nullable=True)   # e.g. 'Knee', 'Hamstring'
    side            = Column(String(10),  nullable=True)   # 'Left' | 'Right' | 'N/A'
    is_out             = Column(Boolean, nullable=False, default=False)  # fast filter: status == 'Out'
    is_season_ending   = Column(Boolean, nullable=False, default=False)  # torn ACL, surgery, IR, etc.
    estimated_return   = Column(Date,    nullable=True)
    notes              = Column(Text,    nullable=True)
    data_source        = Column(String(50), nullable=False)

    created_at      = Column(DateTime(timezone=True), server_default=text("now()"))

    __table_args__ = (
        # Ensure we don't double-insert the same update
        UniqueConstraint("player_id", "game_id", "report_sequence",
                         name="uq_injury_player_game_seq"),
        # Chronological status lookup: all reports for a player ordered by date
        Index("ix_injury_player_id",          "player_id"),
        Index("ix_injury_game_id",            "game_id"),
        Index("ix_injury_player_date",        "player_id", "report_date"),
        Index("ix_injury_player_game",        "player_id", "game_id"),
        # Fast filtering on status type for "who is out tonight" queries
        Index("ix_injury_status",             "status"),
        Index("ix_injury_is_out",             "is_out"),
        Index("ix_injury_is_season_ending",   "is_season_ending"),
        Index("ix_injury_report_date_status", "report_date", "status"),
    )

    player = relationship("Player", back_populates="injury_reports")
    game   = relationship("Game",   back_populates="injury_reports")


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------
def run_migration():
    print("\n[1/3] Testing connection...")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("      Connection successful.")

    print("\n[2/3] Creating tables (checkfirst=True  - safe to re-run)...")
    Base.metadata.create_all(engine, checkfirst=True)

    print("\n[3/3] Tables confirmed in schema:")
    for table in Base.metadata.sorted_tables:
        col_count = len(table.columns)
        idx_count = len(table.indexes)
        print(f"      + {table.name:<45} {col_count:>2} columns  {idx_count:>2} indexes")

    print("\nMigration complete.\n")


if __name__ == "__main__":
    run_migration()
