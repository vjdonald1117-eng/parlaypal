"""
Central SQLAlchemy configuration for Parlay Pal (engine, sessions, ORM base).

prediction_log is created on import if missing (same behavior as former api.py).
"""

import os

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, text
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker

_repo_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_repo_dir, ".env"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=3,
    max_overflow=5,
    connect_args={"sslmode": "require", "options": "-c timezone=utc"},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


class PredictionLog(Base):
    """Stores every PlayerSim result produced by /api/refresh."""

    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    logged_at = Column(DateTime(timezone=True), server_default=text("now()"))
    game_date = Column(String(10), nullable=False, index=True)  # "2025-04-01"
    player_name = Column(String(100), nullable=False, index=True)
    team_abbr = Column(String(5), nullable=False)
    opponent = Column(String(5), nullable=False)
    stat = Column(String(5), nullable=False, index=True)
    line = Column(Float, nullable=False)
    line_source = Column(String(5), nullable=False)  # DK | FD | EST
    heuristic_mean = Column(Float, nullable=False)
    ml_mean = Column(Float, nullable=True)  # NULL if XGB unavailable
    over_pct = Column(Float, nullable=False)
    under_pct = Column(Float, nullable=False)
    best_side = Column(String(6), nullable=False)  # OVER | UNDER
    best_pct = Column(Float, nullable=False)
    ev_per_110 = Column(Float, nullable=False)
    verdict = Column(String(30), nullable=False)
    ensemble_lock = Column(Boolean, nullable=False, default=False)
    # Filled in after the game resolves (NULL until then)
    actual_value = Column(Float, nullable=True)
    hit = Column(Boolean, nullable=True)  # True = prediction correct
    explanation_tags = Column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="Serialized list[str] from PlayerSim (pace, defense, NMU, etc.)",
    )


Base.metadata.create_all(engine)


def _ensure_prediction_log_explanation_tags() -> None:
    """
    create_all() does not add new columns to existing tables. Ensure JSONB column exists
    for deployments that predated explanation_tags (PostgreSQL / Supabase).
    """
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE prediction_log "
                    "ADD COLUMN IF NOT EXISTS explanation_tags JSONB "
                    "NOT NULL DEFAULT '[]'::jsonb"
                )
            )
    except Exception:
        pass


_ensure_prediction_log_explanation_tags()
