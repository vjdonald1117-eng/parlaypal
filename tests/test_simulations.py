"""
Unit tests for Monte Carlo helpers in services/simulations.py and a smoke
integration test for the versioned refresh-all API.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from services import simulations as sim


def test_effective_std_widens_when_mean_below_line() -> None:
    """UNDER-side context widens std when sim mean sits below the line ratio."""
    std = 4.0
    line = 20.0
    threshold = line * sim.UNDER_CONTEXT_LINE_RATIO
    # At or above threshold: no widening
    tight = sim.effective_std_for_prop(std, sim_mean=threshold + 0.2, line=line)
    assert tight == std
    # Strictly below threshold: widened std
    wide = sim.effective_std_for_prop(std, sim_mean=threshold - 0.5, line=line)
    assert wide == pytest.approx(std * sim.UNDER_CONTEXT_STD_MULT)


def test_run_simulation_shape_bounds_and_metrics() -> None:
    """run_simulation returns sims with expected length, non-negative draws, and summary keys."""
    adj_mean = 22.0
    std_dev = 4.0
    line = 20.5
    n = 1_500
    out = sim.run_simulation(adj_mean, std_dev, line, n_sims=n)
    sims = out["sims"]
    assert isinstance(sims, np.ndarray)
    assert sims.shape == (n,)
    assert np.all(sims >= 0.0)
    assert np.isfinite(sims).all()
    for key in ("over_pct", "under_pct", "best_side", "best_pct", "verdict", "p50"):
        assert key in out
    assert out["best_side"] in ("OVER", "UNDER")


def test_apply_game_script_failure_drawdown_runs_and_clips() -> None:
    """Asymmetric UNDER path: in-place drawdown runs without error and keeps non-negative values."""
    n = 10_000
    arr = np.random.default_rng(42).uniform(8.0, 35.0, size=n).astype(np.float64)
    before = arr.copy()
    sim._apply_game_script_failure_drawdown(arr)
    assert arr.shape == before.shape
    assert np.all(arr >= 0.0)
    assert np.all(arr <= before + 1e-9)


def test_summarize_prop_under_side_when_line_high() -> None:
    """When almost all mass is below the line, best_side should favor UNDER logic."""
    rng = np.random.default_rng(7)
    draws = rng.normal(loc=5.0, scale=1.0, size=3_000)
    np.clip(draws, 0.0, None, out=draws)
    summary = sim.summarize_prop_from_samples(draws, line=50.0)
    assert summary["best_side"] == "UNDER"
    assert summary["under_pct"] > summary["over_pct"]


@pytest.fixture
def api_client():
    """FastAPI app + TestClient (httpx transport)."""
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from api import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_refresh_running():
    """Avoid 409 from a stuck _refresh_running flag between tests."""
    import api as api_module

    api_module._refresh_running = False
    yield
    api_module._refresh_running = False


def test_v1_refresh_all_returns_200_and_job_id(api_client) -> None:
    """POST /api/v1/refresh-all must enqueue a job and return a UUID-shaped job_id."""
    r = api_client.post(
        "/api/v1/refresh-all",
        params={"day": "today", "n_sims": 5000, "fresh_odds": "false"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("status") == "processing"
    job_id = data.get("job_id")
    assert isinstance(job_id, str) and job_id
    uuid.UUID(job_id)  # raises if malformed
