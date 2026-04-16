"""
models/monte_carlo_sim.py (compatibility shim)
==============================================
CLI and legacy imports forward to ``services.simulations`` where the Monte Carlo
implementation lives.

Run as a module from repo root:
  python -m models.monte_carlo_sim ...
"""

from services.simulations import *  # noqa: F403,F401

if __name__ == "__main__":
    main()
