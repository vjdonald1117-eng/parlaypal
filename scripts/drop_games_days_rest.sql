-- Drop legacy games.days_rest column (safe/no-op if already removed).
-- Canonical rest fields are games.home_days_rest and games.away_days_rest.

ALTER TABLE public.games
DROP COLUMN IF EXISTS days_rest;

