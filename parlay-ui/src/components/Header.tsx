type Stat =
  | 'all'
  | 'pts'
  | 'reb'
  | 'ast'
  | 'stl'
  | 'blk'
  | 'fg3'
  | 'ml'
  | 'gproj'
  | 'parlays'

type Page = 'projections' | 'savedPicks' | 'leaderboard' | 'pastProjections' | 'calibration'

export function Header({
  isLoading,
  gameDate,
  selectedStat,
  simulatedAt,
  currentPage,
  onNavigate,
}: {
  isLoading: boolean
  gameDate: string | null
  selectedStat: Stat
  simulatedAt: string | null
  currentPage: Page
  onNavigate: (page: Page) => void
}) {
  return (
    <header className="border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="min-w-0 space-y-1">
          <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">Parlay Pal - Live NBA Projections</h1>
          <p className="text-sm font-medium text-zinc-400 sm:text-base">
            {isLoading && !gameDate
              ? 'Loading cached projections…'
              : gameDate
                ? selectedStat === 'all'
                  ? `Top OVER / UNDER picks across all cached stats — ${gameDate}`
                  : selectedStat === 'parlays'
                    ? `Best 3-leg parlays (EV-optimized) — ${gameDate}`
                    : selectedStat === 'ml'
                      ? `Games — market ML & spread — ${gameDate}`
                      : selectedStat === 'gproj'
                        ? `Games — sim points, totals, and spread cover — ${gameDate}`
                        : `Showing cached ${selectedStat.toUpperCase()} projections for ${gameDate}`
                : 'Run simulations or load cached data to see the slate date.'}
          </p>
          {simulatedAt && Number.isFinite(Date.parse(String(simulatedAt))) && (
            <p className="text-xs text-zinc-400">Simulated at: {new Date(simulatedAt).toLocaleString()}</p>
          )}
        </div>
        <nav
          className="flex flex-wrap items-center gap-2 border-t border-zinc-800 pt-3 lg:border-t-0 lg:pt-0"
          aria-label="Main"
        >
          {(
            [
              ['projections', 'Live Projections'],
              ['savedPicks', 'Saved Picks'],
              ['leaderboard', 'Leaderboard'],
              ['pastProjections', 'Past'],
              ['calibration', 'Calibration'],
            ] as const
          ).map(([page, label]) => (
            <button
              key={page}
              type="button"
              onClick={() => onNavigate(page)}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                currentPage === page
                  ? 'bg-zinc-800 text-zinc-100 ring-1 ring-zinc-600'
                  : 'text-zinc-400 hover:bg-zinc-900'
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  )
}
