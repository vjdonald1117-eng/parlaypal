import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'

import { Header } from './components/Header'
import {
  ALL_PROP_STATS,
  LINE_ADJUST_STEP,
  ProjectionTable,
  type ProjectionRow,
  getEv,
  getPlayerLabel,
  getWinPct,
  sortProjectionRowsByEdgeDesc,
  toNumber,
  toPercent,
} from './components/ProjectionTable'

type Stat = 'all' | 'pts' | 'reb' | 'ast' | 'stl' | 'blk' | 'fg3' | 'ml' | 'gproj' | 'parlays'

type ProjectionsApiEnvelope = {
  game_date?: string
  timestamp?: string
  day?: string
  results?: ProjectionRow[]
  overs?: ProjectionRow[]
  unders?: ProjectionRow[]
  cached_stats?: string[]
  /** API returns this instead of 404 when the slate has no cached joint run */
  cache_miss?: boolean
  message?: string
}

type RefreshJobStart = {
  job_id: string
  status: string
}

type RefreshJobStatus = {
  job_id: string
  status: string
  result?: unknown
  error?: string
}

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

type EvParlayLeg = {
  game_id: number
  player_id: number
  player_name: string
  stat: string
  line?: number | null
  side: 'OVER' | 'UNDER'
  matchup: string
  probability: number
  decimal_odds: number
  ev: number
}

type EvParlay = {
  legs: EvParlayLeg[]
  parlay_probability: number
  parlay_decimal_odds: number
  parlay_ev: number
}

function BestParlaysSection({
  parlays,
  gameDate,
}: {
  parlays: EvParlay[]
  gameDate: string | null
}) {
  const renderParlayCard = (p: EvParlay, idx: number) => (
    <div
      key={`ev-${idx}`}
      className="rounded-xl border border-zinc-700/90 bg-zinc-950/80 p-4 shadow-md shadow-black/20"
    >
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2 border-b border-zinc-800 pb-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Parlay {idx + 1} · EV-ranked
        </span>
        <span className="text-xs text-zinc-500">
          Odds {p.parlay_decimal_odds.toFixed(2)} · Win {toPercent(p.parlay_probability)} · EV {p.parlay_ev.toFixed(4)}
        </span>
      </div>
      <ol className="space-y-2.5">
        {p.legs.map((L, legIdx) => (
          <li
            key={`${L.player_id}-${L.stat}-${legIdx}`}
            className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-lg border border-zinc-800/80 bg-black/30 px-3 py-2 text-sm"
          >
            <span className="font-semibold text-zinc-200">{L.player_name}</span>
            <span className="text-zinc-500">{L.matchup}</span>
            <span className="rounded bg-zinc-800 px-2 py-0.5 text-xs font-semibold text-amber-100/90">
              {L.side} {String(L.stat).toUpperCase()} {L.line != null ? Number(L.line).toFixed(1) : '—'}
            </span>
            <span className="ml-auto tabular-nums text-xs font-medium text-emerald-400/90">
              {toPercent(L.probability)} · odds {L.decimal_odds.toFixed(2)} · EV {L.ev.toFixed(4)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  )

  const total = parlays.length

  return (
    <section className="space-y-8">
      <div className="rounded-xl border border-zinc-800 bg-zinc-950/90 p-4">
        <h2 className="text-lg font-semibold text-zinc-100">Best 3-leg parlays</h2>
        <p className="mt-1 text-sm text-zinc-400">
          Built from top positive-EV individual props (prediction engine: possession_sim_v1), then ranked by full 3-leg parlay EV.
          <span className="text-zinc-500">
            {' '}
            Returns exactly the top 5 most profitable combinations.
          </span>
        </p>
        <p className="mt-2 text-xs text-zinc-500">
          Slate {gameDate ?? '—'} · {total} parlay{total === 1 ? '' : 's'} built
        </p>
      </div>

      {parlays.length > 0 && <div className="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">{parlays.map((p, i) => renderParlayCard(p, i))}</div>}

      {total === 0 && (
        <p className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-100/90">
          No positive-EV 3-leg combinations found for this slate. Run{' '}
          <span className="font-semibold">Simulations</span> for this date, then open this view again.
        </p>
      )}
    </section>
  )
}

/**
 * Today's calendar date in America/New_York as YYYY-MM-DD.
 * Uses formatToParts — `en-CA` + format() is inconsistent on some Windows browsers
 * and can break string compares against API game_date.
 */
function nyCalendarDateYmd(): string {
  const d = new Date()
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(d)
  const get = (t: Intl.DateTimeFormatPartTypes) => parts.find((p) => p.type === t)?.value ?? ''
  const y = get('year')
  const mo = get('month').padStart(2, '0')
  const da = get('day').padStart(2, '0')
  return `${y}-${mo}-${da}`
}

/** Calendar YYYY-MM-DD + n days (pure date; aligns with API ny_today ± 1 day). */
function addCalendarDaysYmd(ymd: string, days: number): string {
  const m = ymd.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/)
  if (!m) return ymd
  const y = Number(m[1])
  const mo = Number(m[2])
  const d = Number(m[3])
  const utc = Date.UTC(y, mo - 1, d + days)
  const dt = new Date(utc)
  const yy = dt.getUTCFullYear()
  const ms = String(dt.getUTCMonth() + 1).padStart(2, '0')
  const da = String(dt.getUTCDate()).padStart(2, '0')
  return `${yy}-${ms}-${da}`
}

/** Slate date string for Date dropdown (matches GET /api/projections?day=…). */
function slateDateYmdForDayChoice(day: 'today' | 'tomorrow'): string {
  const ny = nyCalendarDateYmd()
  return day === 'today' ? ny : addCalendarDaysYmd(ny, 1)
}

/** Coerce API game_date (date string or ISO datetime) to YYYY-MM-DD for slate comparisons. */
function normalizeGameDateYmd(raw: unknown): string {
  if (raw == null) return ''
  const s = String(raw).trim()
  const m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/)
  if (!m) return s.slice(0, 10)
  return `${m[1]}-${m[2].padStart(2, '0')}-${m[3].padStart(2, '0')}`
}

/** True if game slate is strictly before NY calendar day (both normalized YYYY-MM-DD). */
function isBeforeCalendarDate(gameYmd: string, nyYmd: string): boolean {
  return normalizeGameDateYmd(gameYmd) < normalizeGameDateYmd(nyYmd)
}

type Page = 'projections' | 'savedPicks' | 'leaderboard' | 'pastProjections' | 'calibration'

/** FastAPI returns `detail` as a string or a list of validation errors */
function describeFastApiDetail(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null
  const d = (body as { detail?: unknown }).detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) {
    return d
      .map((item) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const loc = Array.isArray((item as { loc?: unknown }).loc)
            ? (item as { loc: (string | number)[] }).loc.join('.')
            : ''
          const msg = String((item as { msg: string }).msg)
          return loc ? `${loc}: ${msg}` : msg
        }
        return JSON.stringify(item)
      })
      .join('; ')
  }
  return null
}

type CalibrationCell = {
  bucket_lo: number
  bucket_hi: number
  side: string
  n: number
  hits: number
  predicted_mid_pct: number
  actual_hit_pct: number
  gap_pct: number
}

type CalibrationResponse = {
  min_per_cell: number
  cells: CalibrationCell[]
  totals: { n: number; hits: number; overall_hit_pct: number | null }
}

type HistoryRecord = {
  id: number
  logged_at?: string | null
  game_date: string
  player_name: string
  team_abbr: string
  opponent: string
  stat: string
  line: number
  line_source: string
  best_side: string
  ev_per_110: number
  verdict: string
  win_probability: number
  ensemble_lock: boolean
  actual_value?: number | null
  hit: boolean | null
}

const MIN_LEADERBOARD_RESOLVED = 3

/** Props only; leaderboard uses the same persisted top-board rows as Past (not ML / win rows). */
const LEADERBOARD_PROP_STATS = new Set(['pts', 'reb', 'ast', 'stl', 'blk', 'fg3'])
const STAT_SORT_ORDER = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3'] as const

function isLeaderboardPropStat(stat: string): boolean {
  return LEADERBOARD_PROP_STATS.has(stat.toLowerCase())
}

function formatLeaderboardStatLabel(stat: string): string {
  const s = stat.toLowerCase()
  if (s === 'pts') return 'PTS'
  if (s === 'reb') return 'REB'
  if (s === 'ast') return 'AST'
  if (s === 'stl') return 'STL'
  if (s === 'blk') return 'BLK'
  if (s === 'fg3') return '3PM'
  return stat.toUpperCase().slice(0, 4)
}

function statSortKey(stat: string): number {
  const idx = STAT_SORT_ORDER.indexOf(stat.toLowerCase() as (typeof STAT_SORT_ORDER)[number])
  return idx >= 0 ? idx : 99
}

/**
 * Days where every saved-board pick for that player is graded and all hit (e.g. 3/3 PTS+REB+AST).
 */
function countPerfectSavedDays(records: HistoryRecord[]): number {
  const byDate = new Map<string, HistoryRecord[]>()
  for (const r of records) {
    const d = normalizeGameDateYmd(r.game_date)
    const arr = byDate.get(d) ?? []
    arr.push(r)
    byDate.set(d, arr)
  }
  let n = 0
  for (const dayRows of byDate.values()) {
    if (dayRows.length === 0) continue
    if (dayRows.some((r) => r.hit === null)) continue
    if (dayRows.every((r) => r.hit === true)) n++
  }
  return n
}

function groupSavedPicksByDateDesc(records: HistoryRecord[]): [string, HistoryRecord[]][] {
  const byDate = new Map<string, HistoryRecord[]>()
  for (const r of records) {
    const d = normalizeGameDateYmd(r.game_date)
    const arr = byDate.get(d) ?? []
    arr.push(r)
    byDate.set(d, arr)
  }
  for (const arr of byDate.values()) {
    arr.sort((a, b) => statSortKey(a.stat) - statSortKey(b.stat))
  }
  return Array.from(byDate.entries()).sort((a, b) => b[0].localeCompare(a[0]))
}

function dedupeHistoryRecords(rows: HistoryRecord[]): HistoryRecord[] {
  return Array.from(
    rows
      .reduce((acc, r) => {
        const key = `${normalizeGameDateYmd(r.game_date)}|${r.player_name}|${r.best_side}|${r.stat}`
        const existing = acc.get(key)
        const rTime = r.logged_at ? Date.parse(r.logged_at) : 0
        const eTime = existing?.logged_at ? Date.parse(existing.logged_at) : 0
        if (!existing || rTime > eTime) {
          acc.set(key, r)
        }
        return acc
      }, new Map<string, HistoryRecord>())
      .values(),
  )
}

function topNByWinProbability(rows: HistoryRecord[], n: number): HistoryRecord[] {
  return rows
    .slice()
    .sort((a, b) => (b.win_probability ?? 0) - (a.win_probability ?? 0))
    .slice(0, n)
}

function sortHistoryByHitThenEdge(rows: HistoryRecord[]): HistoryRecord[] {
  return rows.slice().sort((a, b) => {
    const rank = (hit: boolean | null) => (hit === true ? 0 : hit === false ? 1 : 2)
    const ra = rank(a.hit)
    const rb = rank(b.hit)
    if (ra !== rb) return ra - rb
    return (b.win_probability ?? 0) - (a.win_probability ?? 0)
  })
}

/** Graded hit/miss uses green/red; ungraded past rows use amber until box scores resolve. */
function hitResultBulletClass(hit: boolean | null): string {
  if (hit === true) return 'bg-emerald-400'
  if (hit === false) return 'bg-rose-500'
  return 'bg-amber-400'
}

function LeaderboardPickBox({ record }: { record: HistoryRecord }) {
  const hit = record.hit
  const hasActual = record.actual_value != null && !Number.isNaN(Number(record.actual_value))
  const displayNum = hasActual ? toNumber(Number(record.actual_value)) : toNumber(record.line)
  const color =
    hit === true
      ? 'border-emerald-500/80 bg-emerald-500/15'
      : hit === false
        ? 'border-rose-500/80 bg-rose-500/15'
        : 'border-amber-500/60 bg-amber-500/10'
  return (
    <div
      className={`shrink-0 rounded-md border px-2.5 py-1.5 ${color}`}
      title={
        hasActual
          ? `Line ${toNumber(record.line)} · actual ${toNumber(Number(record.actual_value))}`
          : `Line ${toNumber(record.line)}`
      }
    >
      <div className="text-[10px] font-semibold uppercase leading-tight tracking-wide text-zinc-400">
        {formatLeaderboardStatLabel(record.stat)}
      </div>
      <div className="text-lg font-semibold tabular-nums leading-tight text-zinc-100">{displayNum}</div>
    </div>
  )
}

/** Match api._mixed_top_props: top-N with up to `perStat` per category first so 3PM is not buried by PTS. */
function mixedTopPropsForAllBoard(
  sideRows: ProjectionRow[],
  finalN: number,
  perStat: number,
): ProjectionRow[] {
  const byStat: Record<string, ProjectionRow[]> = {}
  for (const s of ALL_PROP_STATS) {
    byStat[s] = []
  }
  for (const r of sideRows) {
    const st = String(r.stat ?? '').toLowerCase()
    if (st in byStat) byStat[st].push(r)
  }
  for (const s of ALL_PROP_STATS) {
    byStat[s].sort(sortProjectionRowsByEdgeDesc)
  }
  const picked: ProjectionRow[] = []
  const seen = new Set<string>()
  const dedupeKey = (row: ProjectionRow) =>
    `${getPlayerLabel(row)}|${String(row.stat ?? '').toLowerCase()}|${String(row.best_side ?? '')}|${row.line ?? ''}`

  outer: for (const st of ALL_PROP_STATS) {
    for (const r of byStat[st].slice(0, perStat)) {
      const k = dedupeKey(r)
      if (!seen.has(k)) {
        seen.add(k)
        picked.push(r)
      }
      if (picked.length >= finalN) break outer
    }
  }

  if (picked.length < finalN) {
    const rest: ProjectionRow[] = []
    for (const st of ALL_PROP_STATS) {
      for (const r of byStat[st].slice(perStat)) {
        const k = dedupeKey(r)
        if (!seen.has(k)) rest.push(r)
      }
    }
    rest.sort(sortProjectionRowsByEdgeDesc)
    for (const r of rest) {
      if (picked.length >= finalN) break
      const k = dedupeKey(r)
      if (!seen.has(k)) {
        seen.add(k)
        picked.push(r)
      }
    }
  }

  picked.sort(sortProjectionRowsByEdgeDesc)
  return picked.slice(0, finalN)
}

/**
 * GET /api/projections returns { results: [...] }. Split by best_side (OVER | UNDER).
 */
function normalizeResponse(data: unknown): { overs: ProjectionRow[]; unders: ProjectionRow[] } {
  let rows: ProjectionRow[] = []

  if (Array.isArray(data)) {
    rows = data
  } else if (data && typeof data === 'object') {
    const env = data as ProjectionsApiEnvelope
    if (Array.isArray(env.results)) {
      rows = env.results
    } else {
      return {
        overs: env.overs ?? [],
        unders: env.unders ?? [],
      }
    }
  }

  const side = (r: ProjectionRow) => String(r.best_side ?? '').toUpperCase()
  const overs = rows.filter((r) => side(r) === 'OVER')
  const unders = rows.filter((r) => side(r) === 'UNDER')

  const byWin = (a: ProjectionRow, b: ProjectionRow) =>
    (getWinPct(b) ?? 0) - (getWinPct(a) ?? 0)

  return {
    overs: [...overs].sort(byWin),
    unders: [...unders].sort(byWin),
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object'
}

function App() {
  const [selectedStat, setSelectedStat] = useState<Stat>('pts')
  const [selectedDay, setSelectedDay] = useState<'today' | 'tomorrow'>('tomorrow')
  const [overs, setOvers] = useState<ProjectionRow[]>([])
  const [unders, setUnders] = useState<ProjectionRow[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [gameDate, setGameDate] = useState<string | null>(null)
  const [simulatedAt, setSimulatedAt] = useState<string | null>(null)
  const [currentPage, setCurrentPage] = useState<Page>('projections')
  const [savedOver, setSavedOver] = useState<ProjectionRow | null>(null)
  const [savedUnder, setSavedUnder] = useState<ProjectionRow | null>(null)
  const [simDepth, setSimDepth] = useState<'3000' | '5000' | '10000'>('5000')
  const [refreshElapsedSec, setRefreshElapsedSec] = useState(0)
  /** When true, POST /api/refresh-all bypasses local odds_cache_*.json and refetches DK/FD from The Odds API */
  const [freshOddsOnSim, setFreshOddsOnSim] = useState(true)
  const [isSyncingGameLines, setIsSyncingGameLines] = useState(false)
  /** Steps of LINE_ADJUST_STEP from book line per row key (OVER|UNDER board) */
  const [lineDeltaByKey, setLineDeltaByKey] = useState<Record<string, number>>({})
  const [bestParlays, setBestParlays] = useState<EvParlay[]>([])
  const [simGameRows, setSimGameRows] = useState<ProjectionRow[]>([])
  const [historyRecords, setHistoryRecords] = useState<HistoryRecord[]>([])
  const [isHistoryLoading, setIsHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [selectedHistoryDate, setSelectedHistoryDate] = useState<string | null>(null)
  const [showAllPicksForDate, setShowAllPicksForDate] = useState(false)
  const [showUngradedPastOnly, setShowUngradedPastOnly] = useState(false)
  const [combinedCachedStats, setCombinedCachedStats] = useState<string[]>([])
  const [expandedLeaderboardPlayer, setExpandedLeaderboardPlayer] = useState<string | null>(null)
  const [calibrationData, setCalibrationData] = useState<CalibrationResponse | null>(null)
  const [calibrationLoading, setCalibrationLoading] = useState(false)
  const [calibrationError, setCalibrationError] = useState<string | null>(null)

  const handleUpdateHistory = async () => {
    setIsHistoryLoading(true)
    setHistoryError(null)
    try {
      // Grade pending rows across all dates so Past Projections can show hit/miss after refresh.
      const response = await fetch(`${API_BASE}/api/update-history`, {
        method: 'POST',
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(`Failed to update history: ${response.status} ${text}`)
      }
      await fetchHistory()
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unexpected error while updating history.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  const handleRegradeSelectedDate = async () => {
    if (!selectedHistoryDate) return
    setIsHistoryLoading(true)
    setHistoryError(null)
    try {
      const url = `${API_BASE}/api/update-history?game_date=${encodeURIComponent(selectedHistoryDate)}&force_regrade=true&limit=20000`
      const response = await fetch(url, { method: 'POST' })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(`Re-grade failed: ${response.status} ${text}`)
      }
      await fetchHistory()
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unexpected error while re-grading this slate.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  const fetchCalibration = useCallback(async () => {
    setCalibrationLoading(true)
    setCalibrationError(null)
    try {
      const response = await fetch(`${API_BASE}/api/calibration?min_per_cell=3`)
      if (!response.ok) {
        throw new Error(`Calibration failed: ${response.status}`)
      }
      const data = (await response.json()) as CalibrationResponse
      setCalibrationData(data)
    } catch (err) {
      setCalibrationError(err instanceof Error ? err.message : 'Failed to load calibration.')
      setCalibrationData(null)
    } finally {
      setCalibrationLoading(false)
    }
  }, [])

  const hasData = useMemo(
    () => overs.length > 0 || unders.length > 0 || bestParlays.length > 0,
    [overs.length, unders.length, bestParlays.length],
  )

  const loadSimGames = useCallback(async () => {
    try {
      const response = await fetch(
        `${API_BASE}/api/sim-games?day=${encodeURIComponent(selectedDay)}`,
      )
      if (!response.ok) {
        setSimGameRows([])
        return
      }
      const data: unknown = await response.json()
      const rows =
        data && typeof data === 'object' && 'results' in data
          ? (data as { results: ProjectionRow[] }).results
          : []
      setSimGameRows(Array.isArray(rows) ? rows : [])
    } catch {
      setSimGameRows([])
    }
  }, [selectedDay])

  const fetchProjections = useCallback(async (stat: Stat) => {
    setIsLoading(true)
    setError(null)
    setCombinedCachedStats([])
    setBestParlays([])
    try {
      if (stat === 'parlays') {
        const response = await fetch(`${API_BASE}/api/parlays-best?day=${encodeURIComponent(selectedDay)}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch EV parlays: ${response.status}`)
        }
        const data = (await response.json()) as {
          game_date?: string
          timestamp?: string
          results?: EvParlay[]
        }
        setGameDate(typeof data.game_date === 'string' && data.game_date.length > 0 ? data.game_date : null)
        setSimulatedAt(typeof data.timestamp === 'string' && data.timestamp.length > 0 ? data.timestamp : null)
        setOvers([])
        setUnders([])
        const parlays = Array.isArray(data.results) ? data.results : []
        setBestParlays(parlays)
        if (parlays.length === 0) {
          setError('No positive-EV parlays found for this slate. Run Simulations and try again.')
        }
        return
      }

      if (stat === 'all') {
        // Merge from the same per-stat GET /api/projections calls the rest of the app uses.
        // Avoids /api/projections-combined (404 if the running server is older or routing differs).
        const payloads = await Promise.all(
          ALL_PROP_STATS.map(async (s) => {
            const r = await fetch(
              `${API_BASE}/api/projections?stat=${encodeURIComponent(s)}&day=${encodeURIComponent(selectedDay)}`,
            )
            if (!r.ok) return null
            return (await r.json()) as ProjectionsApiEnvelope
          }),
        )

        const cachedStats: string[] = []
        let latestTs: string | null = null
        let resolvedGameDate: string | null = null
        const allOver: ProjectionRow[] = []
        const allUnder: ProjectionRow[] = []

        for (let i = 0; i < ALL_PROP_STATS.length; i++) {
          const data = payloads[i]
          if (!data || !Array.isArray(data.results) || data.results.length === 0) {
            continue
          }
          const st = ALL_PROP_STATS[i]
          cachedStats.push(st)
          const gd = data.game_date
          if (typeof gd === 'string' && gd.length > 0 && !resolvedGameDate) {
            resolvedGameDate = gd
          }
          const ts = data.timestamp
          if (typeof ts === 'string' && ts.length > 0 && (!latestTs || ts > latestTs)) {
            latestTs = ts
          }
          for (const raw of data.results) {
            const row: ProjectionRow = { ...raw, stat: st }
            const side = String(row.best_side ?? '').toUpperCase()
            if (side === 'OVER') allOver.push(row)
            else if (side === 'UNDER') allUnder.push(row)
          }
        }

        setGameDate(resolvedGameDate)
        setSimulatedAt(latestTs)
        setCombinedCachedStats(cachedStats)

        if (cachedStats.length === 0) {
          setOvers([])
          setUnders([])
          setGameDate(slateDateYmdForDayChoice(selectedDay))
          setSimulatedAt(null)
          setError(
            'No projections for this slate (memory cache empty and nothing in history DB for this date). Pick the same Date you used for Run Simulations (Today vs Tomorrow), or run simulations again with Today selected.',
          )
          return
        }

        setOvers(mixedTopPropsForAllBoard(allOver, 10, 2))
        setUnders(mixedTopPropsForAllBoard(allUnder, 10, 2))
        return
      }
      if (stat === 'ml') {
        const response = await fetch(
          `${API_BASE}/api/moneylines?day=${encodeURIComponent(selectedDay)}`,
        )
        if (!response.ok) {
          throw new Error(`Failed to fetch moneylines: ${response.status}`)
        }
        const data: unknown = await response.json()
        if (data && typeof data === 'object' && 'game_date' in data) {
          const gd = (data as ProjectionsApiEnvelope).game_date
          setGameDate(typeof gd === 'string' && gd.length > 0 ? gd : null)
          const ts = (data as ProjectionsApiEnvelope).timestamp
          setSimulatedAt(typeof ts === 'string' && ts.length > 0 ? ts : null)
        } else {
          setGameDate(null)
          setSimulatedAt(null)
        }
        const env = data as ProjectionsApiEnvelope & { warning?: string }
        const o = Array.isArray(env.overs) ? env.overs : []
        const u = Array.isArray(env.unders) ? env.unders : []
        setOvers(o)
        setUnders(u)
        if (o.length === 0 && u.length === 0 && typeof env.warning === 'string' && env.warning.length > 0) {
          setError(env.warning)
        }
        return
      }
      if (stat === 'gproj') {
        const response = await fetch(
          `${API_BASE}/api/sim-games?day=${encodeURIComponent(selectedDay)}`,
        )
        if (!response.ok) {
          throw new Error(`Failed to fetch game projections: ${response.status}`)
        }
        const data: unknown = await response.json()
        if (data && typeof data === 'object' && 'game_date' in data) {
          const gd = (data as ProjectionsApiEnvelope).game_date
          setGameDate(typeof gd === 'string' && gd.length > 0 ? gd : null)
          const ts = (data as ProjectionsApiEnvelope).timestamp
          setSimulatedAt(typeof ts === 'string' && ts.length > 0 ? ts : null)
        } else {
          setGameDate(null)
          setSimulatedAt(null)
        }
        const env = data as ProjectionsApiEnvelope
        const rows = Array.isArray(env.results) ? env.results : []
        setOvers(rows)
        setUnders([])
        if (rows.length === 0) {
          setError('No cached game projections for this date. Click Run Simulations to generate them.')
        }
        return
      }

      const response = await fetch(
        `${API_BASE}/api/projections?stat=${encodeURIComponent(stat)}&day=${encodeURIComponent(selectedDay)}`,
      )
      if (!response.ok) {
        const text = await response.text()
        let detail = `HTTP ${response.status}`
        try {
          const parsed = describeFastApiDetail(JSON.parse(text) as unknown)
          if (parsed) detail = parsed
        } catch {
          if (text.trim()) detail = text.trim().slice(0, 400)
        }
        throw new Error(detail)
      }

      const data: unknown = await response.json()
      const env = data as ProjectionsApiEnvelope
      if (data && typeof data === 'object' && 'game_date' in data) {
        const gd = env.game_date
        setGameDate(typeof gd === 'string' && gd.length > 0 ? gd : null)
        const ts = env.timestamp
        setSimulatedAt(typeof ts === 'string' && ts.length > 0 ? ts : null)
      } else {
        setGameDate(null)
        setSimulatedAt(null)
      }
      if (env.cache_miss) {
        setOvers([])
        setUnders([])
        const dayLabel = selectedDay === 'today' ? 'Today' : 'Tomorrow'
        setError(
          env.message ??
            `No cached simulation for ${dayLabel} (${env.game_date ?? 'this slate'}). Set Date to match the slate you refreshed, or click Run Simulations with this date selected.`,
        )
        return
      }
      const normalized = normalizeResponse(data)
      setOvers(normalized.overs)
      setUnders(normalized.unders)
    } catch (err) {
      const raw = err instanceof Error ? err.message : 'Unexpected error while loading projections.'
      if (raw === 'Failed to fetch' || raw.includes('NetworkError') || raw.includes('Load failed')) {
        setError(
          'Cannot reach the API (network). Open the app at http://127.0.0.1:5173/ with uvicorn on 127.0.0.1:8000, or set VITE_API_BASE=http://127.0.0.1:8000 — avoid localhost:8000 on Windows if you see this.',
        )
      } else {
        setError(raw)
      }
    } finally {
      setIsLoading(false)
    }
  }, [selectedDay])

  const handleLineDeltaStep = useCallback((rowKey: string, dir: -1 | 1, baseLine: number) => {
    setLineDeltaByKey((prev) => {
      const cur = prev[rowKey] ?? 0
      const next = cur + dir
      const newLine = baseLine + next * LINE_ADJUST_STEP
      if (newLine < 0.25) return prev
      return { ...prev, [rowKey]: next }
    })
  }, [])

  const handleLineDeltaReset = useCallback((rowKey: string) => {
    setLineDeltaByKey((prev) => {
      if (!(rowKey in prev)) return prev
      const { [rowKey]: _, ...rest } = prev
      return rest
    })
  }, [])

  const handleRunSimulations = async () => {
    setIsRefreshing(true)
    setRefreshElapsedSec(0)
    setError(null)
    const startedAt = Date.now()
    const timerId = window.setInterval(() => {
      setRefreshElapsedSec(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)
    try {
      const nSims = Number(simDepth)
      const fresh = freshOddsOnSim ? 'true' : 'false'
      const refreshQuery = `day=${encodeURIComponent(selectedDay)}&n_sims=${encodeURIComponent(String(nSims))}&fresh_odds=${encodeURIComponent(fresh)}`
      const refreshUrlV1 = `${API_BASE}/api/v1/refresh-all?${refreshQuery}`
      const refreshUrlLegacy = `${API_BASE}/api/refresh-all?${refreshQuery}`
      let response = await fetch(refreshUrlV1, { method: 'POST' })
      if (response.status === 404) {
        // Backward compatibility while some local API runs still expose legacy /api/* routes.
        response = await fetch(refreshUrlLegacy, { method: 'POST' })
      }
      if (!response.ok) {
        const text = await response.text()
        let extra = text.trim().slice(0, 400)
        try {
          const parsed = describeFastApiDetail(JSON.parse(text) as unknown)
          if (parsed) extra = parsed
        } catch {
          /* keep text slice */
        }
        throw new Error(`Failed to run simulations: HTTP ${response.status}${extra ? ` — ${extra}` : ''}`)
      }
      const startPayload = (await response.json()) as RefreshJobStart & Record<string, unknown>
      let completedResult: unknown = null
      if (startPayload?.job_id) {
        const startMs = Date.now()
        while (true) {
          if (Date.now() - startMs > 600_000) {
            throw new Error('Simulation job timed out after 10 minutes.')
          }

          const jobId = encodeURIComponent(startPayload.job_id)
          const pollUrlV1 = `${API_BASE}/api/v1/jobs/${jobId}`
          const pollUrlLegacy = `${API_BASE}/api/jobs/${jobId}`
          let pollResponse = await fetch(pollUrlV1)
          if (pollResponse.status === 404) {
            pollResponse = await fetch(pollUrlLegacy)
          }
          if (!pollResponse.ok) {
            const text = await pollResponse.text()
            throw new Error(`Failed to check simulation job: HTTP ${pollResponse.status}${text ? ` — ${text.slice(0, 200)}` : ''}`)
          }
          const job = (await pollResponse.json()) as RefreshJobStatus
          const status = String(job.status ?? '').toLowerCase()

          if (status === 'completed') {
            completedResult = job.result
            break
          }
          if (status === 'failed') {
            const reason = typeof job.error === 'string' && job.error.trim().length > 0 ? job.error.trim() : 'Simulation job failed.'
            throw new Error(reason)
          }

          await new Promise<void>((resolve) => {
            window.setTimeout(resolve, 2000)
          })
        }
      } else if (String(startPayload?.status ?? '').toLowerCase() === 'ok') {
        // Legacy synchronous refresh endpoint returns final payload directly.
        completedResult = startPayload
      } else {
        throw new Error('Simulation request did not return a job id.')
      }

      const resultObj = isRecord(completedResult) ? completedResult : null
      if (resultObj && selectedStat !== 'ml' && selectedStat !== 'gproj' && selectedStat !== 'parlays') {
        const byStat = isRecord(resultObj.by_stat) ? resultObj.by_stat : null
        if (selectedStat === 'all' && byStat) {
          const allOver: ProjectionRow[] = []
          const allUnder: ProjectionRow[] = []
          for (const st of ALL_PROP_STATS) {
            const rows = byStat[st]
            if (!Array.isArray(rows)) continue
            for (const raw of rows as ProjectionRow[]) {
              const row: ProjectionRow = { ...raw, stat: st }
              const side = String(row.best_side ?? '').toUpperCase()
              if (side === 'OVER') allOver.push(row)
              if (side === 'UNDER') allUnder.push(row)
            }
          }
          setOvers(mixedTopPropsForAllBoard(allOver, 10, 2))
          setUnders(mixedTopPropsForAllBoard(allUnder, 10, 2))
        } else if (selectedStat !== 'all' && byStat && Array.isArray(byStat[selectedStat])) {
          const normalized = normalizeResponse({ results: byStat[selectedStat] as ProjectionRow[] })
          setOvers(normalized.overs)
          setUnders(normalized.unders)
        }
      }

      await fetchProjections(selectedStat)
      if (selectedStat !== 'ml' && selectedStat !== 'gproj') {
        await loadSimGames()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error while refreshing simulations.')
    } finally {
      window.clearInterval(timerId)
      setRefreshElapsedSec(0)
      setIsRefreshing(false)
    }
  }

  const handleSyncGameLines = async () => {
    setIsSyncingGameLines(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/api/sync-game-lines`, { method: 'POST' })
      const text = await response.text()
      if (!response.ok) {
        throw new Error(`Sync failed: HTTP ${response.status} ${text.slice(0, 200)}`)
      }
      let data: unknown = {}
      try {
        data = JSON.parse(text) as unknown
      } catch {
        /* ignore */
      }
      const rec = data as { schedule_sync_ok?: boolean; detail?: string }
      if (rec.schedule_sync_ok === false) {
        setError(rec.detail ?? 'Schedule sync exited with an error; check ODDS_API_KEY and API logs.')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error while syncing game lines.')
    } finally {
      setIsSyncingGameLines(false)
    }
  }

  useEffect(() => {
    void fetchProjections(selectedStat)
  }, [selectedStat, selectedDay, fetchProjections])

  useEffect(() => {
    setLineDeltaByKey({})
  }, [gameDate, simulatedAt, selectedStat, selectedDay])

  useEffect(() => {
    if (currentPage !== 'projections' || selectedStat === 'ml' || selectedStat === 'gproj' || selectedStat === 'parlays') {
      setSimGameRows([])
      return
    }
    void loadSimGames()
  }, [currentPage, selectedStat, loadSimGames])

  const fetchHistory = async () => {
    setIsHistoryLoading(true)
    setHistoryError(null)
    try {
      const response = await fetch(`${API_BASE}/api/history?limit=5000`)
      if (!response.ok) {
        throw new Error(`Failed to load history: ${response.status}`)
      }
      const data: unknown = await response.json()
      if (!data || typeof data !== 'object' || !('records' in data)) {
        throw new Error('Malformed history response.')
      }
      const raw = (data as { records: HistoryRecord[] }).records
      const records = raw.map((r) => {
        const ymd = normalizeGameDateYmd(r.game_date)
        return { ...r, game_date: ymd || String(r.game_date ?? '') }
      })
      setHistoryRecords(records)
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : 'Unexpected error while loading history.')
    } finally {
      setIsHistoryLoading(false)
    }
  }

  useEffect(() => {
    if (
      (currentPage === 'pastProjections' ||
        currentPage === 'savedPicks' ||
        currentPage === 'leaderboard') &&
      historyRecords.length === 0 &&
      !isHistoryLoading
    ) {
      void fetchHistory()
    }
  }, [currentPage, historyRecords.length, isHistoryLoading])

  useEffect(() => {
    if (currentPage === 'pastProjections') {
      void fetchHistory()
    }
  }, [currentPage])

  const pastGameDates = useMemo(() => {
    const nyToday = nyCalendarDateYmd()
    const seen = new Set<string>()
    for (const r of historyRecords) {
      const d = normalizeGameDateYmd(r.game_date)
      if (!/^\d{4}-\d{2}-\d{2}$/.test(d)) continue
      if (isBeforeCalendarDate(d, nyToday)) seen.add(d)
    }
    return Array.from(seen).sort((a, b) => (a < b ? 1 : -1))
  }, [historyRecords])

  useEffect(() => {
    if (pastGameDates.length === 0) {
      setSelectedHistoryDate(null)
      return
    }
    setSelectedHistoryDate((prev) => (prev && pastGameDates.includes(prev) ? prev : pastGameDates[0]))
  }, [pastGameDates])

  useEffect(() => {
    setShowAllPicksForDate(false)
    setShowUngradedPastOnly(false)
  }, [selectedHistoryDate])

  useEffect(() => {
    if (currentPage === 'calibration') {
      void fetchCalibration()
    }
  }, [currentPage, fetchCalibration])

  const pastHistoryRows = useMemo(() => {
    const nyToday = nyCalendarDateYmd()
    return historyRecords.filter((r) => {
      const d = normalizeGameDateYmd(r.game_date)
      return (
        isBeforeCalendarDate(d, nyToday) &&
        (!selectedHistoryDate || d === selectedHistoryDate)
      )
    })
  }, [historyRecords, selectedHistoryDate])

  const dedupedPastHistoryRows = useMemo(() => dedupeHistoryRecords(pastHistoryRows), [pastHistoryRows])

  const topTenPicksForSelectedDate = useMemo(
    () => sortHistoryByHitThenEdge(topNByWinProbability(dedupedPastHistoryRows, 10)),
    [dedupedPastHistoryRows],
  )

  const allPicksSortedForSelectedDate = useMemo(
    () => sortHistoryByHitThenEdge(dedupedPastHistoryRows),
    [dedupedPastHistoryRows],
  )

  const picksTableRowsBase = useMemo(
    () => (showAllPicksForDate ? allPicksSortedForSelectedDate : topTenPicksForSelectedDate),
    [showAllPicksForDate, allPicksSortedForSelectedDate, topTenPicksForSelectedDate],
  )

  const picksTableRows = useMemo(() => {
    if (!showUngradedPastOnly) return picksTableRowsBase
    return picksTableRowsBase.filter((r) => r.hit === null)
  }, [picksTableRowsBase, showUngradedPastOnly])

  /** Same “saved board” as Past Projections: at most 10 rows per past game_date (deduped). */
  const leaderboardEligibleRecords = useMemo(() => {
    const nyToday = nyCalendarDateYmd()
    const byDate = new Map<string, HistoryRecord[]>()
    for (const r of historyRecords) {
      const d = normalizeGameDateYmd(r.game_date)
      if (!isBeforeCalendarDate(d, nyToday)) continue
      const arr = byDate.get(d) ?? []
      arr.push(r)
      byDate.set(d, arr)
    }
    const out: HistoryRecord[] = []
    for (const rows of byDate.values()) {
      out.push(...topNByWinProbability(dedupeHistoryRecords(rows), 10))
    }
    return out
  }, [historyRecords])

  const yesterdayTopFive = useMemo(() => {
    const yesterdayDate = pastGameDates[0]
    if (!yesterdayDate) return { date: null as string | null, rows: [] as HistoryRecord[] }

    const rows = historyRecords.filter((r) => normalizeGameDateYmd(r.game_date) === yesterdayDate)
    const deduped = dedupeHistoryRecords(rows)
    const ordered = sortHistoryByHitThenEdge(topNByWinProbability(deduped, 5))
    return { date: yesterdayDate, rows: ordered }
  }, [historyRecords, pastGameDates])

  /** Leaderboard uses only persisted daily top-board rows (same slice as Past), props only. */
  const savedBoardPropRecords = useMemo(
    () => leaderboardEligibleRecords.filter((r) => isLeaderboardPropStat(r.stat)),
    [leaderboardEligibleRecords],
  )

  const savedBoardLeaderboard = useMemo(() => {
    const m = new Map<string, { displayName: string; records: HistoryRecord[] }>()
    for (const r of savedBoardPropRecords) {
      const key = r.player_name.trim().toLowerCase()
      if (!key) continue
      if (!m.has(key)) {
        m.set(key, { displayName: r.player_name.trim(), records: [] })
      }
      m.get(key)!.records.push(r)
    }
    const rows: Array<{
      playerKey: string
      displayName: string
      records: HistoryRecord[]
      picksByDate: [string, HistoryRecord[]][]
      hits: number
      total: number
      rate: number
      perfectDays: number
    }> = []
    for (const [playerKey, { displayName, records }] of m) {
      const resolved = records.filter((r) => r.hit !== null)
      const total = resolved.length
      if (total < MIN_LEADERBOARD_RESOLVED) continue
      const hits = resolved.filter((r) => r.hit === true).length
      rows.push({
        playerKey,
        displayName,
        records,
        picksByDate: groupSavedPicksByDateDesc(records),
        hits,
        total,
        rate: hits / total,
        perfectDays: countPerfectSavedDays(records),
      })
    }
    rows.sort((a, b) => {
      if (b.rate !== a.rate) return b.rate - a.rate
      if (b.perfectDays !== a.perfectDays) return b.perfectDays - a.perfectDays
      return b.total - a.total
    })
    return rows.slice(0, 50)
  }, [savedBoardPropRecords])

  const historyByPlayerSide = useMemo(() => {
    const map = new Map<string, { player: string; side: string; stat: string; played: number; hits: number }>()
    for (const r of leaderboardEligibleRecords) {
      if (r.hit === null) continue
      const key = `${r.player_name}|${r.best_side}|${r.stat}`
      const entry = map.get(key) ?? {
        player: r.player_name,
        side: r.best_side,
        stat: r.stat,
        played: 0,
        hits: 0,
      }
      entry.played += 1
      if (r.hit) entry.hits += 1
      map.set(key, entry)
    }
    return [...map.values()]
  }, [leaderboardEligibleRecords])

  const historicalTopOver = useMemo(() => {
    const candidates = historyByPlayerSide.filter((r) => r.side === 'OVER' && r.played >= 5)
    if (candidates.length === 0) return null
    candidates.sort((a, b) => b.hits / b.played - a.hits / a.played)
    return candidates[0]
  }, [historyByPlayerSide])

  const historicalTopUnder = useMemo(() => {
    const candidates = historyByPlayerSide.filter((r) => r.side === 'UNDER' && r.played >= 5)
    if (candidates.length === 0) return null
    candidates.sort((a, b) => b.hits / b.played - a.hits / a.played)
    return candidates[0]
  }, [historyByPlayerSide])

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      <Header
        isLoading={isLoading}
        gameDate={gameDate}
        selectedStat={selectedStat}
        simulatedAt={simulatedAt}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
      />

      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6 lg:px-8">
        {currentPage === 'projections' && (
          <>
            <section className="rounded-xl border border-zinc-800 bg-zinc-950/90 p-4 shadow-lg shadow-black/30">
              <div className="flex flex-wrap items-end gap-3">
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="stat-select" className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Stat
                  </label>
                  <select
                    id="stat-select"
                    value={selectedStat}
                    onChange={(event) => setSelectedStat(event.target.value as Stat)}
                    className="min-w-[10rem] rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none ring-zinc-500 transition focus:ring-2"
                  >
                    <option value="all">All (top 10 O/U)</option>
                    <option value="pts">Points</option>
                    <option value="reb">Rebounds</option>
                    <option value="ast">Assists</option>
                    <option value="stl">Steals</option>
                    <option value="blk">Blocks</option>
                    <option value="fg3">3-pointers made</option>
                    <option value="ml">Games (ML & spread)</option>
                    <option value="gproj">Games (sim points & cover)</option>
                    <option value="parlays">Best parlays (3-leg)</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <label htmlFor="day-select" className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                    Date
                  </label>
                  <select
                    id="day-select"
                    value={selectedDay}
                    onChange={(event) => setSelectedDay(event.target.value as 'today' | 'tomorrow')}
                    className="min-w-[9rem] rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none ring-zinc-500 transition focus:ring-2"
                  >
                    <option value="today">Today</option>
                    <option value="tomorrow">Tomorrow</option>
                  </select>
                </div>
                <div className="flex flex-col gap-1.5">
                  <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">Sim depth</span>
                  <select
                    value={simDepth}
                    onChange={(e) => setSimDepth(e.target.value as '3000' | '5000' | '10000')}
                    className="min-w-[11rem] rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none ring-zinc-500 transition focus:ring-2"
                  >
                    <option value="3000">Fast (3k)</option>
                    <option value="5000">Standard (5k)</option>
                    <option value="10000">Full (10k)</option>
                  </select>
                </div>
                <label className="flex max-w-[14rem] cursor-pointer items-start gap-2 text-xs leading-snug text-zinc-400">
                  <input
                    type="checkbox"
                    checked={freshOddsOnSim}
                    onChange={(e) => setFreshOddsOnSim(e.target.checked)}
                    className="mt-0.5 rounded border-zinc-600 bg-zinc-950 text-zinc-300 focus:ring-zinc-500"
                  />
                  <span>
                    Fresh player props (DK/FD via Odds API). Uncheck to reuse the 30‑minute local cache and save API
                    requests. Team spread/total in the database is refreshed on every Run Simulations and via Update
                    game lines.
                  </span>
                </label>
                <button
                  type="button"
                  onClick={handleSyncGameLines}
                  disabled={isRefreshing || isSyncingGameLines}
                  title="Writes Vegas spread and total into the database from the Odds API (today + next 2 ET days). No simulation."
                  className="inline-flex shrink-0 items-center justify-center rounded-lg border border-zinc-600 bg-zinc-900 px-4 py-2.5 text-sm font-medium text-zinc-100 transition hover:border-zinc-500 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isSyncingGameLines ? 'Syncing…' : 'Update game lines'}
                </button>
                <button
                  type="button"
                  onClick={handleRunSimulations}
                  disabled={isRefreshing}
                  title={
                    selectedStat === 'ml'
                      ? 'Runs joint prop + sim game ML, then reloads book ML & spread from the odds API.'
                      : selectedStat === 'gproj'
                        ? 'Runs joint simulation and refreshes model game projections (team points, total, spread cover).'
                      : 'Runs schedule sync, props, and joint Monte Carlo for the selected Date. Often takes several minutes—watch terminal lines prefixed [api] (uvicorn) or your API-only terminal.'
                  }
                  className="inline-flex shrink-0 items-center justify-center rounded-lg border border-zinc-400 bg-zinc-300 px-5 py-2.5 text-sm font-semibold text-zinc-950 shadow-sm transition hover:bg-zinc-200 hover:border-zinc-300 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isRefreshing ? 'Running...' : 'Run Simulations'}
                </button>
              </div>
              {isRefreshing && (
                <p className="mt-2 text-sm text-amber-200/90">
                  Joint run in progress—often several minutes. Keep this tab open until the button says Run Simulations
                  again; lines prefixed [api] (or your uvicorn terminal) show live progress.
                </p>
              )}
              {isRefreshing && (
                <p className="mt-1 text-xs text-amber-300/90">
                  Processing... {refreshElapsedSec}s elapsed
                </p>
              )}
              <p className="mt-3 text-sm text-zinc-400">
                {isLoading
                  ? 'Refreshing tables…'
                    : selectedStat === 'ml'
                    ? 'Book prices from The Odds API (DK/FD). Run Simulations also runs the joint player + model-ML sim and refreshes these lines.'
                    : selectedStat === 'gproj'
                      ? 'Game-level outputs from the joint simulation: projected team points, projected total, and cover-by-margin vs closing spread.'
                    : selectedStat === 'all'
                      ? 'Top 10 OVER and UNDER mix every prop from one joint run (same noise within each game). Run Simulations refreshes all stats together.'
                      : selectedStat === 'parlays'
                        ? 'Builds EV-optimized 3-leg parlays from possession_sim_v1 player_props: top 20 positive-EV legs -> all 3-leg combos -> top 5 by parlay EV.'
                        : 'Each run simulates PTS, REB, AST, STL, BLK, and 3PM together for the slate, then fills this stat from that pass.'}
              </p>
              {selectedStat === 'all' &&
                combinedCachedStats.length > 0 &&
                combinedCachedStats.length < ALL_PROP_STATS.length && (
                  <p className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100/95">
                    Combined top-10 only sees stats already in cache:{' '}
                    <span className="font-semibold text-amber-50">
                      {combinedCachedStats.map((s) => s.toUpperCase()).join(', ')}
                    </span>
                    . Click <span className="font-semibold">Run Simulations</span> to run the unified joint pass and
                    populate every prop.
                  </p>
                )}
              {error && <p className="mt-3 text-sm font-medium text-rose-400">{error}</p>}
            </section>

            {selectedStat === 'parlays' ? (
              <BestParlaysSection parlays={bestParlays} gameDate={gameDate} />
            ) : (
              <section className="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <ProjectionTable
                  title={
                    selectedStat === 'ml'
                      ? 'Top favorites (ML & spread)'
                      : selectedStat === 'gproj'
                        ? 'Game simulation projections'
                      : selectedStat === 'all'
                        ? 'Top OVERs (all stats)'
                        : 'Top OVERs'
                  }
                  rows={overs}
                  side="OVER"
                  onSaveTopPick={selectedStat === 'ml' || selectedStat === 'gproj' ? undefined : (row) => setSavedOver(row)}
                  showStatColumn={selectedStat === 'all'}
                  lineExplorerBoard={selectedStat === 'ml' || selectedStat === 'gproj' ? undefined : 'OVER'}
                  lineDeltaByKey={selectedStat === 'ml' || selectedStat === 'gproj' ? undefined : lineDeltaByKey}
                  onLineDeltaStep={selectedStat === 'ml' || selectedStat === 'gproj' ? undefined : handleLineDeltaStep}
                  onLineDeltaReset={selectedStat === 'ml' || selectedStat === 'gproj' ? undefined : handleLineDeltaReset}
                />
                {selectedStat !== 'gproj' && (
                  <ProjectionTable
                    title={
                      selectedStat === 'ml'
                        ? 'Top underdogs (ML & spread)'
                        : selectedStat === 'all'
                          ? 'Top UNDERs (all stats)'
                          : 'Top UNDERs'
                    }
                    rows={unders}
                    side="UNDER"
                    onSaveTopPick={selectedStat === 'ml' ? undefined : (row) => setSavedUnder(row)}
                    showStatColumn={selectedStat === 'all'}
                    lineExplorerBoard={selectedStat === 'ml' ? undefined : 'UNDER'}
                    lineDeltaByKey={selectedStat === 'ml' ? undefined : lineDeltaByKey}
                    onLineDeltaStep={selectedStat === 'ml' ? undefined : handleLineDeltaStep}
                    onLineDeltaReset={selectedStat === 'ml' ? undefined : handleLineDeltaReset}
                  />
                )}
              </section>
            )}

            {selectedStat !== 'ml' && selectedStat !== 'gproj' && selectedStat !== 'parlays' && simGameRows.length > 0 && (
              <ProjectionTable
                title="Model team win % (same joint PTS draws — top scorers summed per side)"
                rows={simGameRows}
                showStatColumn
              />
            )}

            {!isLoading && !hasData && !error && (
              <p className="text-center text-sm text-zinc-400">
                No data loaded yet. Try clicking <span className="font-semibold text-zinc-300">Run Simulations</span>.
              </p>
            )}
          </>
        )}

        {currentPage === 'savedPicks' && (
          <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-950/90 p-5 shadow-lg shadow-black/20">
            <h2 className="text-lg font-semibold text-zinc-100">Saved Top Picks</h2>
            <p className="text-sm text-zinc-400">
              Save any row from the OVER or UNDER tables as your top pick, then review them here while building cards.
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-zinc-600 bg-zinc-950/60 p-4">
                <h3 className="text-sm font-semibold text-zinc-200">Top OVER Pick</h3>
                {savedOver ? (
                  <div className="mt-2 space-y-1 text-sm text-zinc-200">
                    <p className="text-base font-semibold">
                      {getPlayerLabel(savedOver)}{' '}
                      <span className="text-xs font-medium text-zinc-400">
                        ({savedOver.team_abbr} vs {savedOver.opponent})
                      </span>
                    </p>
                    <p>
                      Line: <span className="font-semibold">{toNumber(savedOver.line)}</span>
                    </p>
                    <p>
                      Win %:{' '}
                      <span className="font-semibold text-zinc-100">{toPercent(getWinPct(savedOver))}</span>
                    </p>
                    <p>
                      EV:{' '}
                      <span className="font-semibold text-zinc-100">{toNumber(getEv(savedOver))}</span>
                    </p>
                    <p className="text-xs text-zinc-400">Verdict: {savedOver.verdict ?? 'No Verdict'}</p>
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-zinc-400">No OVER pick saved yet. Save one from the projections view.</p>
                )}
              </div>
              <div className="rounded-lg border border-zinc-600 bg-zinc-950/60 p-4">
                <h3 className="text-sm font-semibold text-zinc-200">Top UNDER Pick</h3>
                {savedUnder ? (
                  <div className="mt-2 space-y-1 text-sm text-zinc-200">
                    <p className="text-base font-semibold">
                      {getPlayerLabel(savedUnder)}{' '}
                      <span className="text-xs font-medium text-zinc-400">
                        ({savedUnder.team_abbr} vs {savedUnder.opponent})
                      </span>
                    </p>
                    <p>
                      Line: <span className="font-semibold">{toNumber(savedUnder.line)}</span>
                    </p>
                    <p>
                      Win %:{' '}
                      <span className="font-semibold text-zinc-100">{toPercent(getWinPct(savedUnder))}</span>
                    </p>
                    <p>
                      EV:{' '}
                      <span className="font-semibold text-zinc-100">{toNumber(getEv(savedUnder))}</span>
                    </p>
                    <p className="text-xs text-zinc-400">Verdict: {savedUnder.verdict ?? 'No Verdict'}</p>
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-zinc-400">
                    No UNDER pick saved yet. Save one from the projections view.
                  </p>
                )}
              </div>
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              Historical boards count only picks that made the daily top-10 list in Past Projections (per stat), not every
              logged simulation.
            </p>
            <div className="mt-6 grid gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-zinc-700 bg-zinc-950/60 p-4">
                <h3 className="text-sm font-semibold text-zinc-200">Historical OVER Board</h3>
                {historicalTopOver ? (
                  <div className="mt-2 space-y-1 text-sm text-zinc-200">
                    <p className="text-base font-semibold">
                      {historicalTopOver.player}{' '}
                      <span className="text-sm font-normal text-zinc-400">
                        ({historicalTopOver.stat.toUpperCase()} OVER)
                      </span>
                    </p>
                    <p>
                      Hit rate:{' '}
                      <span className="font-semibold text-zinc-100">
                        {((historicalTopOver.hits / historicalTopOver.played) * 100).toFixed(1)}%
                      </span>
                    </p>
                    <p className="text-xs text-zinc-400">
                      Based on {historicalTopOver.played} resolved picks that made the daily top-10 board (same slate as
                      Past Projections).
                    </p>
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-zinc-400">
                    No qualified OVER history yet. Need at least 5 resolved top-board picks for a player/stat.
                  </p>
                )}
              </div>
              <div className="rounded-lg border border-zinc-700 bg-zinc-950/60 p-4">
                <h3 className="text-sm font-semibold text-zinc-200">Historical UNDER Board</h3>
                {historicalTopUnder ? (
                  <div className="mt-2 space-y-1 text-sm text-zinc-200">
                    <p className="text-base font-semibold">
                      {historicalTopUnder.player}{' '}
                      <span className="text-sm font-normal text-zinc-400">
                        ({historicalTopUnder.stat.toUpperCase()} UNDER)
                      </span>
                    </p>
                    <p>
                      Hit rate:{' '}
                      <span className="font-semibold text-zinc-100">
                        {((historicalTopUnder.hits / historicalTopUnder.played) * 100).toFixed(1)}%
                      </span>
                    </p>
                    <p className="text-xs text-zinc-400">
                      Based on {historicalTopUnder.played} resolved picks that made the daily top-10 board (same slate as
                      Past Projections).
                    </p>
                  </div>
                ) : (
                  <p className="mt-2 text-sm text-zinc-400">
                    No qualified UNDER history yet. Need at least 5 resolved top-board picks for a player/stat.
                  </p>
                )}
              </div>
            </div>
          </section>
        )}

        {currentPage === 'leaderboard' && (
          <section className="space-y-5 rounded-xl border border-zinc-800 bg-zinc-950/90 p-5 shadow-lg shadow-black/20">
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">Leaderboard</h2>
              <p className="mt-1 text-sm text-zinc-400">
                Uses only the <span className="text-zinc-300">persisted daily top-10 board</span> (same rows as Past
                projections)—not the full history feed. Props (PTS, REB, AST, …) only; each row is one saved pick. A
                player with 3 props graded on one slate counts as 3 resolved (e.g. 3/3 if all hit). Tiebreak: more
                &quot;perfect&quot; slates where every saved pick that day hit. Need at least{' '}
                {MIN_LEADERBOARD_RESOLVED} graded saved picks per player.
              </p>
            </div>
            {isHistoryLoading && <p className="text-sm text-zinc-400">Loading history…</p>}
            {historyError && <p className="text-sm font-medium text-rose-400">{historyError}</p>}
            {!isHistoryLoading && !historyError && (
              <div className="overflow-x-auto rounded-lg border border-zinc-800">
                <table className="min-w-full divide-y divide-zinc-800 text-sm">
                  <thead className="bg-zinc-950">
                    <tr className="text-left text-zinc-400">
                      <th className="w-10 px-2 py-3 font-medium" aria-label="Expand picks" />
                      <th className="px-4 py-3 font-medium">#</th>
                      <th className="px-4 py-3 font-medium">Player</th>
                      <th className="px-4 py-3 font-medium">Hits / resolved</th>
                      <th className="px-4 py-3 font-medium">Hit rate</th>
                      <th className="whitespace-nowrap px-4 py-3 font-medium">Perfect slates</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800 text-zinc-200">
                    {savedBoardLeaderboard.map((row, i) => {
                      const open = expandedLeaderboardPlayer === row.playerKey
                      return (
                        <Fragment key={row.playerKey}>
                          <tr>
                            <td className="px-2 py-2 align-middle">
                              <button
                                type="button"
                                aria-expanded={open}
                                aria-label={open ? 'Hide pick breakdown' : 'Show pick breakdown'}
                                className="flex h-8 w-8 items-center justify-center rounded-md border border-zinc-600 bg-zinc-900 text-zinc-300 transition hover:bg-zinc-800 hover:text-zinc-100"
                                onClick={() =>
                                  setExpandedLeaderboardPlayer((k) => (k === row.playerKey ? null : row.playerKey))
                                }
                              >
                                <span className="text-xs font-bold" aria-hidden>
                                  {open ? '−' : '+'}
                                </span>
                              </button>
                            </td>
                            <td className="whitespace-nowrap px-4 py-3 text-zinc-400">{i + 1}</td>
                            <td className="px-4 py-3 font-medium text-zinc-100">{row.displayName}</td>
                            <td className="whitespace-nowrap px-4 py-3 tabular-nums">
                              {row.hits}/{row.total}
                            </td>
                            <td className="whitespace-nowrap px-4 py-3 font-semibold text-zinc-200">
                              {toPercent(row.rate)}
                            </td>
                            <td className="whitespace-nowrap px-4 py-3 tabular-nums text-zinc-300">
                              {row.perfectDays}
                            </td>
                          </tr>
                          {open && (
                            <tr className="bg-zinc-950/80">
                              <td colSpan={6} className="px-4 py-4">
                                <p className="mb-3 text-xs font-medium uppercase tracking-wide text-zinc-500">
                                  Saved picks by game date (scroll horizontally)
                                </p>
                                <div className="space-y-4">
                                  {row.picksByDate.map(([date, dayPicks]) => {
                                    const byOutcome = (pred: (h: boolean | null) => boolean) =>
                                      dayPicks.filter((r) => pred(r.hit)).sort((a, b) => statSortKey(a.stat) - statSortKey(b.stat))
                                    const hitPicks = byOutcome((h) => h === true)
                                    const missPicks = byOutcome((h) => h === false)
                                    const pendingPicks = byOutcome((h) => h === null)
                                    return (
                                      <div key={date}>
                                        <p className="mb-2 text-xs font-semibold text-zinc-400">{date}</p>
                                        <div className="space-y-2">
                                          {hitPicks.length > 0 && (
                                            <div>
                                              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-400/90">
                                                Hit
                                              </p>
                                              <div className="overflow-x-auto overflow-y-hidden pb-1">
                                                <div className="flex flex-nowrap gap-2">
                                                  {hitPicks.map((rec) => (
                                                    <LeaderboardPickBox key={rec.id} record={rec} />
                                                  ))}
                                                </div>
                                              </div>
                                            </div>
                                          )}
                                          {missPicks.length > 0 && (
                                            <div>
                                              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-rose-400/90">
                                                Miss
                                              </p>
                                              <div className="overflow-x-auto overflow-y-hidden pb-1">
                                                <div className="flex flex-nowrap gap-2">
                                                  {missPicks.map((rec) => (
                                                    <LeaderboardPickBox key={rec.id} record={rec} />
                                                  ))}
                                                </div>
                                              </div>
                                            </div>
                                          )}
                                          {pendingPicks.length > 0 && (
                                            <div>
                                              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-amber-400/90">
                                                Not graded yet
                                              </p>
                                              <div className="overflow-x-auto overflow-y-hidden pb-1">
                                                <div className="flex flex-nowrap gap-2">
                                                  {pendingPicks.map((rec) => (
                                                    <LeaderboardPickBox key={rec.id} record={rec} />
                                                  ))}
                                                </div>
                                              </div>
                                            </div>
                                          )}
                                        </div>
                                      </div>
                                    )
                                  })}
                                </div>
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      )
                    })}
                    {savedBoardLeaderboard.length === 0 && (
                      <tr>
                        <td className="px-4 py-8 text-center text-zinc-400" colSpan={6}>
                          No players with {MIN_LEADERBOARD_RESOLVED}+ graded saved-board picks yet. Run simulations so
                          picks log to history, then grade from Past (Update).
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {currentPage === 'pastProjections' && (
          <section className="space-y-5 rounded-xl border border-zinc-800 bg-zinc-950/90 p-5 shadow-lg shadow-black/20">
            <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-zinc-100">Past Projections</h2>
                <p className="text-sm text-zinc-400">
                  Top 10 picks per date (all stats), ranked by edge. Use Show all picks to see the full slate for that
                  day.
                </p>
              </div>
              <div className="flex flex-col gap-2 md:flex-row md:items-center md:gap-4">
                <div className="flex flex-col gap-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">Game Date</span>
                  <select
                    value={selectedHistoryDate ?? ''}
                    onChange={(event) =>
                      setSelectedHistoryDate(event.target.value.length > 0 ? event.target.value : null)
                    }
                    className="w-40 rounded-lg border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-xs text-zinc-100 outline-none ring-zinc-500 transition focus:ring-2"
                  >
                    {pastGameDates.length === 0 ? (
                      <option value="" disabled>
                        No past dates yet
                      </option>
                    ) : (
                      pastGameDates.map((date) => (
                        <option key={date} value={date}>
                          {date}
                        </option>
                      ))
                    )}
                  </select>
                </div>
                <button
                  type="button"
                  onClick={() => setShowAllPicksForDate((v) => !v)}
                  disabled={dedupedPastHistoryRows.length === 0}
                  className="inline-flex items-center justify-center rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-100 shadow-sm transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {showAllPicksForDate ? 'Show top 10 only' : 'Show all picks'}
                </button>
                <label className="inline-flex cursor-pointer items-center gap-2 text-xs text-zinc-300">
                  <input
                    type="checkbox"
                    checked={showUngradedPastOnly}
                    onChange={(e) => setShowUngradedPastOnly(e.target.checked)}
                    className="rounded border-zinc-600 bg-zinc-950 text-zinc-300 focus:ring-zinc-500"
                  />
                  Ungraded only
                </label>
                <button
                  type="button"
                  onClick={() => void handleUpdateHistory()}
                  className="inline-flex items-center justify-center rounded-lg border border-zinc-400 bg-zinc-300 px-3 py-1.5 text-xs font-semibold text-zinc-950 shadow-sm transition hover:bg-zinc-200 hover:border-zinc-300"
                >
                  Update
                </button>
                <button
                  type="button"
                  title="Clear grades for this date and re-run box-score matching (e.g. after a data fix)"
                  onClick={() => void handleRegradeSelectedDate()}
                  disabled={!selectedHistoryDate || isHistoryLoading}
                  className="inline-flex items-center justify-center rounded-lg border border-amber-700/80 bg-amber-950/50 px-3 py-1.5 text-xs font-semibold text-amber-100 shadow-sm transition hover:bg-amber-900/50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Re-grade this date
                </button>
              </div>
            </div>

            {isHistoryLoading && <p className="text-sm text-zinc-400">Loading past projections…</p>}
            {historyError && <p className="text-sm font-medium text-rose-400">{historyError}</p>}

            {!isHistoryLoading && !historyError && historyRecords.length === 0 && (
              <p className="text-sm text-zinc-400">
                No history available yet. As games resolve and hits are recorded, they will appear here.
              </p>
            )}

            {!isHistoryLoading && !historyError && historyRecords.length > 0 && pastGameDates.length === 0 && (
              <p className="text-sm text-zinc-400">
                Nothing to show for past dates yet. Today&apos;s slate is excluded here; check back after the Eastern
                calendar advances or run simulations for earlier days.
              </p>
            )}

            {!isHistoryLoading && !historyError && pastGameDates.length > 0 && (
              <div className="space-y-3">
                <div className="rounded-lg border border-zinc-800 bg-zinc-950/40 p-3">
                  <p className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
                    Yesterday Top 5 {yesterdayTopFive.date ? `(${yesterdayTopFive.date})` : ''}
                  </p>
                  {yesterdayTopFive.rows.length === 0 ? (
                    <p className="mt-2 text-xs text-zinc-400">No picks logged for yesterday in this history window.</p>
                  ) : (
                    <div className="mt-2 space-y-1.5 text-xs">
                      {yesterdayTopFive.rows.map((r) => {
                        const bulletColor = hitResultBulletClass(r.hit)
                        return (
                          <div key={`yt5-${r.id}`} className="flex items-center justify-between gap-3 text-zinc-200">
                            <div className="flex min-w-0 items-center gap-2">
                              <span className={`inline-block h-2.5 w-2.5 rounded-full ${bulletColor}`} />
                              <span className="truncate">
                                {r.player_name} {r.best_side} {toNumber(r.line)} {r.stat.toUpperCase()}
                              </span>
                            </div>
                            <span className="shrink-0 text-zinc-300">{toPercent(r.win_probability)}</span>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>

                {pastHistoryRows.length > 0 && (
                  <>
                    <p className="text-xs text-zinc-500">
                      Showing {picksTableRows.length} pick{picksTableRows.length === 1 ? '' : 's'} for{' '}
                      {selectedHistoryDate ?? '—'}
                      {showUngradedPastOnly ? ' (ungraded only)' : ''} — order: hits, then misses, then not graded. Use
                      Update to pull box scores. Use Re-grade this date after NBA corrects a box score so hits recompute for
                      that slate only.
                    </p>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-zinc-400">
                      <div className="flex items-center gap-1">
                        <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" /> <span>Hit</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-500" /> <span>Miss</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-400" />{' '}
                        <span>Not graded yet</span>
                      </div>
                    </div>

                    <div className="max-h-[480px] overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950/60">
                      <table className="min-w-full divide-y divide-zinc-800 text-xs">
                        <thead className="bg-zinc-950 text-zinc-400">
                          <tr>
                            <th className="px-3 py-2 text-left font-medium">Result</th>
                            <th className="px-3 py-2 text-left font-medium">Player</th>
                            <th className="px-3 py-2 text-left font-medium">Stat</th>
                            <th className="px-3 py-2 text-left font-medium">Matchup</th>
                            <th className="px-3 py-2 text-left font-medium">Side</th>
                            <th className="px-3 py-2 text-left font-medium">Line</th>
                            <th className="px-3 py-2 text-left font-medium">Win %</th>
                            <th className="px-3 py-2 text-left font-medium">EV</th>
                            <th className="px-3 py-2 text-left font-medium">Verdict</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800">
                          {picksTableRows.map((r) => {
                            const bulletColor = hitResultBulletClass(r.hit)
                            return (
                              <tr key={r.id} className="text-zinc-200">
                                <td className="px-3 py-2">
                                  <div className="flex items-center gap-2">
                                    <span className={`inline-block h-2.5 w-2.5 rounded-full ${bulletColor}`} />
                                    <span className="text-xs text-zinc-300">
                                      {r.actual_value === null || r.actual_value === undefined
                                        ? '—'
                                        : toNumber(r.actual_value)}
                                    </span>
                                  </div>
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-sm font-medium">
                                  {r.player_name}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-zinc-400">{r.stat.toUpperCase()}</td>
                                <td className="px-3 py-2 whitespace-nowrap text-zinc-300">
                                  {r.team_abbr} vs {r.opponent}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-xs text-zinc-300">
                                  {r.best_side}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  {toNumber(r.line)}
                                  <span className="ml-2 text-[10px] text-zinc-500">({r.line_source})</span>
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-xs">
                                  {toPercent(r.win_probability)}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap text-xs">
                                  {toNumber(r.ev_per_110)}
                                </td>
                                <td className="px-3 py-2 max-w-[220px] truncate text-xs text-zinc-300">
                                  {r.verdict}
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            )}
          </section>
        )}

        {currentPage === 'calibration' && (
          <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-950/90 p-5 shadow-lg shadow-black/20">
            <div>
              <h2 className="text-lg font-semibold text-zinc-100">Calibration</h2>
              <p className="mt-1 text-sm text-zinc-400">
                Graded props only: predicted win% (5-point buckets, midpoint vs label) vs realized hit rate. Large gaps
                mean EV inputs are miscalibrated.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void fetchCalibration()}
              disabled={calibrationLoading}
              className="rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-1.5 text-xs font-semibold text-zinc-100 hover:bg-zinc-700 disabled:opacity-50"
            >
              {calibrationLoading ? 'Loading…' : 'Refresh'}
            </button>
            {calibrationError && <p className="text-sm font-medium text-rose-400">{calibrationError}</p>}
            {calibrationData?.totals && (
              <p className="text-xs text-zinc-400">
                All graded props in DB: {calibrationData.totals.n} picks,{' '}
                {calibrationData.totals.overall_hit_pct != null
                  ? `${calibrationData.totals.overall_hit_pct}% hit`
                  : '—'}
                . Table below hides buckets with fewer than {calibrationData.min_per_cell} rows.
              </p>
            )}
            {calibrationLoading && !calibrationData && (
              <p className="text-sm text-zinc-400">Loading calibration…</p>
            )}
            {!calibrationLoading && calibrationData && calibrationData.cells.length === 0 && (
              <p className="text-sm text-zinc-400">
                No buckets meet the minimum count yet. Log more graded picks or lower the threshold in the API.
              </p>
            )}
            {calibrationData && calibrationData.cells.length > 0 && (
              <div className="overflow-x-auto rounded-lg border border-zinc-800">
                <table className="min-w-full divide-y divide-zinc-800 text-xs">
                  <thead className="bg-zinc-950 text-zinc-400">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">Win % bucket</th>
                      <th className="px-3 py-2 text-left font-medium">Side</th>
                      <th className="px-3 py-2 text-right font-medium">N</th>
                      <th className="px-3 py-2 text-right font-medium">Hits</th>
                      <th className="px-3 py-2 text-right font-medium">Predicted mid</th>
                      <th className="px-3 py-2 text-right font-medium">Actual %</th>
                      <th className="px-3 py-2 text-right font-medium">Gap (actual − pred)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-800 text-zinc-200">
                    {calibrationData.cells.map((c) => (
                      <tr key={`${c.bucket_lo}-${c.side}`}>
                        <td className="px-3 py-2 whitespace-nowrap">
                          {c.bucket_lo}%–{c.bucket_hi}%
                        </td>
                        <td className="px-3 py-2">{c.side}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{c.n}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{c.hits}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{c.predicted_mid_pct.toFixed(1)}%</td>
                        <td className="px-3 py-2 text-right tabular-nums">{c.actual_hit_pct.toFixed(2)}%</td>
                        <td
                          className={`px-3 py-2 text-right font-medium tabular-nums ${
                            Math.abs(c.gap_pct) > 5 ? 'text-amber-300' : 'text-zinc-300'
                          }`}
                        >
                          {c.gap_pct > 0 ? '+' : ''}
                          {c.gap_pct.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
