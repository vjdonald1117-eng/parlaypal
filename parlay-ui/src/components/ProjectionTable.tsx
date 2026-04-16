import { useMemo } from 'react'

/** Matches backend /api/projections and /api/refresh result items */
export type ProjectionRow = {
  game_id?: number
  player_name?: string
  team_abbr?: string
  opponent?: string
  matchup?: string
  stat?: string
  /** Prop: book line. For gproj, market game total (O/U). */
  line?: number
  /** Game only: closing spread (points); optional detail for gproj. */
  spread_line?: number
  heuristic_mean?: number
  win_probability?: number
  ev_per_110?: number
  verdict?: string
  best_side?: string
  /** Set when two-stage MC stopped after the first 2k batch (API `sim_note`) */
  sim_note?: string
  /** Legacy / alternate keys */
  player?: string
  projected_mean?: number
  win_pct?: number
  ev?: number
  ensemble_lock?: boolean
  /** Short model context from backend (pace, defense, NMU, minutes, etc.) */
  explanation_tags?: string[]
  /** Historical / MC scale from API (joint run = raw σ; used for what-if line tool) */
  std_dev?: number
}

export const ALL_PROP_STATS = ['pts', 'reb', 'ast', 'stl', 'blk', 'fg3'] as const

/** API/DB may send numbers as strings; calling .toFixed on a string crashes React (white screen). */
export function coerceFiniteNumber(value: unknown): number | undefined {
  if (value === null || value === undefined) return undefined
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : undefined
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value)
    return Number.isFinite(n) ? n : undefined
  }
  return undefined
}

export function toPercent(value?: unknown): string {
  const n = coerceFiniteNumber(value)
  if (n === undefined) return '-'
  const normalized = n <= 1 ? n * 100 : n
  return `${normalized.toFixed(1)}%`
}

export function toNumber(value?: unknown): string {
  const n = coerceFiniteNumber(value)
  if (n === undefined) return '-'
  return n.toFixed(2)
}

function isGameProjectionRow(row: ProjectionRow): boolean {
  return String(row.stat ?? '').toLowerCase() === 'gproj'
}

/** Market O/U total for gproj Line column (1 decimal, Vegas-style). */
function formatGameTotalLineCell(row: ProjectionRow): string {
  const n = coerceFiniteNumber(row.line)
  if (n === undefined) return '—'
  return n.toFixed(1)
}

function isPositiveWinPct(value?: number): boolean {
  if (value === undefined || Number.isNaN(value)) return false
  return (value <= 1 ? value * 100 : value) > 55
}

export function getWinPct(row: ProjectionRow): number | undefined {
  return coerceFiniteNumber(row.win_probability ?? row.win_pct)
}

/** Comparable 0–100 for sorting (API may send 0–1 or 0–100). */
function winPctSortValue(row: ProjectionRow): number {
  const v = getWinPct(row)
  if (v === undefined || Number.isNaN(v)) return -1
  return v <= 1 ? v * 100 : v
}

export function sortProjectionRowsByEdgeDesc(a: ProjectionRow, b: ProjectionRow): number {
  const wb = winPctSortValue(b)
  const wa = winPctSortValue(a)
  if (wb !== wa) return wb - wa
  return (b.ensemble_lock ? 1 : 0) - (a.ensemble_lock ? 1 : 0)
}

function getProjectedMean(row: ProjectionRow): number | undefined {
  return coerceFiniteNumber(row.heuristic_mean ?? row.projected_mean)
}

export function getEv(row: ProjectionRow): number | undefined {
  return coerceFiniteNumber(row.ev_per_110 ?? row.ev)
}

function formatMatchup(row: ProjectionRow): string {
  if (row.matchup) return row.matchup
  if (row.team_abbr && row.opponent) return `${row.team_abbr} vs ${row.opponent}`
  return '-'
}

export function getPlayerLabel(row: ProjectionRow): string {
  return row.player_name ?? row.player ?? '-'
}

/** Mirrors `monte_carlo_sim.effective_std_for_prop` for browser what-if MC */
const LINE_UNDER_RATIO = 0.985
const UNDER_STD_MULT = 1.08
export const LINE_ADJUST_STEP = 1

function gaussianSample(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

function effectiveStdForLineExplorer(stdDev: number, simMean: number, line: number): number {
  if (line <= 0 || stdDev <= 0) return stdDev
  if (simMean < line * LINE_UNDER_RATIO) return stdDev * UNDER_STD_MULT
  return stdDev
}

type LineExplorerRates = {
  overPct: number
  underPct: number
  bestPct: number
  bestSide: 'OVER' | 'UNDER'
}

function estimateHitRatesAtLine(
  simMean: number,
  stdDev: number,
  targetLine: number,
  nSamples: number = 8000,
): LineExplorerRates {
  const sigma = effectiveStdForLineExplorer(stdDev, simMean, targetLine)
  let over = 0
  for (let i = 0; i < nSamples; i++) {
    const raw = simMean + sigma * gaussianSample()
    const x = raw > 0 ? raw : 0
    if (x > targetLine) over++
  }
  const overPct = (over / nSamples) * 100
  const underPct = 100 - overPct
  const bestSide: 'OVER' | 'UNDER' = overPct >= underPct ? 'OVER' : 'UNDER'
  return { overPct, underPct, bestPct: Math.max(overPct, underPct), bestSide }
}

function lineExplorerRowKey(row: ProjectionRow, board: 'OVER' | 'UNDER', idx: number): string {
  return `${board}|${getPlayerLabel(row)}|${(row.stat ?? '').toLowerCase()}|${row.team_abbr ?? ''}|${idx}`
}

const lineExplorerBtnClass =
  'inline-flex h-7 w-7 shrink-0 items-center justify-center rounded border border-zinc-600 bg-zinc-900 text-sm font-semibold text-zinc-200 transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40'

/** Subtle pill accents for explanation_tags (max 3 from API). */
const EXPLANATION_BADGE_STYLES = [
  'border-amber-800/45 bg-amber-950/55 text-amber-100/95',
  'border-sky-800/45 bg-sky-950/55 text-sky-100/95',
  'border-violet-800/45 bg-violet-950/55 text-violet-100/95',
] as const

function LineExplorerPanel({
  baseLine,
  adjLine,
  deltaSteps,
  rowKey,
  rates,
  onStep,
  onReset,
}: {
  baseLine: number
  adjLine: number
  deltaSteps: number
  rowKey: string
  rates: LineExplorerRates | null
  onStep: (key: string, dir: -1 | 1, baseLine: number) => void
  onReset: (key: string) => void
}) {
  const canDec = adjLine - LINE_ADJUST_STEP >= 0.25

  return (
    <div className="flex flex-col gap-1 py-0.5">
      <div className="flex items-center gap-1">
        <button
          type="button"
          aria-label="Decrease line by one point"
          className={lineExplorerBtnClass}
          disabled={!canDec}
          onClick={() => onStep(rowKey, -1, baseLine)}
        >
          −
        </button>
        <span className="min-w-[2.75rem] text-center tabular-nums font-medium text-zinc-100">
          {adjLine.toFixed(1)}
        </span>
        <button
          type="button"
          aria-label="Increase line by one point"
          className={lineExplorerBtnClass}
          onClick={() => onStep(rowKey, 1, baseLine)}
        >
          +
        </button>
      </div>
      {rates ? (
        <div className="max-w-[11rem] space-y-0.5">
          <p
            className="text-[11px] font-semibold leading-snug text-zinc-200"
            title="8k heuristic MC at the line above (not the full backend PTS/XGB run). Win % column matches this only after you use ± to change the line."
          >
            Pred. win %: {rates.bestPct.toFixed(1)}%
            <span className="ml-1 font-normal text-zinc-500">({rates.bestSide})</span>
          </p>
          <p className="text-[10px] leading-snug text-zinc-400">
            Hit % · O {rates.overPct.toFixed(1)}% · U {rates.underPct.toFixed(1)}%
          </p>
          {deltaSteps === 0 ? (
            <p className="text-[9px] leading-snug text-zinc-600">
              Win % column = saved board run (may differ slightly).
            </p>
          ) : null}
        </div>
      ) : (
        <p className="max-w-[11rem] text-[10px] text-zinc-600">
          Need projected mean + std_dev (unified sim) to estimate hit %.
        </p>
      )}
      {deltaSteps !== 0 ? (
        <button
          type="button"
          className="self-start text-[10px] font-medium text-zinc-500 underline decoration-zinc-600 underline-offset-2 hover:text-zinc-400"
          onClick={() => onReset(rowKey)}
        >
          Reset to book
        </button>
      ) : null}
    </div>
  )
}

function ProjectionTableRow({
  row,
  lineKey,
  useLineExplorer,
  lineExplorerBoard,
  deltaSteps,
  onLineDeltaStep,
  onLineDeltaReset,
  showStatColumn,
  onSaveTopPick,
  side,
}: {
  row: ProjectionRow
  lineKey: string
  useLineExplorer: boolean
  lineExplorerBoard: 'OVER' | 'UNDER' | undefined
  deltaSteps: number
  onLineDeltaStep: (rowKey: string, dir: -1 | 1, baseLine: number) => void
  onLineDeltaReset: (rowKey: string) => void
  showStatColumn?: boolean
  onSaveTopPick?: (row: ProjectionRow) => void
  side?: 'OVER' | 'UNDER'
}) {
  const rowStat = String(row.stat ?? '').toLowerCase()
  const explorerEnabled = Boolean(
    useLineExplorer && lineExplorerBoard && rowStat !== 'win',
  )
  const baseLineNum = coerceFiniteNumber(row.line)
  const mean = getProjectedMean(row)
  const std = coerceFiniteNumber(row.std_dev)
  const adjLine = baseLineNum !== undefined ? baseLineNum + deltaSteps * LINE_ADJUST_STEP : NaN

  const explorerRates = useMemo((): LineExplorerRates | null => {
    if (!explorerEnabled) return null
    if (mean === undefined || std === undefined || std <= 0) return null
    if (baseLineNum === undefined || Number.isNaN(adjLine)) return null
    return estimateHitRatesAtLine(mean, std, adjLine)
  }, [explorerEnabled, mean, std, baseLineNum, adjLine])

  const boardWinPct = getWinPct(row)
  /** Win % column = API/board run at book line unless user moved ± (then use explorer at that line). */
  const useExplorerWinInColumn = explorerRates != null && deltaSteps !== 0
  const winPctForColumn = useExplorerWinInColumn ? explorerRates!.bestPct : boardWinPct
  const winStrong = isPositiveWinPct(winPctForColumn)

  const primaryLabel = isGameProjectionRow(row) ? formatMatchup(row) : getPlayerLabel(row)
  const projectedMean = getProjectedMean(row)
  const projectedDisplay =
    projectedMean === undefined
      ? '-'
      : isGameProjectionRow(row)
        ? projectedMean.toFixed(1)
        : toNumber(projectedMean)

  return (
    <tr className="text-zinc-200">
      <td className="px-4 py-3">
        <div className="flex min-w-0 max-w-xs flex-col gap-1.5">
          <span className="whitespace-nowrap font-medium text-zinc-200">{primaryLabel}</span>
          {row.explanation_tags && row.explanation_tags.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {row.explanation_tags.map((t, i) => (
                <span
                  key={`${i}-${t}`}
                  title={t}
                  className={`inline-flex max-w-full shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold leading-tight tracking-tight ${EXPLANATION_BADGE_STYLES[i % EXPLANATION_BADGE_STYLES.length]}`}
                >
                  <span className="truncate">{t}</span>
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </td>
      {showStatColumn && (
        <td className="whitespace-nowrap px-4 py-3 font-medium text-zinc-200">
          {(row.stat ?? '').toUpperCase()}
        </td>
      )}
      <td className="whitespace-nowrap px-4 py-3 text-zinc-300">{formatMatchup(row)}</td>
      <td className="px-4 py-3 align-top text-zinc-200">
        {explorerEnabled && baseLineNum !== undefined ? (
          <LineExplorerPanel
            baseLine={baseLineNum}
            adjLine={adjLine}
            deltaSteps={deltaSteps}
            rowKey={lineKey}
            rates={explorerRates}
            onStep={onLineDeltaStep}
            onReset={onLineDeltaReset}
          />
        ) : (
          <span className="whitespace-nowrap tabular-nums" title={isGameProjectionRow(row) ? 'Market game total (O/U)' : undefined}>
            {isGameProjectionRow(row) ? formatGameTotalLineCell(row) : toNumber(row.line)}
          </span>
        )}
      </td>
      <td className="whitespace-nowrap px-4 py-3">{projectedDisplay}</td>
      <td
        className="px-4 py-3"
        title={
          useExplorerWinInColumn
            ? `Win % for adjusted line ${adjLine.toFixed(1)} (browser MC). Reset line to match board Win % again.`
            : undefined
        }
      >
        <div className={`font-semibold tabular-nums ${winStrong ? 'text-zinc-100' : 'text-zinc-400'}`}>
          <span>{toPercent(winPctForColumn)}</span>
          {row.sim_note ? (
            <span className="mt-0.5 block text-[10px] font-normal normal-case text-zinc-500">
              {row.sim_note}
            </span>
          ) : null}
        </div>
      </td>
      <td
        className={`whitespace-nowrap px-4 py-3 font-semibold ${
          (getEv(row) ?? 0) > 0 ? 'text-zinc-100' : 'text-zinc-400'
        }`}
      >
        {toNumber(getEv(row))}
      </td>
      <td className="px-4 py-3">
        <VerdictBadge verdict={row.verdict} />
      </td>
      {onSaveTopPick && (
        <td className="px-4 py-3 text-right">
          <button
            type="button"
            onClick={() => onSaveTopPick(row)}
            className="rounded-full border border-zinc-500 bg-zinc-800 px-3 py-1 text-xs font-semibold text-zinc-100 hover:bg-zinc-700"
          >
            Save {side ?? ''}
          </button>
        </td>
      )}
    </tr>
  )
}

function VerdictBadge({ verdict }: { verdict?: string }) {
  const text = verdict ?? 'No Verdict'
  const isVeryStrong = text.toLowerCase().includes('very strong edge')

  return (
    <span
      className={
        isVeryStrong
          ? 'rounded-full border border-zinc-500 bg-zinc-800 px-2.5 py-1 text-xs font-semibold text-zinc-100'
          : 'rounded-full border border-zinc-700 bg-zinc-900 px-2.5 py-1 text-xs font-medium text-zinc-300'
      }
    >
      {text}
    </span>
  )
}

export function ProjectionTable({
  title,
  rows,
  side,
  onSaveTopPick,
  showStatColumn,
  lineExplorerBoard,
  lineDeltaByKey,
  onLineDeltaStep,
  onLineDeltaReset,
}: {
  title: string
  rows: ProjectionRow[]
  side?: 'OVER' | 'UNDER'
  onSaveTopPick?: (row: ProjectionRow) => void
  showStatColumn?: boolean
  /** When set with callbacks, Line column gets ±1 adjuster and heuristic hit rates */
  lineExplorerBoard?: 'OVER' | 'UNDER'
  lineDeltaByKey?: Record<string, number>
  onLineDeltaStep?: (rowKey: string, dir: -1 | 1, baseLine: number) => void
  onLineDeltaReset?: (rowKey: string) => void
}) {
  const hasGameRows = rows.some(isGameProjectionRow)
  const firstHeader = hasGameRows ? 'Game' : 'Player Name'
  const meanHeader = hasGameRows ? 'Projected Total' : 'Projected Mean'
  const lineHeader = hasGameRows ? 'O/U' : 'Line'
  const lineHeaderTitle = hasGameRows
    ? 'Market game total (over/under). Compare to Projected Total.'
    : 'Book line: use − / + to move the line by 1 and see heuristic O/U hit rates.'
  const colSpan = 7 + (showStatColumn ? 1 : 0) + (onSaveTopPick ? 1 : 0)
  const useLineExplorer =
    (lineExplorerBoard === 'OVER' || lineExplorerBoard === 'UNDER') &&
    lineDeltaByKey != null &&
    onLineDeltaStep != null &&
    onLineDeltaReset != null
  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-950/90 shadow-lg shadow-black/30">
      <div className="border-b border-zinc-800 px-4 py-3">
        <h2 className="text-lg font-semibold text-zinc-100">{title}</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-zinc-800 text-sm">
          <thead className="bg-zinc-950">
            <tr className="text-left text-zinc-400">
              <th className="px-4 py-3 font-medium">{firstHeader}</th>
              {showStatColumn && <th className="px-4 py-3 font-medium">Stat</th>}
              <th className="px-4 py-3 font-medium">Matchup</th>
              <th className="px-4 py-3 font-medium" title={lineHeaderTitle}>
                {lineHeader}
              </th>
              <th className="px-4 py-3 font-medium">{meanHeader}</th>
              <th
                className="px-4 py-3 font-medium"
                title="Saved model win % at the book line. After you move the line with ±, this column switches to match the explorer MC at that line."
              >
                Win %
              </th>
              <th className="px-4 py-3 font-medium">EV</th>
              <th className="px-4 py-3 font-medium">Verdict</th>
              {onSaveTopPick && <th className="px-4 py-3 font-medium text-right">Save</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800">
            {rows.map((row, idx) => {
              const rowKey =
                isGameProjectionRow(row) && row.game_id != null
                  ? `gproj-${row.game_id}`
                  : `${getPlayerLabel(row)}-${row.stat ?? ''}-${idx}`
              const lineKey =
                useLineExplorer && lineExplorerBoard
                  ? lineExplorerRowKey(row, lineExplorerBoard, idx)
                  : ''
              const deltaSteps =
                useLineExplorer && lineKey ? (lineDeltaByKey![lineKey] ?? 0) : 0
              return (
                <ProjectionTableRow
                  key={rowKey}
                  row={row}
                  lineKey={lineKey}
                  useLineExplorer={useLineExplorer}
                  lineExplorerBoard={lineExplorerBoard}
                  deltaSteps={deltaSteps}
                  onLineDeltaStep={onLineDeltaStep ?? (() => {})}
                  onLineDeltaReset={onLineDeltaReset ?? (() => {})}
                  showStatColumn={showStatColumn}
                  onSaveTopPick={onSaveTopPick}
                  side={side}
                />
              )
            })}
            {rows.length === 0 && (
              <tr>
                <td className="px-4 py-6 text-center text-zinc-400" colSpan={colSpan}>
                  No projections available yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
