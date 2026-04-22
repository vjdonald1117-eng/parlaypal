import { toNumber } from './ProjectionTable'
import type { ParlaySlipLeg } from './ProjectionTable'

export function ParlaySlip({
  selectedLegs,
  onClearSlip,
}: {
  selectedLegs: ParlaySlipLeg[]
  onClearSlip: () => void
}) {
  return (
    <aside className="w-full rounded-2xl border border-zinc-700/80 bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 p-4 shadow-2xl shadow-black/40 xl:sticky xl:top-24 xl:w-[22rem]">
      <div className="mb-4 flex items-center justify-between border-b border-zinc-700/70 pb-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-zinc-400">Parlay Slip</p>
          <h3 className="text-lg font-semibold text-zinc-100">Build Your Ticket</h3>
        </div>
        <div className="inline-flex min-w-12 items-center justify-center rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2.5 py-1 text-xs font-semibold text-emerald-300">
          {selectedLegs.length} Leg{selectedLegs.length === 1 ? '' : 's'}
        </div>
      </div>

      <div className="max-h-[55vh] space-y-2 overflow-y-auto pr-1">
        {selectedLegs.length === 0 ? (
          <div className="rounded-lg border border-dashed border-zinc-600 bg-zinc-900/40 px-3 py-6 text-center text-sm text-zinc-400">
            Add props with the + button to start your parlay.
          </div>
        ) : (
          selectedLegs.map((leg, idx) => (
            <div key={`${leg.id}-${idx}`} className="rounded-lg border border-zinc-700 bg-zinc-900/70 p-3">
              <p className="truncate text-sm font-semibold text-zinc-100">{leg.playerName}</p>
              <p className="mt-1 text-xs text-zinc-400">{leg.matchup}</p>
              <div className="mt-2 flex items-center justify-between text-xs">
                <span className="rounded bg-zinc-800 px-2 py-0.5 font-semibold text-zinc-100">
                  {leg.side} {leg.stat}
                </span>
                <span className="font-medium tabular-nums text-zinc-300">
                  {leg.line != null ? toNumber(leg.line) : '—'}
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      <button
        type="button"
        onClick={onClearSlip}
        disabled={selectedLegs.length === 0}
        className="mt-4 w-full rounded-lg border border-zinc-500 bg-zinc-800 px-3 py-2 text-sm font-semibold text-zinc-100 transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Clear Slip
      </button>
    </aside>
  )
}
