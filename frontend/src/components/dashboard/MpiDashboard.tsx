import { useState, useMemo, useCallback, memo } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────────────

interface MpiBreakdown {
  power:      number  // weight 0.25
  agility:    number  // weight 0.20
  endurance:  number  // weight 0.20
  efficiency: number  // weight 0.20
  cognitive:  number  // weight 0.15
}

interface MpiPlayer {
  trackId:      number
  jerseyNumber: number
  name:         string
  team:         'A' | 'B'
  mpiComposite: number   // 0–100
  mpiBreakdown: MpiBreakdown
}

export interface MpiDashboardProps {
  players:         MpiPlayer[]
  selectedQuarter: number   // 0=All, 1-4 (used as initial value; quarter picker is internal)
}

// ── Constants ──────────────────────────────────────────────────────────────────

const BREAKDOWN_CONFIG = [
  { key: 'power',      label: 'Power',      weight: 0.25, color: '#6366F1' },
  { key: 'agility',    label: 'Agility',    weight: 0.20, color: '#10B981' },
  { key: 'endurance',  label: 'Endurance',  weight: 0.20, color: '#F59E0B' },
  { key: 'efficiency', label: 'Efficiency', weight: 0.20, color: '#00BCD4' },
  { key: 'cognitive',  label: 'Cognitive',  weight: 0.15, color: '#8B5CF6' },
] as const

const TEAM_BORDER: Record<'A' | 'B', string> = {
  A: 'border-l-2 border-l-primary',
  B: 'border-l-2 border-l-surface-dark',
}

const TEAM_LABEL: Record<'A' | 'B', string> = {
  A: 'text-primary',
  B: 'text-surface-dark',
}

function barColor(score: number): string {
  if (score >= 80) return '#00BCD4'   // primary — bright teal
  if (score >= 70) return '#00838F'   // primary-dark
  if (score >= 60) return '#0097A7'   // mid teal
  return '#9CA3AF'                    // muted gray
}

// ── Breakdown tooltip (hover) ──────────────────────────────────────────────────

function BreakdownTooltip({ breakdown }: { breakdown: MpiBreakdown }) {
  return (
    <div
      className="
        absolute z-20 left-1/2 -translate-x-1/2 bottom-[calc(100%+6px)]
        bg-surface-dark text-white text-xs rounded-lg p-3 shadow-xl
        w-44 pointer-events-none
        opacity-0 group-hover:opacity-100
        transition-opacity duration-150
      "
    >
      <div className="font-bold text-white/70 mb-2 text-[10px] uppercase tracking-wide">
        Breakdown
      </div>
      {BREAKDOWN_CONFIG.map(({ key, label, weight }) => (
        <div key={key} className="flex justify-between items-center py-0.5">
          <span className="text-white/80">
            {label}
            <span className="text-white/40 ml-1">×{weight}</span>
          </span>
          <span className="font-bold tabular-nums">
            {breakdown[key].toFixed(1)}
          </span>
        </div>
      ))}
      {/* Arrow */}
      <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0
        border-x-4 border-x-transparent border-t-4 border-t-surface-dark" />
    </div>
  )
}

// ── Breakdown panel (click-expand) ────────────────────────────────────────────

function BreakdownPanel({ breakdown }: { breakdown: MpiBreakdown }) {
  return (
    <div className="mt-3 pt-3 border-t border-gray-100 space-y-2">
      {BREAKDOWN_CONFIG.map(({ key, label, weight, color }) => {
        const val = breakdown[key]
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="text-[10px] text-text-secondary w-20 flex-shrink-0">
              {label}
              <span className="text-text-secondary/60 ml-0.5">×{weight}</span>
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, val)}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-[10px] font-bold tabular-nums w-8 text-right text-text-primary">
              {val.toFixed(0)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Player row ─────────────────────────────────────────────────────────────────

interface PlayerRowProps {
  player:     MpiPlayer
  isExpanded: boolean
  onToggle:   () => void
}

const PlayerRow = memo(function PlayerRow({
  player,
  isExpanded,
  onToggle,
}: PlayerRowProps) {
  const color  = barColor(player.mpiComposite)
  const pct    = Math.min(100, Math.max(0, player.mpiComposite))

  return (
    <div
      className={`rounded-lg bg-gray-50 pl-3 pr-4 py-3 ${TEAM_BORDER[player.team]} cursor-pointer
        hover:bg-gray-100 transition-colors duration-150 select-none`}
      onClick={onToggle}
      role="button"
      aria-expanded={isExpanded}
    >
      {/* Main row */}
      <div className="flex items-center gap-3 relative group">
        {/* Jersey + name */}
        <div className="flex-shrink-0 w-32">
          <span
            className={`text-[10px] font-bold ${TEAM_LABEL[player.team]} mr-1 tabular-nums`}
          >
            #{player.jerseyNumber}
          </span>
          <span className="text-sm font-semibold text-text-primary truncate">
            {player.name}
          </span>
        </div>

        {/* Bar */}
        <div className="flex-1 bg-gray-200 rounded-full h-3 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>

        {/* Score */}
        <span
          className="flex-shrink-0 w-9 text-right text-sm font-bold tabular-nums"
          style={{ color }}
        >
          {player.mpiComposite.toFixed(0)}
        </span>

        {/* Expand chevron */}
        <span className="flex-shrink-0 text-text-secondary">
          {isExpanded
            ? <ChevronUp size={14} />
            : <ChevronDown size={14} />
          }
        </span>

        {/* Hover tooltip — sits above the row */}
        <BreakdownTooltip breakdown={player.mpiBreakdown} />
      </div>

      {/* Expanded breakdown (click-to-open) */}
      <div
        className="overflow-hidden transition-all duration-300"
        style={{ maxHeight: isExpanded ? '200px' : '0px' }}
      >
        <BreakdownPanel breakdown={player.mpiBreakdown} />
      </div>
    </div>
  )
})

// ── Empty state ────────────────────────────────────────────────────────────────

function EmptyState() {
  return (
    <div className="py-10 text-center text-text-secondary">
      <div className="text-3xl mb-3">📊</div>
      <p className="text-sm font-medium">Belum ada data MPI</p>
      <p className="text-xs mt-1">Tersedia setelah video diproses</p>
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function MpiDashboard({
  players,
  selectedQuarter,
}: MpiDashboardProps) {
  // Quarter selector is managed internally; selectedQuarter is the initial value
  const [quarter, setQuarter]     = useState<number>(selectedQuarter)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const sorted = useMemo(
    () => [...players].sort((a, b) => b.mpiComposite - a.mpiComposite),
    [players],
  )

  const toggleExpand = useCallback((trackId: number) => {
    setExpandedId((prev) => (prev === trackId ? null : trackId))
  }, [])

  const quarterLabel = quarter === 0 ? 'All' : `Q${quarter}`

  return (
    <div className="bg-surface rounded-xl border border-gray-100 shadow-sm flex flex-col">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <h3 className="font-display text-lg font-bold tracking-wide">
          MPI COMPOSITE INDEX
        </h3>

        <select
          value={quarter}
          onChange={(e) => {
            setQuarter(Number(e.target.value))
            setExpandedId(null)
          }}
          className="text-xs border border-gray-200 rounded-md px-2 py-1.5
            text-text-secondary bg-surface cursor-pointer
            hover:border-gray-300 transition-colors"
          aria-label="Select quarter"
        >
          <option value={0}>All</option>
          <option value={1}>Q1</option>
          <option value={2}>Q2</option>
          <option value={3}>Q3</option>
          <option value={4}>Q4</option>
        </select>
      </div>

      {/* ── Team legend ───────────────────────────────────────────────────── */}
      <div className="flex items-center gap-4 px-5 pt-3 pb-1">
        <span className="flex items-center gap-1.5 text-[10px] text-text-secondary font-medium">
          <span className="inline-block w-2 h-2 rounded-full bg-primary" />
          Tim A
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-text-secondary font-medium">
          <span className="inline-block w-2 h-2 rounded-full bg-surface-dark" />
          Tim B
        </span>
        <span className="ml-auto text-[10px] text-text-secondary italic">
          {quarterLabel} — {sorted.length} player{sorted.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* ── Bar list ──────────────────────────────────────────────────────── */}
      <div className="px-4 py-3 space-y-2 flex-1">
        {sorted.length === 0 ? (
          <EmptyState />
        ) : (
          sorted.map((player) => (
            <PlayerRow
              key={player.trackId}
              player={player}
              isExpanded={expandedId === player.trackId}
              onToggle={() => toggleExpand(player.trackId)}
            />
          ))
        )}
      </div>

      {/* ── Formula footer ────────────────────────────────────────────────── */}
      <div className="px-5 py-3 border-t border-gray-100 text-[10px] text-text-secondary leading-relaxed">
        <span className="font-semibold text-text-primary">Formula: </span>
        {BREAKDOWN_CONFIG.map(({ label, weight }, i) => (
          <span key={label}>
            {label}×{weight}
            {i < BREAKDOWN_CONFIG.length - 1 ? ' + ' : ''}
          </span>
        ))}
      </div>
    </div>
  )
}
