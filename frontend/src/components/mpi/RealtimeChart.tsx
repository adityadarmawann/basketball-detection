import { useState, useMemo, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { TooltipProps } from 'recharts'

// ── Types ──────────────────────────────────────────────────────────────────────

export interface DataPoint {
  frame: number
  value: number
  timestamp: string
}

export interface SeriesData {
  name: string
  color: string
  dataPoints: DataPoint[]
}

export interface RealtimeChartProps {
  data: Record<number, SeriesData>
  metricLabel: string
  maxDataPoints?: number
  height?: number
  /** Show ⚠️ Estimasi CV badge (for approx metrics like speed, jump height) */
  isApprox?: boolean
}

// ── Constants ──────────────────────────────────────────────────────────────────

/**
 * Fallback color palette aligned with the design system.
 * Parents should supply `color` per series; this is the safety net.
 *
 * Index 0 → Team A (primary teal)
 * Index 1 → Team B (dark navy)
 * Index 2+ → other players
 */
const PALETTE = [
  '#00BCD4', // primary / team-a
  '#1A1A2E', // team-b
  '#10B981', // success green
  '#F59E0B', // accent gold
  '#EF4444', // danger red
  '#8B5CF6', // purple
  '#F97316', // orange
  '#06B6D4', // cyan-500
]

const SKELETON_HEIGHTS = [35, 62, 45, 78, 52, 84, 41, 67, 73, 50, 31, 88]

// ── Skeleton ───────────────────────────────────────────────────────────────────

function ChartSkeleton({ height }: { height: number }) {
  return (
    <div
      style={{ height }}
      className="flex flex-col animate-pulse select-none"
      aria-label="Loading chart"
    >
      {/* Fake bars */}
      <div className="flex items-end gap-1 flex-1 px-2 pt-2">
        {SKELETON_HEIGHTS.map((h, i) => (
          <div
            key={i}
            className="flex-1 bg-gray-200 rounded-sm"
            style={{ height: `${h}%` }}
          />
        ))}
      </div>
      {/* Fake x-axis */}
      <div className="h-3 bg-gray-200 rounded mt-3 mx-2" />
      {/* Fake legend */}
      <div className="flex gap-3 mt-4 px-2">
        <div className="h-3 w-16 bg-gray-200 rounded-full" />
        <div className="h-3 w-20 bg-gray-200 rounded-full" />
        <div className="h-3 w-14 bg-gray-200 rounded-full" />
      </div>
    </div>
  )
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────────
// Defined outside RealtimeChart to keep a stable reference — avoids Recharts
// re-registering the tooltip on every parent render.

interface TooltipContentProps extends TooltipProps<number, string> {
  seriesMap: Record<number, SeriesData>
}

function TooltipContent({ active, payload, label, seriesMap }: TooltipContentProps) {
  if (!active || !payload?.length) return null

  return (
    <div className="bg-surface border border-gray-200 rounded-lg p-3 shadow-lg text-sm min-w-[148px]">
      <p className="text-text-secondary text-xs mb-2 tabular-nums">Frame {label}</p>
      {payload.map((entry) => {
        const series = seriesMap[Number(entry.dataKey)]
        return (
          <div
            key={entry.dataKey}
            className="flex items-center justify-between gap-4 py-0.5"
          >
            <span
              className="font-medium truncate max-w-[88px] text-xs"
              style={{ color: entry.color }}
            >
              {series?.name ?? String(entry.dataKey)}
            </span>
            <span className="font-bold tabular-nums text-text-primary text-xs ml-auto">
              {typeof entry.value === 'number' ? entry.value.toFixed(1) : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function RealtimeChart({
  data,
  metricLabel,
  maxDataPoints = 200,
  height = 300,
  isApprox = false,
}: RealtimeChartProps) {
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  // Stable list of player IDs — only recomputes when keys change, not on value updates
  const playerIds = useMemo(() => Object.keys(data), [data])

  /**
   * Merge per-player dataPoint arrays into a single frame-indexed array
   * that Recharts expects: [{ frame, "1": 32.5, "7": 28.1 }, ...]
   *
   * Rolling window: keep only the last `maxDataPoints` frames globally.
   * Frames where a player has no data are left as `undefined` so Recharts
   * renders a gap (connectNulls=false).
   */
  const chartData = useMemo(() => {
    const frameMap = new Map<number, Record<string, number>>()

    for (const idStr of playerIds) {
      const series = data[Number(idStr)]
      if (!series) continue
      // Pre-slice per player so the map never grows beyond maxDataPoints frames
      const points = series.dataPoints.slice(-maxDataPoints)
      for (const pt of points) {
        let entry = frameMap.get(pt.frame)
        if (!entry) {
          entry = { frame: pt.frame }
          frameMap.set(pt.frame, entry)
        }
        entry[idStr] = pt.value
      }
    }

    return Array.from(frameMap.values())
      .sort((a, b) => a.frame - b.frame)
      .slice(-maxDataPoints) // rolling window across all players
  }, [data, playerIds, maxDataPoints])

  const hasData = chartData.length > 0

  const togglePlayer = useCallback((id: string) => {
    setHidden((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }, [])

  // Wrap TooltipContent with current data — useCallback so the reference is
  // stable between renders unless `data` itself changes.
  const renderTooltip = useCallback(
    (props: TooltipProps<number, string>) => (
      <TooltipContent {...props} seriesMap={data} />
    ),
    [data],
  )

  return (
    <div className="bg-surface rounded-xl border border-gray-100 shadow-sm p-5">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-semibold text-text-secondary">
          {metricLabel}
        </span>
        {isApprox && (
          <span className="flex items-center gap-1 text-xs text-warning bg-warning/10 px-2 py-0.5 rounded-full font-medium select-none">
            ⚠️ Estimasi CV
          </span>
        )}
      </div>

      {/* ── Legend (toggle per player) ──────────────────────────────────── */}
      {playerIds.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {playerIds.map((id, idx) => {
            const series = data[Number(id)]
            if (!series) return null
            const color = series.color || PALETTE[idx % PALETTE.length]
            const isHidden = hidden.has(id)

            return (
              <button
                key={id}
                type="button"
                onClick={() => togglePlayer(id)}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border
                  transition-smooth cursor-pointer select-none ${
                    isHidden
                      ? 'bg-gray-100 border-gray-200 text-text-secondary opacity-50'
                      : 'bg-surface border-gray-300 text-text-primary hover:border-gray-400'
                  }`}
              >
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: isHidden ? '#D1D5DB' : color }}
                />
                {series.name}
              </button>
            )
          })}
        </div>
      )}

      {/* ── Chart or Skeleton ────────────────────────────────────────────── */}
      {!hasData ? (
        <ChartSkeleton height={height} />
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart
            data={chartData}
            margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#F3F4F6"
              vertical={false}
            />
            <XAxis
              dataKey="frame"
              tick={{ fontSize: 10, fill: '#6B7280' }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
              label={{
                value: 'Frame',
                position: 'insideBottomRight',
                offset: -4,
                fontSize: 10,
                fill: '#6B7280',
              }}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#6B7280' }}
              tickLine={false}
              axisLine={false}
              domain={['auto', 'auto']}
              width={44}
            />
            <Tooltip
              content={renderTooltip}
              cursor={{ stroke: '#E5E7EB', strokeWidth: 1 }}
            />
            {playerIds.map((id, idx) => {
              const series = data[Number(id)]
              if (!series) return null
              const color = series.color || PALETTE[idx % PALETTE.length]
              const isHidden = hidden.has(id)

              return (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={id}
                  stroke={color}
                  strokeWidth={2}
                  // Keep the element in the DOM; just make it invisible.
                  // Avoids Recharts re-registering / re-animating on toggle.
                  strokeOpacity={isHidden ? 0 : 1}
                  dot={false}
                  activeDot={
                    isHidden ? false : { r: 4, strokeWidth: 0, fill: color }
                  }
                  // Disable built-in animation: for a rolling real-time chart,
                  // update frequency (not animation) creates the smooth effect.
                  isAnimationActive={false}
                  connectNulls={false}
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
