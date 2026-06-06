import { MpiMetrics } from '../../types'

interface MpiMetricsCardProps {
  metrics: MpiMetrics
  playerName: string
}

export default function MpiMetricsCard({
  metrics,
  playerName,
}: MpiMetricsCardProps) {
  const MetricItem = ({
    label,
    value,
    unit,
    isApprox = false,
  }: {
    label: string
    value: number | string
    unit: string
    isApprox?: boolean
  }) => (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      {isApprox && (
        <div className="text-warning text-lg mb-1">⚠️</div>
      )}
      <div className="text-xs font-bold text-text-secondary mb-1">
        {label}
      </div>
      <div className="text-lg font-bold tabular-nums">
        {typeof value === 'number' ? value.toFixed(1) : value}
      </div>
      <div className="text-xs text-text-secondary">{unit}</div>
    </div>
  )

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm mb-6">
      <h3 className="font-display text-lg font-bold mb-4">
        {playerName} — Physical Metrics
      </h3>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6">
        <MetricItem
          label="Distance Covered"
          value={metrics.distanceCoveredM / 1000}
          unit="km"
        />
        <MetricItem
          label="Avg Speed"
          value={metrics.avgSpeedKmh}
          unit="km/h"
        />
        <MetricItem
          label="Max Speed"
          value={metrics.maxSpeedKmh}
          unit="km/h"
        />
        <MetricItem
          label="Jump Height"
          value={metrics.jumpHeightCm}
          unit="cm"
          isApprox={true}
        />
        <MetricItem
          label="Acceleration"
          value={metrics.accelerationMs2}
          unit="m/s²"
          isApprox={true}
        />
        <MetricItem
          label="Agility"
          value={metrics.agility}
          unit="/100"
          isApprox={true}
        />
        <MetricItem
          label="Endurance"
          value={metrics.endurance}
          unit="/100"
          isApprox={true}
        />
        <MetricItem
          label="Fatigue"
          value={metrics.fatigue}
          unit="/100"
          isApprox={true}
        />
        <MetricItem
          label="MPI Score"
          value={metrics.mpiComposite}
          unit="/100"
        />
      </div>
    </div>
  )
}
