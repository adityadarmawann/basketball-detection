import { useState } from 'react'
import { useMatchStore } from '../store/matchStore'
import MpiMetricsCard from '../components/mpi/MpiMetricsCard'
import ApproxWarning from '../components/mpi/ApproxWarning'

export default function MpiPage() {
  const [selectedQuarter, setSelectedQuarter] = useState(1)
  const { mpi, stats } = useMatchStore()

  const mpiArray = Object.entries(mpi)
    .filter(([_, m]) => m.quarter === selectedQuarter)
    .sort((a, b) => b[1].mpiComposite - a[1].mpiComposite)

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="bg-gradient-to-r from-primary-dark to-primary rounded-lg p-8 mb-8 text-white">
        <h1 className="font-display text-4xl font-bold">Physical Metrics (MPI)</h1>
        <p className="text-primary-dark/80 mt-2">Movement & Performance Index analysis</p>
      </div>

      <ApproxWarning />

      {/* Quarter Filter */}
      <div className="flex gap-2 mb-6">
        {[1, 2, 3, 4].map((q) => (
          <button
            key={q}
            onClick={() => setSelectedQuarter(q)}
            className={`px-4 py-2 rounded font-bold transition-smooth ${
              selectedQuarter === q
                ? 'bg-primary text-white'
                : 'bg-surface text-text-primary hover:bg-gray-100'
            }`}
          >
            Quarter {q}
          </button>
        ))}
      </div>

      {/* MPI Cards Grid */}
      {mpiArray.length > 0 ? (
        <div className="grid gap-6">
          {mpiArray.map(([playerId, metrics]) => {
            const player = stats[parseInt(playerId)]
            return (
              <MpiMetricsCard
                key={playerId}
                metrics={metrics}
                playerName={
                  player
                    ? `#${player.jerseyNumber} ${player.name}`
                    : `Player ${playerId}`
                }
              />
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12 text-text-secondary bg-surface rounded-lg">
          <p className="mb-4">Belum ada data MPI.</p>
          <p className="text-sm">Metrics akan tersedia setelah video diproses.</p>
        </div>
      )}
    </div>
  )
}
