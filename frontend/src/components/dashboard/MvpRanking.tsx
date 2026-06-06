import { Trophy } from 'lucide-react'

interface PlayerRank {
  jerseyNumber: number
  name: string
  eff: number
  mpi: number
}

interface MvpRankingProps {
  players: PlayerRank[]
}

export default function MvpRanking({ players }: MvpRankingProps) {
  const topPlayers = [...players].sort((a, b) => b.eff - a.eff).slice(0, 3)

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <div className="flex items-center gap-2 mb-4">
        <Trophy className="text-accent-gold" size={24} />
        <h3 className="font-display text-xl font-bold">MVP RANKING</h3>
      </div>

      <div className="space-y-3">
        {topPlayers.map((player, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg flex items-center gap-3 ${
              idx === 0 ? 'bg-accent-gold/10 border border-accent-gold' : 'bg-gray-50'
            }`}
          >
            <div className="flex-shrink-0">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-white ${
                  idx === 0
                    ? 'bg-accent-gold'
                    : idx === 1
                      ? 'bg-gray-400'
                      : 'bg-orange-600'
                }`}
              >
                {idx + 1}
              </div>
            </div>

            <div className="flex-1">
              <div className="font-bold">
                #{player.jerseyNumber} {player.name}
              </div>
              <div className="text-sm text-text-secondary">
                EFF: {player.eff} | MPI: {player.mpi.toFixed(0)}
              </div>
            </div>

            {idx === 0 && (
              <div className="text-accent-gold font-bold text-lg">⭐</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
