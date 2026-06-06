interface MatchHeaderProps {
  teamA: string
  teamB: string
  scoreA: number
  scoreB: number
  quarter: number
  gameClock: string
  shotClock: number
  isLive: boolean
}

export default function MatchHeader({
  teamA,
  teamB,
  scoreA,
  scoreB,
  quarter,
  gameClock,
  shotClock,
  isLive,
}: MatchHeaderProps) {
  return (
    <div className="bg-surface-dark text-white rounded-lg p-6 shadow-lg mb-6">
      <div className="grid grid-cols-5 gap-4 items-center">
        {/* Team A */}
        <div className="text-center">
          <div className="font-display text-sm font-bold text-text-secondary mb-1">
            {teamA}
          </div>
          <div className="font-display text-4xl font-bold text-white">
            {scoreA}
          </div>
        </div>

        {/* Separator */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-text-secondary text-xs font-bold">
            Q{quarter}
          </div>
          <div className="w-1 h-12 bg-primary rounded-full"></div>
        </div>

        {/* Game Clock */}
        <div className="text-center">
          <div className="font-mono text-3xl font-bold text-primary mb-1">
            {gameClock}
          </div>
          <div className="text-xs text-text-secondary">
            {isLive && <span className="badge-live">● LIVE</span>}
          </div>
        </div>

        {/* Separator */}
        <div className="flex flex-col items-center gap-2">
          <div className="text-text-secondary text-xs font-bold">
            SHOT
          </div>
          <div className="w-1 h-12 bg-primary rounded-full"></div>
        </div>

        {/* Team B */}
        <div className="text-center">
          <div className="font-display text-sm font-bold text-text-secondary mb-1">
            {teamB}
          </div>
          <div className="font-display text-4xl font-bold text-white">
            {scoreB}
          </div>
        </div>
      </div>

      {/* Shot Clock Bar */}
      <div className="mt-6 flex items-center gap-3">
        <span className="text-xs font-bold text-text-secondary">SHOT CLOCK:</span>
        <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-300"
            style={{ width: `${(shotClock / 24) * 100}%` }}
          ></div>
        </div>
        <span className="text-lg font-bold text-primary w-8 text-right">
          {shotClock}
        </span>
      </div>
    </div>
  )
}
