interface ScoreboardProps {
  teamA: string
  teamB: string
  scoreA: number
  scoreB: number
  quarter: number
  gameClock: string
  shotClock: number
}

export default function Scoreboard({
  teamA,
  teamB,
  scoreA,
  scoreB,
  quarter,
  gameClock,
  shotClock,
}: ScoreboardProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      {/* Quarter Info */}
      <div className="bg-surface rounded-lg p-4 shadow-sm">
        <div className="text-text-secondary text-xs font-bold mb-2">QUARTER</div>
        <div className="font-display text-3xl font-bold">Q{quarter}</div>
      </div>

      {/* Game Clock */}
      <div className="bg-surface rounded-lg p-4 shadow-sm">
        <div className="text-text-secondary text-xs font-bold mb-2">TIME</div>
        <div className="font-mono text-3xl font-bold text-primary">
          {gameClock}
        </div>
      </div>

      {/* Score Summary */}
      <div className="col-span-2 bg-surface rounded-lg p-4 shadow-sm">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-text-secondary text-xs font-bold">
              {teamA}
            </div>
            <div className="font-display text-4xl font-bold text-primary">
              {scoreA}
            </div>
          </div>
          <div className="flex flex-col items-center justify-center">
            <div className="text-text-secondary text-xs font-bold mb-2">
              SHOT
            </div>
            <div className="text-2xl font-bold text-warning">
              {shotClock}
            </div>
          </div>
          <div>
            <div className="text-text-secondary text-xs font-bold">
              {teamB}
            </div>
            <div className="font-display text-4xl font-bold text-primary">
              {scoreB}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
