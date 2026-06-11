import { useRef } from 'react'
import { GameEvent } from '../../types'

const getEventIcon = (eventType: string) => {
  const icons: Record<string, string> = {
    FGM: '✓',
    FGA: '🎯',
    REB: '📦',
    AST: '🤝',
    STL: '🚨',
    BLK: '🚫',
    TOV: '💥',
    FOUL: '⚠️',
  }
  return icons[eventType] || '•'
}

interface LiveEventFeedProps {
  events: GameEvent[]
  teamA: string
  teamB: string
}

export default function LiveEventFeed({
  events,
  teamA: _teamA,
  teamB: _teamB,
}: LiveEventFeedProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null)

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <h3 className="font-display text-xl font-bold mb-4">LIVE EVENT LOG</h3>

      <div
        ref={scrollContainerRef}
        className="space-y-2 max-h-64 overflow-y-auto"
      >
        {events.length === 0 ? (
          <div className="text-center py-8 text-text-secondary">
            Menunggu event...
          </div>
        ) : (
          [...events].reverse().map((event, idx) => (
            <div
              key={idx}
              className={`p-3 rounded-lg flex gap-3 text-sm border-l-4 ${
                event.team === 'A'
                  ? 'border-team-a bg-blue-50'
                  : 'border-team-b bg-gray-50'
              }`}
            >
              <span className="text-lg flex-shrink-0">
                {getEventIcon(event.eventType)}
              </span>
              <div className="flex-1">
                <div className="font-bold">
                  {event.gameClock} — {event.eventType}
                </div>
                <div className="text-xs text-text-secondary">
                  {event.playerName?.startsWith('Track_')
                    ? <span className="italic text-orange-400">?? Unidentified</span>
                    : `#${event.playerId} ${event.playerName}`
                  }
                  {event.points && ` (+${event.points})`}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
