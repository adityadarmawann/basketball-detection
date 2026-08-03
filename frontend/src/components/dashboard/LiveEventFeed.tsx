import { useRef, useState, useEffect } from 'react'
import axios from 'axios'
import { GameEvent } from '../../types'

const API = import.meta.env.VITE_API_URL ?? ''

const getEventIcon = (eventType: string) => {
  const icons: Record<string, string> = {
    FGM: '✓',
    FGA: '🎯',
    FTM: '🏀',
    FTA: '🎯',
    REB: '📦',
    AST: '🤝',
    STL: '🚨',
    BLK: '🚫',
    TOV: '💥',
    FOUL: '⚠️',
  }
  return icons[eventType] || '•'
}

function normalizeEvent(e: Record<string, unknown>): GameEvent {
  // Derive MM:SS from timestamp_ms when game_clock is missing (MongoDB-stored events)
  const tsMs = Number(e.timestamp_ms ?? 0)
  const elapsedSec = tsMs / 1000
  const computedClock = tsMs > 0
    ? `${String(Math.floor(elapsedSec / 60)).padStart(2, '0')}:${String(Math.floor(elapsedSec % 60)).padStart(2, '0')}`
    : ''

  return {
    type: 'event',
    eventType: (e.event_type ?? e.eventType ?? 'FGA') as GameEvent['eventType'],
    trackId: (e.track_id ?? e.trackId) as number | undefined,
    playerId: Number(e.player_id ?? e.playerId ?? 0),
    playerName: String(e.player_name ?? e.playerName ?? ''),
    team: (e.team ?? '') as 'A' | 'B' | '',
    points: e.points as number | undefined,
    quarter: Number(e.quarter ?? 1),
    gameClock: String(e.game_clock ?? e.gameClock ?? computedClock),
    courtPos: (e.court_pos ?? e.courtPos ?? [0, 0]) as [number, number],
  }
}

interface LiveEventFeedProps {
  events: GameEvent[]
  teamA: string
  teamB: string
  matchId?: string
  isLive?: boolean
}

export default function LiveEventFeed({
  events,
  teamA: _teamA,
  teamB: _teamB,
  matchId,
  isLive = true,
}: LiveEventFeedProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [historyEvents, setHistoryEvents] = useState<GameEvent[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState(false)

  // When match is done fetch full history from MongoDB
  useEffect(() => {
    if (isLive || !matchId) return
    setHistoryLoading(true)
    setHistoryError(false)
    axios.get(`${API}/api/events?match_id=${encodeURIComponent(matchId)}&limit=500`)
      .then((res) => {
        const raw: Record<string, unknown>[] = res.data.events ?? []
        setHistoryEvents(raw.map(normalizeEvent))
      })
      .catch(() => setHistoryError(true))
      .finally(() => setHistoryLoading(false))
  }, [matchId, isLive])

  const displayEvents = isLive ? events : historyEvents
  const title = isLive ? 'LIVE EVENT LOG' : 'RIWAYAT EVENT'

  return (
    <div className="bg-surface rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-display text-xl font-bold">{title}</h3>
        {!isLive && (
          <span className="text-xs text-text-secondary">
            {historyLoading
              ? 'Memuat riwayat...'
              : historyError
              ? 'Gagal memuat dari server'
              : `${historyEvents.length} event`}
          </span>
        )}
      </div>

      <div
        ref={scrollContainerRef}
        className="space-y-2 max-h-64 overflow-y-auto"
      >
        {historyLoading ? (
          <div className="text-center py-8 text-text-secondary text-sm">
            Memuat riwayat event...
          </div>
        ) : displayEvents.length === 0 ? (
          <div className="text-center py-8 text-text-secondary">
            {isLive ? 'Menunggu event...' : 'Tidak ada event ditemukan'}
          </div>
        ) : (
          [...displayEvents].reverse().map((event, idx) => (
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
                  {event.quarter ? ` · Q${event.quarter}` : ''}
                </div>
                <div className="text-xs text-text-secondary">
                  {event.playerName?.startsWith('Track_')
                    ? <span className="italic text-orange-400">?? Unidentified</span>
                    : `#${event.playerId} ${event.playerName}`
                  }
                  {event.points ? ` (+${event.points})` : ''}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
