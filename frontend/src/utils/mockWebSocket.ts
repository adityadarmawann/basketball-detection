import { FrameUpdate, GameEvent, Player, PlayerStats } from '../types'

const COURT_LENGTH = 28
const COURT_WIDTH  = 15

// Simulate OCR confirmation delay: one player confirmed every N ticks (100ms each)
const OCR_CONFIRM_INTERVAL = 20  // ~2 seconds per player

const MOCK_ACTIONS: Player['action'][] = [
  'SHOOT', 'PASS', 'DRIBBLE', 'CATCH', 'STEAL',
  'JUMP', 'REBOUND', 'BLOCK', 'STAND',
]
const MOCK_EVENT_TYPES: GameEvent['eventType'][] = ['FGA', 'FGM', 'REB', 'AST', 'STL', 'BLK', 'TOV']

export class MockWebSocketServer {
  private intervalId: number | null = null
  private onFrameUpdate: (data: FrameUpdate) => void
  private onEvent:       (data: GameEvent) => void
  private roster:        PlayerStats[]
  private gameTimeSec  = 12 * 60
  private score        = { teamA: 0, teamB: 0 }
  private quarter      = 1
  private tickCount    = 0

  // Players confirmed so far (jersey OCR done) — starts empty, fills gradually
  private confirmedPlayers: Player[] = []
  // Queue: waiting for jersey OCR confirmation
  private pendingPlayers:   Player[] = []

  constructor(
    onFrameUpdate: (data: FrameUpdate) => void,
    onEvent:       (data: GameEvent) => void,
    roster:        PlayerStats[],
  ) {
    this.onFrameUpdate = onFrameUpdate
    this.onEvent       = onEvent
    this.roster        = roster
    this.buildPendingQueue()
  }

  private buildPendingQueue() {
    // Basketball is 5v5 — pick up to 5 starters per team
    const teamA = this.roster.filter((p) => p.team === 'A').slice(0, 5)
    const teamB = this.roster.filter((p) => p.team === 'B').slice(0, 5)

    this.pendingPlayers = [...teamA, ...teamB].map((p) => ({
      trackId:      p.playerId,
      jerseyNumber: p.jerseyNumber,
      name:         p.name,
      team:         p.team,
      bbox:         [0, 0, 0, 0] as [number, number, number, number],
      courtPos:     [Math.random() * COURT_LENGTH, Math.random() * COURT_WIDTH] as [number, number],
      action:       'STAND' as const,
      speedKmh:     0,
      keypoints:    [],
    }))

    this.confirmedPlayers = []
  }

  start() {
    if (this.intervalId) return
    this.intervalId = window.setInterval(() => this.tick(), 100) // 10 fps
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  private tick() {
    this.tickCount++
    this.gameTimeSec -= 0.1
    if (this.gameTimeSec < 0) {
      this.gameTimeSec = 12 * 60
      this.quarter++
      if (this.quarter > 4) { this.stop(); return }
    }

    // ── Simulate jersey OCR: confirm one player every OCR_CONFIRM_INTERVAL ticks ──
    if (
      this.pendingPlayers.length > 0 &&
      this.tickCount % OCR_CONFIRM_INTERVAL === 0
    ) {
      const confirmed = this.pendingPlayers.shift()!
      this.confirmedPlayers.push(confirmed)
    }

    // ── Update positions for confirmed players only ────────────────────────────
    this.confirmedPlayers.forEach((player) => {
      // Mock always initialises courtPos — non-null assertion is safe here
      const pos = player.courtPos!
      pos[0] = Math.max(0, Math.min(COURT_LENGTH, pos[0] + (Math.random() - 0.5) * 0.5))
      pos[1] = Math.max(0, Math.min(COURT_WIDTH,  pos[1] + (Math.random() - 0.5) * 0.3))

      // Perspective-aware bbox (smaller near top of frame)
      const ps         = 0.5 + (pos[1] / COURT_WIDTH) * 0.5
      const bboxWidth  = 0.05 * ps
      const bboxHeight = 0.18 * ps
      const vx = pos[0] / COURT_LENGTH
      const vy = pos[1] / COURT_WIDTH

      player.bbox = [
        vx - bboxWidth / 2, vy - bboxHeight / 2,
        vx + bboxWidth / 2, vy + bboxHeight / 2,
      ]

      if (Math.random() < 0.1) {
        player.action = MOCK_ACTIONS[Math.floor(Math.random() * MOCK_ACTIONS.length)]
      }
      player.speedKmh = Math.random() * 15
    })

    // ── Occasional game event (only from confirmed players) ───────────────────
    let gameEvent: GameEvent | null = null
    if (this.confirmedPlayers.length > 0 && Math.random() < 0.05) {
      gameEvent = this.generateRandomEvent()
      this.onEvent(gameEvent)
      if (gameEvent.eventType === 'FGM' && gameEvent.points) {
        if (gameEvent.team === 'A') this.score.teamA += gameEvent.points
        else                        this.score.teamB += gameEvent.points
      }
    }

    const frame: FrameUpdate = {
      type:       'frame_update',
      timestamp:  Date.now(),
      quarter:    this.quarter,
      gameClock:  this.formatTime(this.gameTimeSec),
      score:      { ...this.score },
      possession: { teamA: 0.55, teamB: 0.45 },
      players:    this.confirmedPlayers,   // only jersey-confirmed players
      ball: {
        bbox:       [0, 0, 0, 0],
        courtPos:   [Math.random() * COURT_LENGTH, Math.random() * COURT_WIDTH],
        trajectory: [],
      },
      event: gameEvent,
    }

    this.onFrameUpdate(frame)
  }

  private generateRandomEvent(): GameEvent {
    const pool  = this.confirmedPlayers
    const p     = pool[Math.floor(Math.random() * pool.length)]
    const etype = MOCK_EVENT_TYPES[Math.floor(Math.random() * MOCK_EVENT_TYPES.length)]
    const points = etype === 'FGM' ? (Math.random() > 0.3 ? 2 : 3) : undefined

    return {
      type:       'event',
      eventType:  etype,
      playerId:   p.trackId,
      playerName: p.name,
      team:       (p.team || 'A') as 'A' | 'B',
      points,
      quarter:    this.quarter,
      gameClock:  this.formatTime(this.gameTimeSec),
      courtPos:   [Math.random() * COURT_LENGTH, Math.random() * COURT_WIDTH],
    }
  }

  private formatTime(seconds: number): string {
    if (!isFinite(seconds) || seconds < 0) return '00:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
}
