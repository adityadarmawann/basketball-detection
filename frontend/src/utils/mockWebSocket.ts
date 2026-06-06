import { FrameUpdate, GameEvent, Player, PlayerStats } from '../types'

const COURT_LENGTH = 28 // meters
const COURT_WIDTH = 15 // meters

const MOCK_ACTIONS: Player['action'][] = ['RUN', 'DRIBBLE', 'STAND', 'PASS', 'SHOOT']
const MOCK_EVENT_TYPES: GameEvent['eventType'][] = ['FGA', 'FGM', 'REB', 'AST', 'STL', 'BLK', 'TOV']

export class MockWebSocketServer {
  private intervalId: number | null = null
  private onFrameUpdate: (data: FrameUpdate) => void
  private onEvent: (data: GameEvent) => void
  private roster: PlayerStats[]
  private gameTimeSec = 12 * 60 // 12 min quarter
  private score = { teamA: 0, teamB: 0 }
  private quarter = 1
  private players: Player[] = []

  constructor(
    onFrameUpdate: (data: FrameUpdate) => void,
    onEvent: (data: GameEvent) => void,
    roster: PlayerStats[]
  ) {
    this.onFrameUpdate = onFrameUpdate
    this.onEvent = onEvent
    this.roster = roster
    this.initializePlayers()
  }

  private initializePlayers() {
    this.players = this.roster.map((p) => ({
      trackId: p.playerId,
      jerseyNumber: p.jerseyNumber,
      name: p.name,
      team: p.team,
      bbox: [0, 0, 0, 0],
      courtPos: [
        Math.random() * COURT_LENGTH,
        Math.random() * COURT_WIDTH,
      ],
      action: 'STAND',
      speedKmh: 0,
      keypoints: [],
    }))
  }

  start() {
    if (this.intervalId) return
    this.intervalId = window.setInterval(() => this.tick(), 100) // 10fps
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  private tick() {
    this.gameTimeSec -= 0.1
    if (this.gameTimeSec < 0) {
      this.gameTimeSec = 12 * 60
      this.quarter++
      if (this.quarter > 4) this.stop()
    }

    // Update player positions and actions
    this.players.forEach((player) => {
      // Simple random walk
      player.courtPos[0] += (Math.random() - 0.5) * 0.5
      player.courtPos[1] += (Math.random() - 0.5) * 0.3

      // Keep players on the court
      player.courtPos[0] = Math.max(0, Math.min(COURT_LENGTH, player.courtPos[0]))
      player.courtPos[1] = Math.max(0, Math.min(COURT_WIDTH, player.courtPos[1]))

      // Mock BBox based on court position (very simplified)
      // This simulates perspective by making players smaller further up the court (lower y)
      const perspectiveScale = 0.6 + (player.courtPos[1] / COURT_WIDTH) * 0.4;
      const bboxWidth = 0.08 * perspectiveScale;
      const bboxHeight = 0.3 * perspectiveScale;
      const videoX = player.courtPos[0] / COURT_LENGTH;
      const videoY = player.courtPos[1] / COURT_WIDTH;

      player.bbox = [
        videoX - bboxWidth / 2, videoY - bboxHeight / 2,
        videoX + bboxWidth / 2, videoY + bboxHeight / 2,
      ];

      if (Math.random() < 0.1) {
        player.action = MOCK_ACTIONS[Math.floor(Math.random() * MOCK_ACTIONS.length)]
      }
      player.speedKmh = Math.random() * 15
    })

    // Occasionally generate a game event
    let gameEvent: GameEvent | null = null
    if (Math.random() < 0.05) { // 5% chance per tick
      gameEvent = this.generateRandomEvent()
      this.onEvent(gameEvent)

      // Update score if it's a scoring event
      if (gameEvent.eventType === 'FGM' && gameEvent.points) {
        const player = this.roster.find(p => p.playerId === gameEvent!.playerId)
        if (player?.team === 'A') {
          this.score.teamA += gameEvent.points
        } else {
          this.score.teamB += gameEvent.points
        }
      }
    }

    const frame: FrameUpdate = {
      type: 'frame_update',
      timestamp: Date.now(),
      quarter: this.quarter,
      gameClock: this.formatTime(this.gameTimeSec),
      score: { ...this.score },
      possession: { teamA: 0.55, teamB: 0.45 },
      players: this.players,
      ball: {
        bbox: [0, 0, 0, 0],
        courtPos: [
          Math.random() * COURT_LENGTH,
          Math.random() * COURT_WIDTH,
        ],
        trajectory: [],
      },
      event: gameEvent,
    }

    this.onFrameUpdate(frame)
  }

  private generateRandomEvent(): GameEvent {
    const randomPlayer = this.roster[Math.floor(Math.random() * this.roster.length)]
    const eventType = MOCK_EVENT_TYPES[Math.floor(Math.random() * MOCK_EVENT_TYPES.length)]
    let points
    if (eventType === 'FGM') {
      points = Math.random() > 0.3 ? 2 : 3
    }

    return {
      type: 'event',
      eventType: eventType,
      playerId: randomPlayer.playerId,
      playerName: randomPlayer.name,
      team: randomPlayer.team,
      points,
      quarter: this.quarter,
      gameClock: this.formatTime(this.gameTimeSec),
      courtPos: [
        Math.random() * COURT_LENGTH,
        Math.random() * COURT_WIDTH,
      ],
    }
  }

  private formatTime(seconds: number): string {
    if (!isFinite(seconds) || seconds < 0) return '00:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }
}