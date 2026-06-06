import { create } from 'zustand'
import { MatchState, Player, GameEvent, PlayerStats, MpiMetrics, BallState, RosterPlayer } from '../types'

const initialState: MatchState = {
  matchId: '',
  teamA: { name: '', score: 0 },
  teamB: { name: '', score: 0 },
  quarter: 1,
  gameClock: '12:00',
  shotClock: 24,
  possession: { teamA: 0.5, teamB: 0.5 },
  isLive: false,
  players: [],
  ball: null,
  events: [],
  stats: {},
  mpi: {},
  roster: [],
}

export const useMatchStore = create<MatchState & {
  setMatchId: (id: string) => void
  setTeamA: (name: string) => void
  setTeamB: (name: string) => void
  setScore: (teamA: number, teamB: number) => void
  setQuarter: (q: number) => void
  setGameClock: (clock: string) => void
  setShotClock: (clock: number) => void
  setPossession: (possession: { teamA: number; teamB: number }) => void
  setIsLive: (live: boolean) => void
  setPlayers: (players: Player[]) => void
  setBall: (ball: BallState | null) => void
  updatePlayer: (trackId: number, player: Partial<Player>) => void
  addEvent: (event: GameEvent) => void
  clearEvents: () => void
  setStats: (stats: Record<number, PlayerStats>) => void
  updatePlayerStats: (playerId: number, updates: Partial<PlayerStats>) => void
  setMpi: (mpi: Record<number, MpiMetrics>) => void
  setRoster: (roster: RosterPlayer[]) => void
  reset: () => void
}>((set) => ({
  ...initialState,

  setMatchId: (id) => set({ matchId: id }),
  setTeamA: (name) => set((state) => ({ teamA: { ...state.teamA, name } })),
  setTeamB: (name) => set((state) => ({ teamB: { ...state.teamB, name } })),
  setScore: (teamA, teamB) =>
    set((state) => ({
      teamA: { ...state.teamA, score: teamA },
      teamB: { ...state.teamB, score: teamB },
    })),
  setQuarter: (q) => set({ quarter: q }),
  setGameClock: (clock) => set({ gameClock: clock }),
  setShotClock: (clock) => set({ shotClock: clock }),
  setPossession: (possession) => set({ possession }),
  setIsLive: (live) => set({ isLive: live }),
  setPlayers: (players) => set({ players }),
  setBall: (ball) => set({ ball }),
  updatePlayer: (trackId, updates) =>
    set((state) => ({
      players: state.players.map((p) =>
        p.trackId === trackId ? { ...p, ...updates } : p
      ),
    })),
  addEvent: (event) =>
    set((state) => ({
      events: [event, ...state.events].slice(0, 100),
    })),
  clearEvents: () => set({ events: [] }),
  setStats: (stats) => set({ stats }),
  updatePlayerStats: (playerId, updates) =>
    set((state) => ({
      stats: {
        ...state.stats,
        [playerId]: { ...state.stats[playerId], ...updates },
      },
    })),
  setMpi: (mpi) => set({ mpi }),
  setRoster: (roster) => set({ roster }),
  reset: () => set(initialState),
}))
