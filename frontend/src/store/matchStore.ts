import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { MatchState, Player, GameEvent, PlayerStats, MpiMetrics, BallState, RosterPlayer, Sponsor } from '../types'

const initialState: MatchState = {
  matchId: '',
  region: '',
  teamA: { name: '', score: 0 },
  teamB: { name: '', score: 0 },
  quarter: 1,
  gameClock: '12:00',
  shotClock: 24,
  possession: { teamA: 0.5, teamB: 0.5 },
  isLive: false,
  wsStatus: 'idle',
  isVideoMode: false,
  players: [],
  playersVideoTs: 0,
  ball: null,
  events: [],
  stats: {},
  mpi: {},
  roster: [],
  sponsors: [],
  videoUrl: '',
  matchStep: 'setup',
  pipelineFps: 0,
  gpuMetrics: { gpu: 0, vramUsed: 0, vramTotal: 0 },
  quarterScoreA: 0,
  quarterScoreB: 0,
  jerseyColorA: '',
  jerseyColorB: '',
  detectedColorA: '',
  detectedColorB: '',
}

export const useMatchStore = create<MatchState & {
  setMatchId: (id: string) => void
  setRegion: (region: string) => void
  setTeamA: (name: string) => void
  setTeamB: (name: string) => void
  setScore: (teamA: number, teamB: number) => void
  setQuarter: (q: number) => void
  setGameClock: (clock: string) => void
  setShotClock: (clock: number) => void
  setPossession: (possession: { teamA: number; teamB: number }) => void
  setIsLive: (live: boolean) => void
  setWsStatus: (status: MatchState['wsStatus']) => void
  setIsVideoMode: (v: boolean) => void
  setPlayers: (players: Player[], videoTs?: number) => void
  setBall: (ball: BallState | null) => void
  updatePlayer: (trackId: number, player: Partial<Player>) => void
  addEvent: (event: GameEvent) => void
  clearEvents: () => void
  setStats: (stats: Record<number, PlayerStats>) => void
  updatePlayerStats: (playerId: number, updates: Partial<PlayerStats>) => void
  setMpi: (mpi: Record<number, MpiMetrics>) => void
  setRoster: (roster: RosterPlayer[]) => void
  setSponsors: (sponsors: Sponsor[]) => void
  setVideoUrl: (url: string) => void
  setMatchStep: (step: string) => void
  setPipelineMetrics: (fps: number, gpuMetrics: { gpu: number; vramUsed: number; vramTotal: number }) => void
  setQuarterScore: (a: number, b: number) => void
  setJerseyColorA: (color: string) => void
  setJerseyColorB: (color: string) => void
  setDetectedColors: (colorA: string, colorB: string) => void
  patchEventsByTrackId: (trackId: number, jerseyNumber: number, playerName: string, team: string) => void
  reset: () => void
}>()(
  persist(
    (set) => ({
      ...initialState,

      setMatchId: (id) => set({ matchId: id }),
      setRegion: (region) => set({ region }),
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
      setWsStatus: (wsStatus) => set({ wsStatus }),
      setIsVideoMode: (v) => set({ isVideoMode: v }),
      setPlayers: (players, videoTs) => set({
        players,
        playersVideoTs: videoTs ?? 0,
      }),
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
      setSponsors: (sponsors) => set({ sponsors }),
      setVideoUrl: (videoUrl) => set({ videoUrl }),
      setMatchStep: (matchStep) => set({ matchStep }),
      setPipelineMetrics: (fps, gpuMetrics) => set({ pipelineFps: fps, gpuMetrics }),
      setQuarterScore: (a, b) => set({ quarterScoreA: a, quarterScoreB: b }),
      setJerseyColorA: (color) => set({ jerseyColorA: color }),
      setJerseyColorB: (color) => set({ jerseyColorB: color }),
      setDetectedColors: (colorA, colorB) => set({ detectedColorA: colorA, detectedColorB: colorB }),
      patchEventsByTrackId: (trackId, jerseyNumber, playerName, team) =>
        set((state) => ({
          events: state.events.map((e) =>
            e.trackId === trackId && e.playerName === `Track_${trackId}`
              ? {
                  ...e,
                  playerId:   jerseyNumber,
                  playerName,
                  team: (team || e.team) as 'A' | 'B' | '',
                }
              : e
          ),
        })),
      reset: () => set(initialState),
    }),
    {
      name: 'sv-match',
      // Only persist setup data and roster — live game state always starts fresh
      partialize: (state) => ({
        matchId:      state.matchId,
        region:       state.region,
        teamA:        state.teamA,
        teamB:        state.teamB,
        roster:       state.roster,
        sponsors:     state.sponsors,
        videoUrl:     state.videoUrl,
        matchStep:    state.matchStep,
        jerseyColorA: state.jerseyColorA,
        jerseyColorB: state.jerseyColorB,
      }),
    },
  ),
)
