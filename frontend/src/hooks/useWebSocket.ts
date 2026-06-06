import { useRef, useCallback } from 'react'
import { FrameUpdate, GameEvent, PlayerStats, MpiMetrics } from '../types'
import { useMatchStore } from '../store/matchStore'
import { MockWebSocketServer } from '../utils/mockWebSocket'

const buildInitialStats = (roster: { jerseyNumber: number; name: string; team: 'A' | 'B' }[]): Record<number, PlayerStats> => {
  const stats: Record<number, PlayerStats> = {}
  roster.forEach((p, i) => {
    const id = i + 1
    stats[id] = {
      playerId: id,
      name: p.name,
      jerseyNumber: p.jerseyNumber,
      team: p.team,
      minutes: 0,
      pts: 0,
      twoPointMade: 0,
      twoPointAtt: 0,
      threePointMade: 0,
      threePointAtt: 0,
      ftMade: 0,
      ftAtt: 0,
      fgPercent: 0,
      offReb: 0,
      defReb: 0,
      totReb: 0,
      ast: 0,
      stl: 0,
      tov: 0,
      fouls: 0,
      blocks: 0,
      plusMinus: 0,
      eff: 0,
    }
  })
  return stats
}

const buildInitialMpi = (stats: Record<number, PlayerStats>): Record<number, MpiMetrics> => {
  const mpi: Record<number, MpiMetrics> = {}
  Object.values(stats).forEach((p) => {
    mpi[p.playerId] = {
      playerId: p.playerId,
      quarter: 1,
      distanceCoveredM: 0,
      avgSpeedKmh: 0,
      maxSpeedKmh: 0,
      jumpHeightCm: 0,
      accelerationMs2: 0,
      agility: 0,
      endurance: 0,
      fatigue: 0,
      mpiComposite: 0,
    }
  })
  return mpi
}

const calcEff = (p: PlayerStats): number =>
  (p.pts + p.totReb + p.ast + p.stl + p.blocks) -
  (p.twoPointAtt + p.threePointAtt - p.twoPointMade - p.threePointMade + p.tov)

export const useWebSocket = () => {
  const mockServerRef = useRef<MockWebSocketServer | null>(null)

  const connect = useCallback(() => {
    const store = useMatchStore.getState()
    const { roster } = store

    const initialStats = buildInitialStats(roster)
    const initialMpi = buildInitialMpi(initialStats)
    store.setStats(initialStats)
    store.setMpi(initialMpi)
    store.setIsLive(true)

    const handleFrameUpdate = (frame: FrameUpdate) => {
      const s = useMatchStore.getState()
      s.setScore(frame.score.teamA, frame.score.teamB)
      s.setQuarter(frame.quarter)
      s.setGameClock(frame.gameClock)
      s.setPlayers(frame.players)
      s.setBall(frame.ball)
      s.setPossession(frame.possession)

      // Tick minutes for all players (each frame = 0.1 s = 1/600 min)
      const updatedStats = { ...s.stats }
      Object.values(updatedStats).forEach((p) => {
        updatedStats[p.playerId] = { ...p, minutes: p.minutes + 1 / 600 }
      })

      // Update MPI metrics from player speed data
      const updatedMpi = { ...s.mpi }
      frame.players.forEach((player) => {
        const pid = player.trackId
        if (!updatedMpi[pid]) return
        const prev = updatedMpi[pid]
        const newDist = prev.distanceCoveredM + (player.speedKmh / 36000) // km/h × 0.1s → m
        const newMax = Math.max(prev.maxSpeedKmh, player.speedKmh)
        const newAvg = prev.avgSpeedKmh === 0
          ? player.speedKmh
          : prev.avgSpeedKmh * 0.99 + player.speedKmh * 0.01
        const mpiComposite = Math.min(100,
          40 + (newDist / 50) * 20 + (newAvg / 25) * 20 + Math.random() * 5
        )
        updatedMpi[pid] = {
          ...prev,
          distanceCoveredM: newDist,
          maxSpeedKmh: newMax,
          avgSpeedKmh: newAvg,
          mpiComposite,
          jumpHeightCm: prev.jumpHeightCm || 40 + Math.random() * 30,
          accelerationMs2: prev.accelerationMs2 || 2 + Math.random() * 3,
          agility: prev.agility || 60 + Math.random() * 30,
          endurance: Math.min(100, 50 + newDist / 80),
          fatigue: Math.min(100, newDist / 50),
        }
      })

      s.setStats(updatedStats)
      s.setMpi(updatedMpi)
    }

    const handleEvent = (event: GameEvent) => {
      const s = useMatchStore.getState()
      s.addEvent(event)

      const currentStats = { ...s.stats }
      const playerEntry = Object.values(currentStats).find(
        (p) => p.jerseyNumber === event.playerId || p.playerId === event.playerId
      )
      if (!playerEntry) return

      const p = { ...playerEntry }

      switch (event.eventType) {
        case 'FGM':
          if (event.points === 3) {
            p.threePointMade++
            p.threePointAtt++
          } else {
            p.twoPointMade++
            p.twoPointAtt++
          }
          p.pts += event.points ?? 2
          break
        case 'FGA':
          if (Math.random() > 0.5) p.threePointAtt++
          else p.twoPointAtt++
          break
        case 'REB':
          if (Math.random() > 0.3) { p.defReb++; } else { p.offReb++ }
          p.totReb++
          break
        case 'AST':
          p.ast++
          break
        case 'STL':
          p.stl++
          break
        case 'BLK':
          p.blocks++
          break
        case 'TOV':
          p.tov++
          break
        case 'FOUL':
          p.fouls++
          break
      }

      const totalAtt = p.twoPointAtt + p.threePointAtt
      p.fgPercent = totalAtt > 0 ? ((p.twoPointMade + p.threePointMade) / totalAtt) * 100 : 0
      p.plusMinus = event.team === 'A'
        ? useMatchStore.getState().teamA.score - useMatchStore.getState().teamB.score
        : useMatchStore.getState().teamB.score - useMatchStore.getState().teamA.score
      p.eff = calcEff(p)

      currentStats[p.playerId] = p
      s.setStats(currentStats)
    }

    // Convert roster to PlayerStats[] for MockWebSocketServer
    const mockRoster = Object.values(initialStats)
    mockServerRef.current = new MockWebSocketServer(handleFrameUpdate, handleEvent, mockRoster)
    mockServerRef.current.start()
  }, [])

  const disconnect = useCallback(() => {
    mockServerRef.current?.stop()
    mockServerRef.current = null
    useMatchStore.getState().setIsLive(false)
  }, [])

  return { connect, disconnect }
}
