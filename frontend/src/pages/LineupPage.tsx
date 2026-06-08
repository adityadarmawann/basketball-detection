import { useMatchStore } from '../store/matchStore'
import MatchHeader from '../components/match/MatchHeader'
import PlayerStatsTable from '../components/lineup/PlayerStatsTable'
import QuarterFilter from '../components/match/QuarterFilter'
import { Download } from 'lucide-react'
import { useMemo, useState } from 'react'
import Papa from 'papaparse'
import { GameEvent, PlayerStats } from '../types'

export default function LineupPage() {
  const store = useMatchStore()
  const [selectedQuarter, setSelectedQuarter] = useState(0)

  const { teamAPlayers, teamBPlayers } = useMemo(() => {
    if (selectedQuarter === 0) {
      const all = Object.values(store.stats)
      return {
        teamAPlayers: all.filter((p) => p.team === 'A'),
        teamBPlayers: all.filter((p) => p.team === 'B'),
      }
    }

    // Build zeroed-out per-player stats for the selected quarter from events
    const perPlayer: Record<number, PlayerStats> = {}
    Object.values(store.stats).forEach((p) => {
      perPlayer[p.playerId] = {
        ...p,
        pts: 0, twoPointMade: 0, twoPointAtt: 0,
        threePointMade: 0, threePointAtt: 0,
        ftMade: 0, ftAtt: 0, fgPercent: 0,
        offReb: 0, defReb: 0, totReb: 0,
        ast: 0, stl: 0, tov: 0, fouls: 0, blocks: 0,
        plusMinus: 0, eff: 0,
      }
    })

    store.events
      .filter((e: GameEvent) => e.quarter === selectedQuarter)
      .forEach((e: GameEvent) => {
        const entry = Object.values(perPlayer).find(
          (p) => p.jerseyNumber === e.playerId || p.playerId === e.playerId
        )
        if (!entry) return
        switch (e.eventType) {
          case 'FGM':
            if (e.points === 3) { entry.threePointMade++; entry.threePointAtt++ }
            else { entry.twoPointMade++; entry.twoPointAtt++ }
            entry.pts += e.points ?? 2
            break
          case 'REB': entry.totReb++;  break
          case 'AST': entry.ast++;     break
          case 'STL': entry.stl++;     break
          case 'BLK': entry.blocks++;  break
          case 'TOV': entry.tov++;     break
          case 'FOUL': entry.fouls++;  break
        }
        const totalAtt = entry.twoPointAtt + entry.threePointAtt
        entry.fgPercent = totalAtt > 0
          ? ((entry.twoPointMade + entry.threePointMade) / totalAtt) * 100
          : 0
        entry.eff =
          (entry.pts + entry.totReb + entry.ast + entry.stl + entry.blocks) -
          (totalAtt - entry.twoPointMade - entry.threePointMade + entry.tov)
      })

    const all = Object.values(perPlayer)
    return {
      teamAPlayers: all.filter((p) => p.team === 'A'),
      teamBPlayers: all.filter((p) => p.team === 'B'),
    }
  }, [store.stats, store.events, selectedQuarter])

  const handleQuarterChange = (q: number) => {
    setSelectedQuarter(q)
  }

  const handleExport = () => {
    const allPlayers = [...teamAPlayers, ...teamBPlayers]
    if (allPlayers.length === 0) {
      alert('Tidak ada data pemain untuk diekspor.')
      return
    }

    const csvData = allPlayers.map((p) => ({
      '#': p.jerseyNumber,
      Name: p.name,
      Team: p.team,
      MIN: (p.minutes / 60).toFixed(1),
      PTS: p.pts,
      '2P': `${p.twoPointMade}/${p.twoPointAtt}`,
      '3P': `${p.threePointMade}/${p.threePointAtt}`,
      FT: `${p.ftMade}/${p.ftAtt}`,
      'FG%': p.fgPercent.toFixed(1),
      REB: p.totReb,
      AST: p.ast,
      STL: p.stl,
      BLK: p.blocks,
      TOV: p.tov,
      '+/-': p.plusMinus,
      EFF: p.eff,
    }))

    const csv = Papa.unparse(csvData)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)
    link.setAttribute('href', url)
    link.setAttribute('download', `match-stats-${store.matchId || 'export'}-Q${selectedQuarter || 'all'}.csv`)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Sticky Header */}
      <div className="sticky top-0 bg-background/80 backdrop-blur-sm py-4 z-20">
        <MatchHeader
          teamA={store.teamA.name}
          teamB={store.teamB.name}
          scoreA={store.teamA.score}
          scoreB={store.teamB.score}
          quarter={store.quarter}
          gameClock={store.gameClock}
          shotClock={store.shotClock}
          isLive={store.isLive}
        />
      </div>

      <div className="my-6 flex justify-between items-center">
        <div className="w-full max-w-md">
          <QuarterFilter onFilterChange={handleQuarterChange} />
        </div>
        <button
          onClick={handleExport}
          className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark transition-smooth font-medium text-sm"
        >
          <Download size={16} /> Export CSV
        </button>
      </div>

      <div className="space-y-8">
        <PlayerStatsTable players={teamAPlayers} teamName={store.teamA.name} />
        <PlayerStatsTable players={teamBPlayers} teamName={store.teamB.name} />
      </div>
    </div>
  )
}
