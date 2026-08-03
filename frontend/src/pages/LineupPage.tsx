import { useMatchStore } from '../store/matchStore'
import { useMatchStats } from '../hooks/useMatchStats'
import MatchHeader from '../components/match/MatchHeader'
import PlayerStatsTable from '../components/lineup/PlayerStatsTable'
import QuarterFilter from '../components/match/QuarterFilter'
import { Download } from 'lucide-react'
import { useMemo, useState } from 'react'
import Papa from 'papaparse'

export default function LineupPage() {
  const store = useMatchStore()
  const [selectedQuarter, setSelectedQuarter] = useState(0)

  // Single source of truth: the backend box score (whole match when quarter=0,
  // the real per-quarter stats when 1..4).  No client-side reconstruction, no
  // duplicate EFF formula — normalizePlayerStats() maps the backend numbers and
  // defers to the backend `eff`.
  const { playerStats } = useMatchStats(store.matchId, selectedQuarter)

  const teamAPlayers = useMemo(
    () => Object.values(playerStats).filter((p) => p.team === 'A'),
    [playerStats],
  )
  const teamBPlayers = useMemo(
    () => Object.values(playerStats).filter((p) => p.team === 'B'),
    [playerStats],
  )

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
      MIN: p.minutes.toFixed(1),
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
