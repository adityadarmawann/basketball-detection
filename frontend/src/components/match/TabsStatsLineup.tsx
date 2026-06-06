import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMatchStore } from '../../store/matchStore'
import { PlayerStats } from '../../types'
import clsx from 'clsx'

type Tab = 'stats' | 'lineups'
type Quarter = 0 | 1 | 2 | 3 | 4 // 0 = All

const STAT_ROWS: { label: string; key: keyof PlayerStats }[] = [
  { label: 'PTS', key: 'pts' },
  { label: 'REB', key: 'totReb' },
  { label: 'AST', key: 'ast' },
  { label: 'STL', key: 'stl' },
  { label: 'BLK', key: 'blocks' },
  { label: 'TOV', key: 'tov' },
]

function StatBar({ label, valueA, valueB }: { label: string; valueA: number; valueB: number }) {
  const total = valueA + valueB || 1
  const pctA = (valueA / total) * 100

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1">
        <span className="font-bold tabular-nums text-team-a w-12 text-right">{valueA}</span>
        <span className="text-xs font-bold text-text-secondary uppercase tracking-wide flex-1 text-center">
          {label}
        </span>
        <span className="font-bold tabular-nums text-text-primary w-12">{valueB}</span>
      </div>
      <div className="h-2 w-full rounded-full overflow-hidden flex bg-team-b">
        <div
          className="h-full bg-team-a transition-all duration-500"
          style={{ width: `${pctA}%` }}
        />
      </div>
    </div>
  )
}

export default function TabsStatsLineup() {
  const [activeTab, setActiveTab] = useState<Tab>('stats')
  const [quarter, setQuarter] = useState<Quarter>(0)
  const navigate = useNavigate()
  const { stats, teamA, teamB } = useMatchStore()

  const { teamAStats, teamBStats } = useMemo(() => {
    const players = Object.values(stats)
    const teamAPlayers = players.filter((p) => p.team === 'A')
    const teamBPlayers = players.filter((p) => p.team === 'B')

    const sum = (arr: PlayerStats[], key: keyof PlayerStats) =>
      arr.reduce((acc, p) => acc + ((p[key] as number) || 0), 0)

    return {
      teamAStats: {
        pts: sum(teamAPlayers, 'pts'),
        totReb: sum(teamAPlayers, 'totReb'),
        ast: sum(teamAPlayers, 'ast'),
        stl: sum(teamAPlayers, 'stl'),
        blocks: sum(teamAPlayers, 'blocks'),
        tov: sum(teamAPlayers, 'tov'),
      },
      teamBStats: {
        pts: sum(teamBPlayers, 'pts'),
        totReb: sum(teamBPlayers, 'totReb'),
        ast: sum(teamBPlayers, 'ast'),
        stl: sum(teamBPlayers, 'stl'),
        blocks: sum(teamBPlayers, 'blocks'),
        tov: sum(teamBPlayers, 'tov'),
      },
    }
  }, [stats])

  return (
    <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
      {/* Tab Header */}
      <div className="flex border-b border-gray-200">
        {(['stats', 'lineups'] as Tab[]).map((tab) => (
          <button
            key={tab}
            onClick={() => {
              if (tab === 'lineups') {
                navigate('/match/lineups')
              } else {
                setActiveTab(tab)
              }
            }}
            className={clsx(
              'flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-smooth',
              activeTab === tab && tab !== 'lineups'
                ? 'bg-primary text-white'
                : 'text-text-secondary hover:text-text-primary hover:bg-gray-50'
            )}
          >
            {tab === 'stats' ? 'Stats' : 'Lineups'}
          </button>
        ))}
      </div>

      {activeTab === 'stats' && (
        <div className="p-6">
          {/* Quarter filter */}
          <div className="flex gap-2 mb-6 justify-center">
            {([0, 1, 2, 3, 4] as Quarter[]).map((q) => (
              <button
                key={q}
                onClick={() => setQuarter(q)}
                className={clsx(
                  'px-3 py-1 rounded-full text-xs font-bold transition-smooth',
                  quarter === q
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 text-text-secondary hover:bg-gray-200'
                )}
              >
                {q === 0 ? 'All' : `Q${q}`}
              </button>
            ))}
          </div>

          {/* Team header */}
          <div className="grid grid-cols-3 text-center mb-4">
            <div className="font-display font-bold text-team-a">{teamA.name || 'TIM A'}</div>
            <div className="text-xs text-text-secondary font-bold uppercase">Team Stats</div>
            <div className="font-display font-bold text-text-primary">{teamB.name || 'TIM B'}</div>
          </div>

          {/* Stat bars */}
          {STAT_ROWS.map((row) => (
            <StatBar
              key={row.key}
              label={row.label}
              valueA={(teamAStats as any)[row.key] ?? 0}
              valueB={(teamBStats as any)[row.key] ?? 0}
            />
          ))}
        </div>
      )}
    </div>
  )
}
