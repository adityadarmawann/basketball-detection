import { useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle } from 'lucide-react'
import { useMatchStore } from '../../store/matchStore'
import { GameEvent, PlayerStats } from '../../types'
import EventCorrectionPanel from './EventCorrectionPanel'
import clsx from 'clsx'

type Tab = 'stats' | 'lineups' | 'correction'
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

const emptyTeamStats = () => ({ pts: 0, totReb: 0, ast: 0, stl: 0, blocks: 0, tov: 0 })

export default function TabsStatsLineup() {
  const [activeTab, setActiveTab] = useState<Tab>('stats')
  const [quarter, setQuarter] = useState<Quarter>(0)
  const navigate = useNavigate()
  const { stats, events, teamA, teamB } = useMatchStore()

  // Badge count: unidentified players with scoring events
  const unidentifiedCount = useMemo(
    () => Object.values(stats).filter(
      (p) => p.jerseyNumber == null && (p.pts > 0 || p.twoPointAtt > 0 || p.threePointAtt > 0)
    ).length,
    [stats]
  )

  const { teamAStats, teamBStats } = useMemo(() => {
    if (quarter === 0) {
      // All quarters — sum from cumulative store stats
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
    }

    // Specific quarter — aggregate directly from events
    const aggA = emptyTeamStats()
    const aggB = emptyTeamStats()
    events
      .filter((e: GameEvent) => e.quarter === quarter)
      .forEach((e: GameEvent) => {
        const agg = e.team === 'A' ? aggA : aggB
        switch (e.eventType) {
          case 'FGM': agg.pts += e.points ?? 2; break
          case 'REB': agg.totReb++;              break
          case 'AST': agg.ast++;                 break
          case 'STL': agg.stl++;                 break
          case 'BLK': agg.blocks++;              break
          case 'TOV': agg.tov++;                 break
        }
      })
    return { teamAStats: aggA, teamBStats: aggB }
  }, [stats, events, quarter])

  return (
    <div className="bg-surface rounded-lg shadow-sm overflow-hidden">
      {/* Tab Header */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('stats')}
          className={clsx(
            'flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-smooth',
            activeTab === 'stats' ? 'bg-primary text-white' : 'text-text-secondary hover:text-text-primary hover:bg-gray-50'
          )}
        >
          Stats
        </button>
        <button
          onClick={() => navigate('/match/lineups')}
          className="flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-smooth text-text-secondary hover:text-text-primary hover:bg-gray-50"
        >
          Lineups
        </button>
        <button
          onClick={() => setActiveTab('correction')}
          className={clsx(
            'flex-1 py-3 text-sm font-bold uppercase tracking-wide transition-smooth relative',
            activeTab === 'correction' ? 'bg-amber-500 text-white' : 'text-text-secondary hover:text-text-primary hover:bg-gray-50'
          )}
        >
          <span className="flex items-center justify-center gap-1">
            <AlertTriangle size={13} />
            Koreksi
          </span>
          {unidentifiedCount > 0 && activeTab !== 'correction' && (
            <span className="absolute top-1.5 right-2 w-4 h-4 rounded-full bg-danger text-white text-[10px] font-bold flex items-center justify-center">
              {unidentifiedCount}
            </span>
          )}
        </button>
      </div>

      {activeTab === 'correction' && <EventCorrectionPanel />}

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
