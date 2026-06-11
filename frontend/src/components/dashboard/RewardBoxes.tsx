import { useEffect, useState } from 'react'
import { Trophy, Star, Zap, HeartPulse, Building2 } from 'lucide-react'
import { PlayerStats, MpiMetrics, RosterPlayer, Sponsor, SponsorCategory } from '../../types'

const PLAYER_DURATION_MS  = 30_000
const SPONSOR_DURATION_MS = 15_000

interface TopPlayer {
  playerId:    number
  jerseyNumber: number | null
  name:        string
  statLabel:   string
  statValue:   string | number
  photoUrl?:   string
}

interface RewardBoxProps {
  category:    SponsorCategory
  label:       string
  Icon:        React.ElementType
  topPlayer:   TopPlayer | null
  sponsors:    Sponsor[]
  mvpSponsors: Sponsor[]   // fallback when category has no sponsors
}

function AvatarIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" className="w-full h-full">
      <circle cx="24" cy="24" r="24" fill="#E5E7EB" />
      <circle cx="24" cy="18" r="7" fill="#9CA3AF" />
      <path d="M8 42c0-8.837 7.163-16 16-16s16 7.163 16 16" stroke="#9CA3AF" strokeWidth="3" strokeLinecap="round" fill="none" />
    </svg>
  )
}

function RewardBox({ category, label, Icon, topPlayer, sponsors, mvpSponsors }: RewardBoxProps) {
  const [phase, setPhase] = useState<'player' | 'sponsor'>(topPlayer ? 'player' : 'sponsor')
  const [visible, setVisible] = useState(true)
  const activeSponsor = sponsors.length > 0 ? sponsors[0] : mvpSponsors[0] ?? null

  // Reset to player phase when top player changes
  useEffect(() => {
    if (topPlayer) {
      setVisible(false)
      setTimeout(() => { setPhase('player'); setVisible(true) }, 300)
    } else {
      setVisible(false)
      setTimeout(() => { setPhase('sponsor'); setVisible(true) }, 300)
    }
  }, [topPlayer?.playerId])

  // Cycle between player and sponsor phases
  useEffect(() => {
    if (!topPlayer) return
    const duration = phase === 'player' ? PLAYER_DURATION_MS : SPONSOR_DURATION_MS
    const timer = setTimeout(() => {
      setVisible(false)
      setTimeout(() => {
        setPhase((p) => p === 'player' ? 'sponsor' : 'player')
        setVisible(true)
      }, 300)
    }, duration)
    return () => clearTimeout(timer)
  }, [phase, topPlayer?.playerId])

  const colorMap: Record<SponsorCategory, string> = {
    mvp:       'from-amber-500 to-orange-500',
    score:     'from-blue-500 to-cyan-500',
    speed:     'from-violet-500 to-purple-500',
    endurance: 'from-emerald-500 to-green-500',
  }
  const gradient = colorMap[category]

  return (
    <div className="bg-surface rounded-xl shadow-sm overflow-hidden flex flex-col">
      {/* Header */}
      <div className={`bg-gradient-to-r ${gradient} px-4 py-2.5 flex items-center gap-2`}>
        <Icon size={16} className="text-white" />
        <span className="text-white text-xs font-bold uppercase tracking-widest">{label}</span>
        {activeSponsor && (
          <span className="ml-auto text-white/70 text-[10px] truncate max-w-[100px]">
            by {activeSponsor.companyName}
          </span>
        )}
      </div>

      {/* Body */}
      <div
        className="flex-1 flex flex-col items-center justify-center px-4 py-5 min-h-[200px] transition-opacity duration-300"
        style={{ opacity: visible ? 1 : 0 }}
      >
        {phase === 'player' && topPlayer ? (
          <>
            {/* Player photo */}
            <div className="w-20 h-20 rounded-full overflow-hidden border-4 border-white shadow-md mb-3">
              {topPlayer.photoUrl
                ? <img src={topPlayer.photoUrl} alt={topPlayer.name} className="w-full h-full object-cover" />
                : <AvatarIcon />
              }
            </div>
            {/* Jersey */}
            {topPlayer.jerseyNumber != null && (
              <span className="text-xs font-bold text-text-secondary mb-0.5">
                #{topPlayer.jerseyNumber}
              </span>
            )}
            {/* Name */}
            <p className="font-bold text-sm text-center leading-tight max-w-[140px] line-clamp-2">
              {topPlayer.name}
            </p>
            {/* Stat */}
            <div className="mt-2 flex items-baseline gap-1">
              <span className="text-2xl font-black tabular-nums">{topPlayer.statValue}</span>
              <span className="text-xs text-text-secondary">{topPlayer.statLabel}</span>
            </div>
            {/* Sponsor badge */}
            {activeSponsor && (
              <div className="mt-3 flex items-center gap-1.5 px-2.5 py-1 bg-gray-50 rounded-full border border-gray-200">
                {activeSponsor.logoUrl
                  ? <img src={activeSponsor.logoUrl} alt="logo" className="h-4 w-auto object-contain" />
                  : <Building2 size={11} className="text-gray-400" />
                }
                <span className="text-[10px] text-text-secondary font-medium truncate max-w-[100px]">
                  {activeSponsor.companyName}
                </span>
              </div>
            )}
          </>
        ) : (
          /* Sponsor idle mode */
          <>
            {activeSponsor ? (
              <>
                <div className="w-20 h-20 rounded-xl border border-gray-200 bg-white shadow-sm flex items-center justify-center p-2 mb-3">
                  {activeSponsor.logoUrl
                    ? <img src={activeSponsor.logoUrl} alt="logo" className="w-full h-full object-contain" />
                    : <Building2 size={32} className="text-gray-300" />
                  }
                </div>
                <p className="font-bold text-sm text-center">{activeSponsor.companyName}</p>
                <p className="text-xs text-text-secondary mt-1">Sponsor {label}</p>
              </>
            ) : (
              <div className="text-center text-text-secondary">
                <Icon size={32} className="mx-auto mb-2 opacity-20" />
                <p className="text-xs">Belum ada data</p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

// ── Phase indicator dots ──────────────────────────────────────────────────────

interface RewardBoxesProps {
  stats:    Record<number, PlayerStats>
  mpi:      Record<number, MpiMetrics>
  roster:   RosterPlayer[]
  sponsors: Sponsor[]
}

function getPhoto(roster: RosterPlayer[], jerseyNumber: number | null): string | undefined {
  if (jerseyNumber == null) return undefined
  return roster.find((r) => r.jerseyNumber === jerseyNumber)?.photoUrl
}

export default function RewardBoxes({ stats, mpi, roster, sponsors }: RewardBoxesProps) {
  const mvpSponsors       = sponsors.filter((s) => s.category === 'mvp')
  const scoreSponsors     = sponsors.filter((s) => s.category === 'score')
  const speedSponsors     = sponsors.filter((s) => s.category === 'speed')
  const enduranceSponsors = sponsors.filter((s) => s.category === 'endurance')

  const players = Object.values(stats).filter(p => p.jerseyNumber != null)

  // MVP: best by EFF
  const mvpPlayer = players.length
    ? [...players].sort((a, b) => b.eff - a.eff)[0]
    : null

  // Top Scorer: best by PTS
  const topScorer = players.length
    ? [...players].sort((a, b) => b.pts - a.pts)[0]
    : null

  // Fastest: best by avgSpeedKmh in MPI — only detected players
  const mpiList = Object.values(mpi).filter(m => stats[m.playerId]?.jerseyNumber != null)
  const fastestMpi = mpiList.length
    ? [...mpiList].sort((a, b) => b.avgSpeedKmh - a.avgSpeedKmh)[0]
    : null
  const fastestPlayer = fastestMpi ? stats[fastestMpi.playerId] ?? null : null

  // Endurance: best by endurance score in MPI
  const topEnduranceMpi = mpiList.length
    ? [...mpiList].sort((a, b) => b.endurance - a.endurance)[0]
    : null
  const endurancePlayer = topEnduranceMpi ? stats[topEnduranceMpi.playerId] ?? null : null

  const toTop = (p: PlayerStats | null, stat: string, val: string | number): TopPlayer | null =>
    p ? {
      playerId:    p.playerId,
      jerseyNumber: p.jerseyNumber,
      name:        p.name,
      statLabel:   stat,
      statValue:   val,
      photoUrl:    getPhoto(roster, p.jerseyNumber),
    } : null

  const boxes: RewardBoxProps[] = [
    {
      category:    'mvp',
      label:       'MVP Player',
      Icon:        Trophy,
      topPlayer:   toTop(mvpPlayer, 'EFF', mvpPlayer?.eff ?? 0),
      sponsors:    mvpSponsors,
      mvpSponsors,
    },
    {
      category:    'score',
      label:       'Top Scorer',
      Icon:        Star,
      topPlayer:   toTop(topScorer, 'PTS', topScorer?.pts ?? 0),
      sponsors:    scoreSponsors,
      mvpSponsors,
    },
    {
      category:    'speed',
      label:       'Top Speed',
      Icon:        Zap,
      topPlayer:   fastestPlayer
        ? toTop(fastestPlayer, 'km/h', fastestMpi ? fastestMpi.avgSpeedKmh.toFixed(1) : 0)
        : null,
      sponsors:    speedSponsors,
      mvpSponsors,
    },
    {
      category:    'endurance',
      label:       'Endurance',
      Icon:        HeartPulse,
      topPlayer:   endurancePlayer
        ? toTop(endurancePlayer, 'END', topEnduranceMpi ? topEnduranceMpi.endurance.toFixed(1) : 0)
        : null,
      sponsors:    enduranceSponsors,
      mvpSponsors,
    },
  ]

  return (
    <div>
      <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wide mb-3">
        🏅 Reward &amp; Appreciate
      </h3>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {boxes.map((box) => (
          <RewardBox key={box.category} {...box} />
        ))}
      </div>
    </div>
  )
}
