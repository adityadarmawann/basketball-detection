import { useState, useEffect, RefObject } from 'react'
import { useMatchStore } from '../../store/matchStore'
import { FrameBboxEntry } from '../../types'

// ── Court constants (metres — matches court.py) ───────────────────────────────
const CW = 28.0
const CH = 15.0

// ── SVG viewport ─────────────────────────────────────────────────────────────
const W = 280
const H = Math.round(W * CH / CW)     // 150
const S = W / CW                      // px/metre ≈ 10

const mx = (m: number) => +(m * S).toFixed(2)
const my = (m: number) => +((CH - m) * S).toFixed(2)

// ── Arc helpers ───────────────────────────────────────────────────────────────
function arcPath(hx: number, hy: number, r: number, degFrom: number, degTo: number): string {
  const pts: string[] = []
  for (let d = degFrom; d <= degTo; d++) {
    const rad = (d * Math.PI) / 180
    pts.push(`${mx(hx + r * Math.cos(rad))},${my(hy + r * Math.sin(rad))}`)
  }
  if (!pts.length) return ''
  return `M ${pts[0]} ${pts.slice(1).map(p => `L ${p}`).join(' ')}`
}

function arc3PT(hx: number, hy: number, faceRight: boolean): string {
  const pts: string[] = []
  for (let d = 0; d <= 360; d++) {
    const rad = (d * Math.PI) / 180
    const x = hx + 6.75 * Math.cos(rad)
    const y = hy + 6.75 * Math.sin(rad)
    if (y < 0.9 || y > 14.1) continue
    if (faceRight ? x <= hx : x >= hx) continue
    pts.push(`${mx(x)},${my(y)}`)
  }
  if (!pts.length) return ''
  return `M ${pts[0]} ${pts.slice(1).map(p => `L ${p}`).join(' ')}`
}

// ── Static paths ──────────────────────────────────────────────────────────────
const PATH_CTR_L      = arcPath(14.0, 7.5, 1.8,  90, 270)
const PATH_CTR_R      = arcPath(14.0, 7.5, 1.8, -90,  90)
const PATH_FT_L_SOLID = arcPath( 5.8, 7.5, 1.8, -90,  90)
const PATH_FT_L_DASH  = arcPath( 5.8, 7.5, 1.8,  90, 270)
const PATH_FT_R_SOLID = arcPath(22.2, 7.5, 1.8,  90, 270)
const PATH_FT_R_DASH  = arcPath(22.2, 7.5, 1.8, -90,  90)
const PATH_RA_L       = arcPath( 1.575, 7.5, 1.25, -90,  90)
const PATH_RA_R       = arcPath(26.425, 7.5, 1.25,  90, 270)
const PATH_3PT_L      = arc3PT( 1.575, 7.5,  true)
const PATH_3PT_R      = arc3PT(26.425, 7.5, false)

// ── Binary search (same as CourtOverlay) ─────────────────────────────────────
function findNearest(frames: FrameBboxEntry[], videoMs: number): FrameBboxEntry | null {
  if (frames.length === 0) return null
  let lo = 0, hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].ts < videoMs) lo = mid + 1
    else hi = mid
  }
  const a = frames[lo]
  const b = lo > 0 ? frames[lo - 1] : null
  const best = b && Math.abs(b.ts - videoMs) < Math.abs(a.ts - videoMs) ? b : a
  return Math.abs(best.ts - videoMs) < 2000 ? best : null
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface CourtMap2DProps {
  videoRef?: RefObject<HTMLVideoElement>
  frameData?: FrameBboxEntry[]
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function CourtMap2D({ videoRef, frameData }: CourtMap2DProps) {
  const storePlayers = useMatchStore(s => s.players)
  const storeBall    = useMatchStore(s => s.ball)

  // Track video currentTime for frame-sync mode
  const [videoMs, setVideoMs] = useState(0)
  const useFrameMode = !!(frameData && frameData.length > 0 && videoRef)

  useEffect(() => {
    if (!useFrameMode || !videoRef?.current) return
    const video = videoRef.current
    const onTime = () => setVideoMs(video.currentTime * 1000)
    video.addEventListener('timeupdate', onTime)
    return () => video.removeEventListener('timeupdate', onTime)
  }, [useFrameMode, videoRef])

  // Resolve players + ball from frameData or store
  const nearest = useFrameMode ? findNearest(frameData!, videoMs) : null

  type Dot = { trackId: number; jerseyNumber: number | null; team: string; courtPos: [number, number] }
  const players: Dot[] = nearest
    ? nearest.p
        .filter(p => p.cp != null)
        .map(p => ({ trackId: p.i, jerseyNumber: p.j, team: p.t, courtPos: p.cp as [number, number] }))
    : storePlayers.filter(p => p.courtPos != null).map(p => ({
        trackId: p.trackId,
        jerseyNumber: p.jerseyNumber,
        team: p.team,
        courtPos: p.courtPos as [number, number],
      }))

  const ballPos: [number, number] | null =
    nearest?.bl?.cp ?? storeBall?.courtPos ?? null

  const hoop_r = +(S * 0.225).toFixed(1)

  return (
    <div style={{ background: '#78C850', padding: 4, borderRadius: 4, lineHeight: 0 }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', width: '100%' }}>

        {/* Background */}
        <rect width={W} height={H} fill="#000" />

        {/* Paint fills */}
        <rect x={mx(0)}    y={my(9.95)} width={mx(5.8)}           height={my(5.05) - my(9.95)} fill="#141414" />
        <rect x={mx(22.2)} y={my(9.95)} width={mx(28) - mx(22.2)} height={my(5.05) - my(9.95)} fill="#141414" />

        {/* Court boundary */}
        <rect x={mx(0)} y={my(15)} width={mx(28)} height={my(0) - my(15)} fill="none" stroke="#fff" strokeWidth={1} />

        {/* Centre line */}
        <line x1={mx(14)} y1={my(15)} x2={mx(14)} y2={my(0)} stroke="#fff" strokeWidth={0.8} />

        {/* Centre circle */}
        <path d={PATH_CTR_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_CTR_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* Paint outlines */}
        <rect x={mx(0)}    y={my(9.95)} width={mx(5.8)}           height={my(5.05) - my(9.95)} fill="none" stroke="#fff" strokeWidth={0.8} />
        <rect x={mx(22.2)} y={my(9.95)} width={mx(28) - mx(22.2)} height={my(5.05) - my(9.95)} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* Hash marks */}
        {([0.9, 1.75, 2.6] as const).map(off => {
          const yT = 7.5 + (2.45 - off)
          const yB = 7.5 - (2.45 - off)
          return (
            <g key={off}>
              <line x1={mx(5.8)} y1={my(yT)} x2={mx(6.0)} y2={my(yT)} stroke="#fff" strokeWidth={0.5} />
              <line x1={mx(5.8)} y1={my(yB)} x2={mx(6.0)} y2={my(yB)} stroke="#fff" strokeWidth={0.5} />
              <line x1={mx(22.2)} y1={my(yT)} x2={mx(22.0)} y2={my(yT)} stroke="#fff" strokeWidth={0.5} />
              <line x1={mx(22.2)} y1={my(yB)} x2={mx(22.0)} y2={my(yB)} stroke="#fff" strokeWidth={0.5} />
            </g>
          )
        })}

        {/* FT circles */}
        <path d={PATH_FT_L_SOLID} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_FT_L_DASH}  fill="none" stroke="#fff" strokeWidth={0.8} strokeDasharray="3,2" />
        <path d={PATH_FT_R_SOLID} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_FT_R_DASH}  fill="none" stroke="#fff" strokeWidth={0.8} strokeDasharray="3,2" />

        {/* Restricted area arcs */}
        <path d={PATH_RA_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_RA_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* 3PT corner straights */}
        <line x1={mx(0)}     y1={my(0.9)}  x2={mx(2.99)} y2={my(0.9)}  stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(0)}     y1={my(14.1)} x2={mx(2.99)} y2={my(14.1)} stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(25.01)} y1={my(0.9)}  x2={mx(28)}   y2={my(0.9)}  stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(25.01)} y1={my(14.1)} x2={mx(28)}   y2={my(14.1)} stroke="#fff" strokeWidth={0.8} />

        {/* 3PT arcs */}
        <path d={PATH_3PT_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_3PT_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* Backboards */}
        <line x1={mx(1.2)} y1={my(7.5 - 0.915)} x2={mx(1.2)} y2={my(7.5 + 0.915)} stroke="#fff" strokeWidth={1.5} />
        <line x1={mx(26.8)} y1={my(7.5 - 0.915)} x2={mx(26.8)} y2={my(7.5 + 0.915)} stroke="#fff" strokeWidth={1.5} />

        {/* Hoops */}
        <circle cx={mx(1.575)} cy={my(7.5)} r={hoop_r} fill="none" stroke="#fff" strokeWidth={0.8} />
        <circle cx={mx(26.425)} cy={my(7.5)} r={hoop_r} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* Player dots */}
        {players.map(p => {
          const cx   = mx(p.courtPos[0])
          const cy   = my(p.courtPos[1])
          const r    = 3.5
          const fill = p.team === 'A' ? '#00BCD4' : p.team === 'B' ? '#F97316' : '#9CA3AF'
          const label = p.jerseyNumber != null ? `${p.jerseyNumber}` : `${p.trackId}`
          return (
            <g key={p.trackId}>
              <text
                x={cx} y={cy - r - 1.5}
                textAnchor="middle" dominantBaseline="auto"
                fontSize={7} fontWeight="bold"
                fill={fill} stroke="#000" strokeWidth={0.4} paintOrder="stroke"
              >{label}</text>
              <circle cx={cx} cy={cy} r={r} fill={fill} stroke="#fff" strokeWidth={0.5} />
            </g>
          )
        })}

        {/* Ball dot */}
        {ballPos && (
          <circle cx={mx(ballPos[0])} cy={my(ballPos[1])} r={2.5} fill="#FACC15" />
        )}

      </svg>
    </div>
  )
}
