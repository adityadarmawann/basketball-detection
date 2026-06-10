import { useMatchStore } from '../../store/matchStore'

// ── Court constants (metres — matches court.py) ───────────────────────────────
const CW = 28.0   // court x  (left baseline → right baseline)
const CH = 15.0   // court y  (bottom sideline → top sideline)

// ── SVG viewport ─────────────────────────────────────────────────────────────
const W = 280                         // px wide  (scales down via CSS)
const H = Math.round(W * CH / CW)     // px tall  (= 150) — exact 28:15 ratio
const S = W / CW                      // px / metre  (≈ 10) — same for both axes

// court-metre → SVG-px  (SVG y-axis is flipped vs court y-axis)
const mx = (m: number) => +(m * S).toFixed(2)
const my = (m: number) => +((CH - m) * S).toFixed(2)

// ── Arc helpers (same sweep logic as draw_court.py arc_pts) ──────────────────
function arcPath(
  hx: number, hy: number, r: number,
  degFrom: number, degTo: number,
): string {
  const pts: string[] = []
  for (let d = degFrom; d <= degTo; d++) {
    const rad = (d * Math.PI) / 180
    pts.push(`${mx(hx + r * Math.cos(rad))},${my(hy + r * Math.sin(rad))}`)
  }
  if (!pts.length) return ''
  return `M ${pts[0]} ${pts.slice(1).map(p => `L ${p}`).join(' ')}`
}

// 3PT arc — only the portion in front of the basket, clipped to y ∈ [0.9, 14.1]
function arc3PT(hx: number, hy: number, faceRight: boolean): string {
  const pts: string[] = []
  for (let d = 0; d <= 360; d++) {
    const rad = (d * Math.PI) / 180
    const x   = hx + 6.75 * Math.cos(rad)
    const y   = hy + 6.75 * Math.sin(rad)
    if (y < 0.9 || y > 14.1) continue
    if (faceRight ? x <= hx : x >= hx) continue
    pts.push(`${mx(x)},${my(y)}`)
  }
  if (!pts.length) return ''
  return `M ${pts[0]} ${pts.slice(1).map(p => `L ${p}`).join(' ')}`
}

// ── Static paths — computed once at module load, never re-computed ────────────

// Center circle (two halves so each side "belongs" to its half-court)
const PATH_CTR_L = arcPath(14.0, 7.5, 1.8,  90, 270)
const PATH_CTR_R = arcPath(14.0, 7.5, 1.8, -90,  90)

// Free throw circles:  solid half = outside paint (facing centre),  dashed half = inside paint
const PATH_FT_L_SOLID = arcPath( 5.8,  7.5, 1.8, -90,  90)
const PATH_FT_L_DASH  = arcPath( 5.8,  7.5, 1.8,  90, 270)
const PATH_FT_R_SOLID = arcPath(22.2,  7.5, 1.8,  90, 270)
const PATH_FT_R_DASH  = arcPath(22.2,  7.5, 1.8, -90,  90)

// Restricted area arcs (facing centre)
const PATH_RA_L = arcPath( 1.575, 7.5, 1.25, -90,  90)
const PATH_RA_R = arcPath(26.425, 7.5, 1.25,  90, 270)

// 3PT arcs
const PATH_3PT_L = arc3PT( 1.575, 7.5,  true)
const PATH_3PT_R = arc3PT(26.425, 7.5, false)

// ── Component ─────────────────────────────────────────────────────────────────
export default function CourtMap2D() {
  const players = useMatchStore(s => s.players)
  const ball    = useMatchStore(s => s.ball)

  const hoop_r = +(S * 0.225).toFixed(1)   // hoop ring radius in SVG px

  return (
    // Outer div = mint/tosca border band (matching draw_court.py C_BORDER)
    <div style={{
      background: '#78C850',   // BGR(80,200,120) → RGB(120,200,80)
      padding: 4,
      borderRadius: 4,
      lineHeight: 0,
    }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', width: '100%' }}>

        {/* ── Background ─────────────────────────────────────────────────── */}
        <rect width={W} height={H} fill="#000" />

        {/* ── Paint fills (draw_court.py C_PAINT = very dark) ───────────── */}
        {/* Left outer paint: x 0–5.8, y 3.65–11.35 */}
        <rect
          x={mx(0)}    y={my(11.35)}
          width={mx(5.8)}
          height={my(3.65) - my(11.35)}
          fill="#141414"
        />
        {/* Right outer paint: x 22.2–28, y 3.65–11.35 */}
        <rect
          x={mx(22.2)} y={my(11.35)}
          width={mx(28) - mx(22.2)}
          height={my(3.65) - my(11.35)}
          fill="#141414"
        />

        {/* ── Court boundary ─────────────────────────────────────────────── */}
        <rect
          x={mx(0)} y={my(15)}
          width={mx(28)} height={my(0) - my(15)}
          fill="none" stroke="#fff" strokeWidth={1}
        />

        {/* ── Centre line ────────────────────────────────────────────────── */}
        <line
          x1={mx(14)} y1={my(15)}
          x2={mx(14)} y2={my(0)}
          stroke="#fff" strokeWidth={0.8}
        />

        {/* ── Centre circle ──────────────────────────────────────────────── */}
        <path d={PATH_CTR_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_CTR_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* ── Outer paint rectangles (3 sides each; baseline = boundary) ── */}
        <polyline
          points={`${mx(0)},${my(11.35)} ${mx(5.8)},${my(11.35)} ${mx(5.8)},${my(3.65)} ${mx(0)},${my(3.65)}`}
          fill="none" stroke="#fff" strokeWidth={0.8}
        />
        <polyline
          points={`${mx(28)},${my(11.35)} ${mx(22.2)},${my(11.35)} ${mx(22.2)},${my(3.65)} ${mx(28)},${my(3.65)}`}
          fill="none" stroke="#fff" strokeWidth={0.8}
        />

        {/* ── Inner paint rectangles ─────────────────────────────────────── */}
        <polyline
          points={`${mx(0)},${my(9.3)} ${mx(5.8)},${my(9.3)} ${mx(5.8)},${my(5.7)} ${mx(0)},${my(5.7)}`}
          fill="none" stroke="#fff" strokeWidth={0.4}
        />
        <polyline
          points={`${mx(28)},${my(9.3)} ${mx(22.2)},${my(9.3)} ${mx(22.2)},${my(5.7)} ${mx(28)},${my(5.7)}`}
          fill="none" stroke="#fff" strokeWidth={0.4}
        />

        {/* ── Free throw circles (solid outside, dashed inside paint) ────── */}
        <path d={PATH_FT_L_SOLID} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_FT_L_DASH}  fill="none" stroke="#fff" strokeWidth={0.8} strokeDasharray="3,2" />
        <path d={PATH_FT_R_SOLID} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_FT_R_DASH}  fill="none" stroke="#fff" strokeWidth={0.8} strokeDasharray="3,2" />

        {/* ── Restricted area arcs ───────────────────────────────────────── */}
        <path d={PATH_RA_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_RA_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* ── 3PT corner straights ───────────────────────────────────────── */}
        <line x1={mx(0)}     y1={my(0.9)}  x2={mx(2.99)} y2={my(0.9)}  stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(0)}     y1={my(14.1)} x2={mx(2.99)} y2={my(14.1)} stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(25.01)} y1={my(0.9)}  x2={mx(28)}   y2={my(0.9)}  stroke="#fff" strokeWidth={0.8} />
        <line x1={mx(25.01)} y1={my(14.1)} x2={mx(28)}   y2={my(14.1)} stroke="#fff" strokeWidth={0.8} />

        {/* ── 3PT arcs ───────────────────────────────────────────────────── */}
        <path d={PATH_3PT_L} fill="none" stroke="#fff" strokeWidth={0.8} />
        <path d={PATH_3PT_R} fill="none" stroke="#fff" strokeWidth={0.8} />

        {/* ── Backboards ─────────────────────────────────────────────────── */}
        <line
          x1={mx(1.2)} y1={my(7.5 - 0.915)}
          x2={mx(1.2)} y2={my(7.5 + 0.915)}
          stroke="#fff" strokeWidth={1.5}
        />
        <line
          x1={mx(26.8)} y1={my(7.5 - 0.915)}
          x2={mx(26.8)} y2={my(7.5 + 0.915)}
          stroke="#fff" strokeWidth={1.5}
        />

        {/* ── Hoops (ring, not filled dot) ───────────────────────────────── */}
        <circle
          cx={mx(1.575)} cy={my(7.5)} r={hoop_r}
          fill="none" stroke="#fff" strokeWidth={0.8}
        />
        <circle
          cx={mx(26.425)} cy={my(7.5)} r={hoop_r}
          fill="none" stroke="#fff" strokeWidth={0.8}
        />

        {/* ── Player dots ────────────────────────────────────────────────── */}
        {players
          .filter(p => p.courtPos != null)
          .map(p => (
            <circle
              key={p.trackId}
              cx={mx(p.courtPos![0])}
              cy={my(p.courtPos![1])}
              r={3.5}
              fill={
                p.team === 'A' ? '#00BCD4'
              : p.team === 'B' ? '#F97316'
              : '#9CA3AF'
              }
              stroke="#fff"
              strokeWidth={0.5}
            />
          ))
        }

        {/* ── Ball dot ───────────────────────────────────────────────────── */}
        {ball?.courtPos && (
          <circle
            cx={mx(ball.courtPos[0])}
            cy={my(ball.courtPos[1])}
            r={2.5}
            fill="#FACC15"
          />
        )}

      </svg>
    </div>
  )
}
