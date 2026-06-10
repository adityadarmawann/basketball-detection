import { useEffect, useRef } from 'react'
import { useMatchStore } from '../../store/matchStore'
import { Player, FrameBboxEntry, RosterPlayer } from '../../types'

interface CourtOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>
  /** If provided, uses binary-search time lookup for frame-accurate bboxes.
   *  Falls back to matchStore (WS) when absent or empty. */
  frameData?: FrameBboxEntry[]
}

const TEAM_A_COLOR    = '#00BCD4'
const TEAM_B_COLOR    = '#F97316'
const UNCONFIRMED     = '#9CA3AF'
const BALL_COLOR      = '#FACC15'
const STALE_THRESHOLD = 2500   // ms — only used in WS fallback mode

// ── Binary search: find entry whose ts is nearest to videoMs ─────────────────
function findNearest(frameData: FrameBboxEntry[], videoMs: number): FrameBboxEntry | null {
  if (frameData.length === 0) return null
  let lo = 0, hi = frameData.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frameData[mid].ts < videoMs) lo = mid + 1
    else hi = mid
  }
  const a = frameData[lo]
  const b = lo > 0 ? frameData[lo - 1] : null
  const best = b && Math.abs(b.ts - videoMs) < Math.abs(a.ts - videoMs) ? b : a
  return Math.abs(best.ts - videoMs) < 2000 ? best : null
}

// ── Player bbox with 4-panel layout ──────────────────────────────────────────
//
//  ┌─────────────────┐   ← top chip: "#7 | Budi" or "ID:3"
//  │                 │
//  │   bbox rect     │   ← colored border box
//  │                 │
//  └─────────────────┘
//  3.2 km/h             ← speed (dark bg, gray text) — hidden when 0
//  DRIBBLE              ← action (team color bg, white text) — hidden when STAND/empty
//
function drawPlayerBox(
  ctx:         CanvasRenderingContext2D,
  x:           number,
  y:           number,
  w:           number,
  h:           number,
  color:       string,
  topLabel:    string,
  speedLabel:  string,
  actionLabel: string,
) {
  if (w < 4 || h < 4) return

  const CHIP_H = 16
  const LINE_H = 14

  // ── Top chip ─────────────────────────────────────────────────────
  ctx.font = 'bold 11px Inter, sans-serif'
  const topW = ctx.measureText(topLabel).width + 8
  ctx.fillStyle = color
  ctx.fillRect(x - 1, y - CHIP_H, topW, CHIP_H)
  ctx.fillStyle = '#fff'
  ctx.fillText(topLabel, x + 3, y - 4)

  // ── Bbox rectangle ───────────────────────────────────────────────
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(x, y, w, h)

  // ── Bottom rows ──────────────────────────────────────────────────
  let footerY = y + h + LINE_H

  if (speedLabel) {
    ctx.font = '10px Inter, sans-serif'
    const sw = ctx.measureText(speedLabel).width + 6
    ctx.fillStyle = 'rgba(0,0,0,0.65)'
    ctx.fillRect(x - 1, footerY - LINE_H + 2, sw, LINE_H)
    ctx.fillStyle = '#ccc'
    ctx.fillText(speedLabel, x + 2, footerY - 2)
    footerY += LINE_H
  }

  if (actionLabel) {
    ctx.font = 'bold 10px Inter, sans-serif'
    const aw = ctx.measureText(actionLabel).width + 6
    ctx.fillStyle = color
    ctx.fillRect(x - 1, footerY - LINE_H + 2, aw, LINE_H)
    ctx.fillStyle = '#fff'
    ctx.fillText(actionLabel, x + 2, footerY - 2)
  }
}

// ── Ball bbox ─────────────────────────────────────────────────────────────────
function drawBallBox(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
) {
  if (w < 4 || h < 4) return
  const CHIP_H = 14
  ctx.font = 'bold 10px Inter, sans-serif'
  const lw = ctx.measureText('BALL').width + 6
  ctx.fillStyle = BALL_COLOR
  ctx.fillRect(x - 1, y - CHIP_H, lw, CHIP_H)
  ctx.fillStyle = '#000'
  ctx.fillText('BALL', x + 2, y - 3)
  ctx.strokeStyle = BALL_COLOR
  ctx.lineWidth = 2
  ctx.strokeRect(x, y, w, h)
}

// ── Roster jersey→name lookup helper ─────────────────────────────────────────
function rosterName(roster: RosterPlayer[], jerseyNumber: number | null): string | null {
  if (jerseyNumber == null) return null
  const entry = roster.find(r => r.jerseyNumber === jerseyNumber)
  return entry ? entry.name.split(' ')[0] : null
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function CourtOverlay({ videoRef, frameData }: CourtOverlayProps) {
  const canvasRef      = useRef<HTMLCanvasElement>(null)
  const players        = useMatchStore((s) => s.players)
  const playersVideoTs = useMatchStore((s) => s.playersVideoTs)
  const roster         = useMatchStore((s) => s.roster)

  useEffect(() => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animId: number

    const render = () => {
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width  = video.clientWidth
        canvas.height = video.clientHeight
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const W = canvas.width
      const H = canvas.height
      const videoMs = video.currentTime * 1000

      if (frameData && frameData.length > 0) {
        // ── Frame-accurate mode (post-analysis replay) ───────────────────────
        const entry = findNearest(frameData, videoMs)
        if (entry) {
          for (const p of entry.p) {
            const [x1, y1, x2, y2] = p.b
            const color = p.t === 'A' ? TEAM_A_COLOR : p.t === 'B' ? TEAM_B_COLOR : UNCONFIRMED

            // Top label: "#7 | Budi" when roster match found, else "#7" or "ID:3"
            const name = rosterName(roster, p.j)
            const topLabel = p.j != null
              ? (name ? `#${p.j} | ${name}` : `#${p.j}`)
              : `ID:${p.i}`

            const speedLabel  = p.s > 0 ? `${p.s.toFixed(1)} km/h` : ''
            const actionLabel = p.a && p.a !== 'STAND' ? p.a : ''

            drawPlayerBox(
              ctx,
              x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H,
              color, topLabel, speedLabel, actionLabel,
            )
          }
          if (entry.bl) {
            const [bx1, by1, bx2, by2] = entry.bl.b
            drawBallBox(ctx, bx1 * W, by1 * H, (bx2 - bx1) * W, (by2 - by1) * H)
          }
        }
      } else {
        // ── WS / live fallback mode ──────────────────────────────────────────
        const isStale = playersVideoTs > 0 && (videoMs - playersVideoTs) > STALE_THRESHOLD
        if (!isStale) {
          players.forEach((player: Player) => {
            const [x1, y1, x2, y2] = player.bbox
            if (x1 === 0 && x2 === 0) return
            const w = (x2 - x1) * W
            const h = (y2 - y1) * H
            if (w < 4 || h < 4) return

            const color = player.team === 'A' ? TEAM_A_COLOR
                        : player.team === 'B' ? TEAM_B_COLOR
                        : UNCONFIRMED

            const topLabel = player.jerseyNumber != null
              ? `#${player.jerseyNumber} | ${player.name.split(' ')[0]}`
              : `ID:${player.trackId}`

            const speedLabel  = player.speedKmh > 0 ? `${player.speedKmh.toFixed(1)} km/h` : ''
            const actionLabel = player.action && player.action !== 'STAND' ? player.action : ''

            drawPlayerBox(ctx, x1 * W, y1 * H, w, h, color, topLabel, speedLabel, actionLabel)
          })
        }
      }

      animId = requestAnimationFrame(render)
    }

    render()
    return () => cancelAnimationFrame(animId)
  }, [players, playersVideoTs, frameData, roster, videoRef])

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
}
