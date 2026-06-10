import { useEffect, useRef } from 'react'
import { useMatchStore } from '../../store/matchStore'
import { Player, FrameBboxEntry } from '../../types'

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
  // lo = first index with ts >= videoMs; check lo-1 too
  const a = frameData[lo]
  const b = lo > 0 ? frameData[lo - 1] : null
  const best = b && Math.abs(b.ts - videoMs) < Math.abs(a.ts - videoMs) ? b : a
  // Discard if more than 2 s away (e.g. video seeked way past stored data)
  return Math.abs(best.ts - videoMs) < 2000 ? best : null
}

// ── Single bbox + label drawing helper ───────────────────────────────────────
function drawBox(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number,
  color: string,
  label: string,
) {
  if (w < 4 || h < 4) return
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(x, y, w, h)

  ctx.font = 'bold 12px Inter, sans-serif'
  const textW = ctx.measureText(label).width + 10
  const labelH = 18
  ctx.fillStyle = color
  ctx.fillRect(x - 1, y - labelH, textW, labelH)
  ctx.fillStyle = '#ffffff'
  ctx.fillText(label, x + 4, y - 5)
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function CourtOverlay({ videoRef, frameData }: CourtOverlayProps) {
  const canvasRef      = useRef<HTMLCanvasElement>(null)
  const players        = useMatchStore((s) => s.players)
  const playersVideoTs = useMatchStore((s) => s.playersVideoTs)

  useEffect(() => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas || !video) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animId: number

    const render = () => {
      // Keep canvas size in sync with displayed video dimensions
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width  = video.clientWidth
        canvas.height = video.clientHeight
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const W = canvas.width
      const H = canvas.height
      const videoMs = video.currentTime * 1000

      if (frameData && frameData.length > 0) {
        // ── Frame-accurate mode ──────────────────────────────────────────────
        // Binary-search for the stored frame whose timestamp is closest to the
        // current video position.  Works correctly after scrubbing or replay.
        const entry = findNearest(frameData, videoMs)
        if (entry) {
          for (const p of entry.p) {
            const [x1, y1, x2, y2] = p.b
            drawBox(
              ctx,
              x1 * W, y1 * H, (x2 - x1) * W, (y2 - y1) * H,
              p.t === 'A' ? TEAM_A_COLOR : p.t === 'B' ? TEAM_B_COLOR : UNCONFIRMED,
              p.j != null ? `#${p.j}` : `ID:${p.i}`,
            )
          }
          if (entry.bl) {
            const [bx1, by1, bx2, by2] = entry.bl.b
            drawBox(ctx, bx1 * W, by1 * H, (bx2 - bx1) * W, (by2 - by1) * H,
              BALL_COLOR, 'BALL')
          }
        }
      } else {
        // ── WS / live fallback mode ──────────────────────────────────────────
        // Uses the last matchStore update.  Bboxes are hidden when the video
        // has advanced more than STALE_THRESHOLD ms past the last WS broadcast.
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
            const label = player.jerseyNumber != null
              ? `#${player.jerseyNumber} ${player.name.split(' ')[0]}`
              : `ID:${player.trackId}`
            drawBox(ctx, x1 * W, y1 * H, w, h, color, label)
          })
        }
      }

      animId = requestAnimationFrame(render)
    }

    render()
    return () => cancelAnimationFrame(animId)
  }, [players, playersVideoTs, frameData, videoRef])

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
}
