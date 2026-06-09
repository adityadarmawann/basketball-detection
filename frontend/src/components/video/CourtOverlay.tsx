import { useEffect, useRef } from 'react'
import { useMatchStore } from '../../store/matchStore'
import { Player } from '../../types'

interface CourtOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>
}

const TEAM_A_COLOR = '#00BCD4' // cyan
const TEAM_B_COLOR = '#F97316' // orange — kontras di background gelap

export default function CourtOverlay({ videoRef }: CourtOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const players = useMatchStore((s) => s.players)

  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationFrameId: number

    const render = () => {
      // Match canvas size to video element's rendered size
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth
        canvas.height = video.clientHeight
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      players.forEach((player: Player) => {
        drawPlayer(ctx, player, canvas.width, canvas.height)
      })

      animationFrameId = window.requestAnimationFrame(render)
    }

    render()

    return () => {
      window.cancelAnimationFrame(animationFrameId)
    }
  }, [players, videoRef])

  const drawPlayer = (
    ctx: CanvasRenderingContext2D,
    player: Player,
    canvasWidth: number,
    canvasHeight: number
  ) => {
    const [x1, y1, x2, y2] = player.bbox
    if (x1 === 0 && x2 === 0) return  // degenerate bbox — skip

    const absX  = x1 * canvasWidth
    const absY  = y1 * canvasHeight
    const width = (x2 - x1) * canvasWidth
    const height = (y2 - y1) * canvasHeight
    if (width < 4 || height < 4) return  // too small to draw

    // Team color: fallback gray when team unknown (jersey not yet confirmed)
    const color = player.team === 'A' ? TEAM_A_COLOR
                : player.team === 'B' ? TEAM_B_COLOR
                : '#9CA3AF'  // gray — tracking but jersey not confirmed yet

    // Bounding box
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.strokeRect(absX, absY, width, height)

    // Label: show jersey number if confirmed, otherwise just show track ID
    const label = player.jerseyNumber != null
      ? `#${player.jerseyNumber} ${player.name.split(' ')[0]}`
      : `ID:${player.trackId}`

    ctx.font = 'bold 12px "Inter", sans-serif'
    const textMetrics = ctx.measureText(label)
    const labelWidth = textMetrics.width + 12
    const labelHeight = 18

    ctx.fillStyle = color
    ctx.fillRect(absX - 1, absY - labelHeight, labelWidth, labelHeight)
    ctx.fillStyle = 'white'
    ctx.fillText(label, absX + 5, absY - 5)

    // Action & speed below bbox
    const speedLabel = `${player.speedKmh.toFixed(1)} km/h`
    ctx.font = '10px "Inter", sans-serif'

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
    ctx.fillRect(absX, absY + height, width, 14)
    ctx.fillStyle = 'white'
    ctx.fillText(player.action, absX + 4, absY + height + 10)

    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)'
    ctx.fillRect(absX, absY + height + 14, width, 14)
    ctx.fillStyle = '#F59E0B'
    ctx.fillText(speedLabel, absX + 4, absY + height + 24)
  }

  return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
}