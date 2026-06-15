import { useRef, useState, useEffect } from 'react'
import { Play, Pause } from 'lucide-react'
import CourtMap2D from './CourtMap2D'
import CourtOverlay from './CourtOverlay'
import { FrameBboxEntry } from '../../types'
import { useMatchStore } from '../../store/matchStore'

interface VideoPlayerProps {
  videoUrl: string
  onFrameUpdate?: (frameData: unknown) => void
  showCourtMap?: boolean
  /** Per-frame bbox data for frame-accurate overlay. Fetched after analysis. */
  frameData?: FrameBboxEntry[]
  /** Called on every timeupdate — lets parent sync game clock to video position. */
  onTimeUpdate?: (currentTime: number, duration: number) => void
}

export default function VideoPlayer({ videoUrl, showCourtMap, frameData, onTimeUpdate }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const pipelineFps = useMatchStore((s) => s.pipelineFps)
  const gpuMetrics  = useMatchStore((s) => s.gpuMetrics)
  const isLive      = useMatchStore((s) => s.isLive)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleDurationChange = () => setDuration(video.duration)
    const handleTimeUpdate = () => {
      setProgress(video.currentTime)
      onTimeUpdate?.(video.currentTime, video.duration || 0)
    }

    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('durationchange', handleDurationChange)
    video.addEventListener('timeupdate', handleTimeUpdate)

    return () => {
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('durationchange', handleDurationChange)
      video.removeEventListener('timeupdate', handleTimeUpdate)
    }
  }, [])

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
    }
  }

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return '00:00'
    const hrs = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    return [hrs, mins, secs].map((x) => String(x).padStart(2, '0')).join(':')
  }

  return (
    <div className="bg-surface rounded-lg shadow-sm overflow-hidden mb-6">
      <div className="relative bg-black aspect-video flex items-center justify-center group">
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          className="w-full h-full object-contain"
        ></video>

        {/* Overlay Controls */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
          <button
            onClick={togglePlay}
            className="p-4 bg-primary rounded-full hover:bg-primary-dark transition-all"
          >
            {isPlaying ? (
              <Pause size={32} className="text-white" />
            ) : (
              <Play size={32} className="text-white fill-white" />
            )}
          </button>
        </div>

        {/* Status Badges */}
        <div className="absolute top-4 left-4 flex gap-2 text-white text-xs font-bold z-10">
          {isLive
            ? <div className="badge-live">● LIVE</div>
            : <div className="px-2 py-1 bg-gray-700/80 rounded">● REPLAY</div>
          }
          {pipelineFps > 0 && (
            <div className="px-2 py-1 bg-black/50 rounded">{pipelineFps} fps</div>
          )}
          {gpuMetrics.gpu > 0 && (
            <div className="px-2 py-1 bg-black/50 rounded">
              GPU {gpuMetrics.gpu}%
              {gpuMetrics.vramTotal > 0 && (
                <span className="ml-1 opacity-75">
                  {(gpuMetrics.vramUsed / 1024).toFixed(1)}/{(gpuMetrics.vramTotal / 1024).toFixed(1)}GB
                </span>
              )}
            </div>
          )}
        </div>

        {/* Canvas Overlay Layer - Renders player bounding boxes */}
        <CourtOverlay videoRef={videoRef} frameData={frameData} />

        {showCourtMap && (
          <div className="absolute top-4 right-4 w-[140px] md:w-[180px] lg:w-[220px] z-10">
            <CourtMap2D videoRef={videoRef} frameData={frameData} />
          </div>
        )}
      </div>

      {/* Controls Bar */}
      <div className="bg-gray-900 text-white p-4 space-y-3">
        {/* Progress Bar — read-only, no seeking */}
        <div className="space-y-1">
          <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-none"
              style={{ width: duration > 0 ? `${(progress / duration) * 100}%` : '0%' }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-400">
            <span>{formatTime(progress)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center gap-4">
          <button
            onClick={togglePlay}
            className="p-2 hover:bg-gray-700 rounded transition-smooth"
          >
            {isPlaying ? (
              <Pause size={20} />
            ) : (
              <Play size={20} className="fill-white" />
            )}
          </button>

          <div className="text-xs text-gray-400">
            {formatTime(duration)}
          </div>
        </div>
      </div>
    </div>
  )
}
