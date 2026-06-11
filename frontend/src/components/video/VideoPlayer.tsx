import { useRef, useState, useEffect } from 'react'
import { Play, Pause, Volume2, VolumeX } from 'lucide-react'
import CourtMap2D from './CourtMap2D'
import CourtOverlay from './CourtOverlay'
import { FrameBboxEntry } from '../../types'

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
  const [volume, setVolume] = useState(1)
  const [progress, setProgress] = useState(0)
  const [duration, setDuration] = useState(0)
  const [fps, setFps] = useState(0)
  const [gpuUsage, setGpuUsage] = useState(0)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleDurationChange = () => setDuration(video.duration)
    const handleTimeUpdate = () => {
      setProgress(video.currentTime)
      onTimeUpdate?.(video.currentTime, video.duration || 0)
      // Simulate FPS and GPU usage for demo
      setFps(Math.floor(Math.random() * 10 + 20))
      setGpuUsage(Math.floor(Math.random() * 40 + 50))
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
          <div className="badge-live">● LIVE</div>
          <div className="px-2 py-1 bg-black/50 rounded">{fps} fps</div>
          <div className="px-2 py-1 bg-black/50 rounded">GPU {gpuUsage}%</div>
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
        {/* Progress Bar */}
        <div className="space-y-1">
          <input
            type="range"
            min="0"
            max={duration || 0}
            value={progress}
            onChange={(e) => {
              if (videoRef.current) {
                videoRef.current.currentTime = parseFloat(e.target.value)
              }
            }}
            className="w-full cursor-pointer"
          />
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

          {/* Volume Control */}
          <div className="flex items-center gap-2 flex-1 max-w-xs">
            {volume === 0 ? (
              <VolumeX size={20} />
            ) : (
              <Volume2 size={20} />
            )}
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => {
                const val = parseFloat(e.target.value)
                setVolume(val)
                if (videoRef.current) {
                  videoRef.current.volume = val
                }
              }}
              className="flex-1 cursor-pointer"
            />
          </div>

          <div className="text-xs text-gray-400">
            {formatTime(duration)}
          </div>
        </div>
      </div>
    </div>
  )
}
