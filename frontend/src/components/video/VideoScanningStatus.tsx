import { useState, useEffect, useRef } from 'react'
import { CheckCircle, Loader2, AlertTriangle, RefreshCw, Square } from 'lucide-react'
import axios, { AxiosError } from 'axios'

interface Props {
  matchId:      string
  onComplete:   () => void
  onReupload?:  () => void
}

type Phase = 'connecting' | 'scanning' | 'done' | 'dev_mode' | 'error'

// Tolerate a transient network drop (ERR_NETWORK_CHANGED, brief 5xx) before
// ever declaring the analysis failed — it keeps running server-side.
const MAX_POLL_RETRIES = 8      // ~20 s of reconnect attempts (resets on any success)
const POLL_RETRY_MS    = 2500

export default function VideoScanningStatus({ matchId, onComplete, onReupload }: Props) {
  const [phase, setPhase] = useState<Phase>('connecting')
  const [progress, setProgress] = useState(0)
  const [framesProcessed, setFramesProcessed] = useState(0)
  const [totalFrames, setTotalFrames] = useState(0)
  const [fps, setFps] = useState(0)
  const [scanLine, setScanLine] = useState(0)
  const [errorMsg, setErrorMsg] = useState('')
  const [isCancelling, setIsCancelling] = useState(false)
  const [reconnecting, setReconnecting] = useState(false)
  const retryRef = useRef(0)

  const handleStop = async () => {
    setIsCancelling(true)
    try {
      await axios.post(`${import.meta.env.VITE_API_URL}/api/upload/stop/${matchId}`)
    } catch {
      // Polling will pick up the cancelled status from backend
    }
  }

  // Animated scan line sweep
  useEffect(() => {
    const t = setInterval(() => {
      setScanLine((prev) => (prev >= 100 ? 0 : prev + 2))
    }, 50)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const res = await axios.get(
          `${import.meta.env.VITE_API_URL}/api/upload/progress/${matchId}`
        )
        if (cancelled) return

        // A poll succeeded → connection is healthy again.
        retryRef.current = 0
        if (reconnecting) setReconnecting(false)

        const { frames_processed, total_frames, fps_actual, status } = res.data
        setFramesProcessed(frames_processed)
        setTotalFrames(total_frames)
        setFps(fps_actual)

        // Frame processing = 0–95%; finalization (DB save, score patch) = 95–99%; done = 100%
        const framePct = total_frames > 0
          ? Math.min(95, Math.round((frames_processed / total_frames) * 95))
          : 0
        setProgress(framePct)

        if (status === 'done') {
          setProgress(100)
          setPhase('done')
        } else if (status === 'error') {
          setPhase('error')
          setErrorMsg('Pipeline analisis gagal. Coba upload ulang video.')
        } else if (status === 'cancelled') {
          setPhase('error')
          setErrorMsg('Analisis dibatalkan.')
        } else {
          setPhase('scanning')
          setTimeout(poll, 2000)
        }
      } catch (err) {
        if (cancelled) return

        const axiosErr = err as AxiosError
        const httpStatus = axiosErr.response?.status

        // 404 = the match genuinely isn't there (backend restarted / never ran).
        // That's not transient, so don't spin — report it.
        if (httpStatus === 404) {
          setPhase('error')
          setErrorMsg('Sesi analisis tidak ditemukan. Backend mungkin di-restart saat proses berjalan. Silakan upload ulang video.')
          return
        }

        // Everything else — a dropped/changed network (ERR_NETWORK_CHANGED),
        // timeout, 503, or a 5xx — is very likely TRANSIENT. The analysis runs
        // server-side and keeps going regardless of the browser, so retry a few
        // times ("Menyambung ulang…") before ever declaring failure. One failed
        // poll must never read as a permanently failed analysis.
        retryRef.current += 1
        if (retryRef.current <= MAX_POLL_RETRIES) {
          setReconnecting(true)
          setErrorMsg('')
          setTimeout(poll, POLL_RETRY_MS)
          return
        }

        // Only after several consecutive failures do we surface the real reason.
        setReconnecting(false)
        setPhase('error')
        if (httpStatus === 503) {
          setErrorMsg('Pipeline analisis tidak tersedia di server (503). Pastikan backend pipeline aktif, lalu upload ulang video.')
        } else if (!axiosErr.response) {
          setErrorMsg('Koneksi ke server terputus terus-menerus. Cek jaringan client — analisis kemungkinan MASIH berjalan di server; refresh setelah koneksi stabil.')
        } else {
          setErrorMsg(`Server error ${httpStatus ?? ''}: ${(axiosErr.response?.data as any)?.detail ?? 'Gagal mengambil status analisis.'}`)
        }
      }
    }

    poll()

    return () => {
      cancelled = true
    }
  }, [matchId])

  // Auto-advance to dashboard 3 s after analysis completes (handles reconnect too)
  const [countdown, setCountdown] = useState(3)
  useEffect(() => {
    if (phase !== 'done') return
    setCountdown(3)
    const tick = setInterval(() => setCountdown((c) => c - 1), 1000)
    const go   = setTimeout(() => onComplete(), 3000)
    return () => { clearInterval(tick); clearTimeout(go) }
  }, [phase, onComplete])

  const isDone = phase === 'done'
  const isError = phase === 'error'
  const isConnecting = phase === 'connecting'
  const isDevMode = phase === 'dev_mode'
  const isFinalizing = phase === 'scanning' && totalFrames > 0 && framesProcessed >= totalFrames

  const etaSeconds =
    fps > 0 && totalFrames > 0 && !isFinalizing
      ? Math.round((totalFrames - framesProcessed) / fps)
      : null
  const etaLabel =
    isFinalizing
      ? 'Finalisasi & menyimpan data...'
      : etaSeconds !== null
      ? etaSeconds > 60
        ? `~${Math.round(etaSeconds / 60)} mnt tersisa`
        : `~${etaSeconds} dtk tersisa`
      : null

  // ── Error state ─────────────────────────────────────────────────────────────
  if (isError) {
    return (
      <div className="bg-gray-900 rounded-lg overflow-hidden shadow-sm border border-red-500/40">
        <div className="flex items-start gap-3 px-4 py-5">
          <AlertTriangle size={20} className="text-red-400 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-bold text-red-400">Analisis Gagal</p>
            <p className="text-xs text-gray-400 mt-1 break-words">{errorMsg}</p>
          </div>
          <div className="flex flex-col gap-2 shrink-0">
            {onReupload && (
              <button
                onClick={onReupload}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-primary hover:bg-primary-dark text-white text-xs font-bold rounded-lg transition-smooth"
              >
                Upload Ulang
              </button>
            )}
            <button
              onClick={() => window.location.reload()}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs font-bold rounded-lg transition-smooth"
            >
              <RefreshCw size={12} />
              Refresh
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Normal / scanning state ──────────────────────────────────────────────────
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden shadow-sm">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-black/50">
        <div className="flex items-center gap-2">
          {isDone ? (
            <CheckCircle size={16} className="text-success" />
          ) : (
            <Loader2 size={16} className="text-primary animate-spin" />
          )}
          <span className="text-sm font-bold text-white">
            {isDone ? 'Analisis Selesai' : isFinalizing ? 'Menyimpan Hasil Analisis...' : 'AI Sedang Menganalisis Video...'}
          </span>
          {reconnecting && !isDone && (
            <span className="flex items-center gap-1.5 text-xs px-2 py-0.5 bg-amber-400/20 text-amber-300 rounded-full border border-amber-400/30">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              Menyambung ulang…
            </span>
          )}
          {isDevMode && (
            <span className="text-xs px-2 py-0.5 bg-yellow-400/20 text-yellow-300 rounded-full border border-yellow-400/30">
              dev mode
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {phase === 'scanning' && !isCancelling && (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-700/80 hover:bg-red-600 text-white text-xs font-bold rounded-lg transition-smooth"
            >
              <Square size={11} className="fill-white" />
              Hentikan
            </button>
          )}
          {isCancelling && (
            <span className="text-xs text-gray-400 font-mono">Menghentikan...</span>
          )}
          {isDone && (
            <button
              onClick={onComplete}
              className="px-4 py-1.5 bg-primary text-white text-sm font-bold rounded-lg hover:bg-primary-dark transition-smooth"
            >
              Lihat Dashboard ({countdown > 0 ? countdown : '→'})
            </button>
          )}
        </div>
      </div>

      {/* Scan animation panel */}
      <div className="relative h-24 bg-gray-950 overflow-hidden select-none">
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage:
              'linear-gradient(to right, #4ade80 1px, transparent 1px), linear-gradient(to bottom, #4ade80 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        {!isDone && (
          <div
            className="absolute top-0 bottom-0 w-px"
            style={{
              left: `${scanLine}%`,
              background: 'linear-gradient(to bottom, transparent, #3b82f6, transparent)',
              boxShadow: '0 0 10px 3px rgba(59,130,246,0.5)',
            }}
          />
        )}

        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 px-4">
          {isConnecting && (
            <span className="text-xs text-gray-400 font-mono tracking-wide">
              Menghubungkan ke pipeline...
            </span>
          )}

          {(phase === 'scanning' || isDevMode) && !isDone && (
            <>
              <span className="text-xs text-gray-300 font-mono tracking-wide">
                {isDevMode
                  ? `Simulasi YOLO + OCR scan — ${progress}% selesai`
                  : `Frame ${framesProcessed.toLocaleString()} / ${totalFrames.toLocaleString()} — ${fps.toFixed(1)} fps`}
              </span>
              {!isDevMode && etaLabel && (
                <span className="text-xs text-gray-500 font-mono">{etaLabel}</span>
              )}
            </>
          )}

          {isDone && (
            <span className="text-sm text-success font-bold font-mono tracking-wide">
              ✓ Scan selesai
              {totalFrames > 0 && ` — ${totalFrames.toLocaleString()} frame diproses`}
            </span>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div className="px-4 pt-3 pb-4 bg-gray-900">
        <div className="flex justify-between mb-1.5">
          <span className="text-xs text-gray-400 font-mono">
            {isConnecting
              ? 'Menghubungkan...'
              : isDone
              ? 'Selesai'
              : isDevMode
              ? 'Deteksi jersey + OCR (simulasi)'
              : isFinalizing
              ? 'Menyimpan statistik & event ke database...'
              : 'Deteksi YOLO · jersey OCR · tracking · statistik'}
          </span>
          <span className="text-xs font-bold font-mono text-primary">{progress}%</span>
        </div>

        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>

        {!isDone && (
          <p className="text-xs text-gray-600 mt-2">
            {isDevMode
              ? 'Backend pipeline tidak tersedia — simulasi aktif untuk pengujian'
              : 'Bbox pemain akan muncul satu per satu setelah jersey dikonfirmasi oleh OCR'}
          </p>
        )}
      </div>
    </div>
  )
}
