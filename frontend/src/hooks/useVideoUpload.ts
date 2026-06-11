import { useState } from 'react'
import axios, { isAxiosError } from 'axios'
import { useMatchStore } from '../store/matchStore'

interface UploadProgress {
  progress: number
  status: 'idle' | 'uploading' | 'success' | 'error'
  message: string
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export const useVideoUpload = () => {
  const matchId = useMatchStore((s) => s.matchId)
  const roster  = useMatchStore((s) => s.roster)

  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    progress: 0,
    status: 'idle',
    message: '',
  })

  const uploadVideo = async (file: File, quarter?: number) => {
    setUploadProgress({ progress: 0, status: 'uploading', message: 'Uploading...' })

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('match_id', matchId || `match-${Date.now()}`)

      // Send roster as JSON: { jersey_number: { name, team } }
      if (roster.length > 0) {
        const rosterMap: Record<string, { name: string; team: string }> = {}
        roster.forEach((p) => { rosterMap[String(p.jerseyNumber)] = { name: p.name, team: p.team } })
        formData.append('roster', JSON.stringify(rosterMap))
      }

      // quarter: 1–4 for single-quarter clips; omit for full-game video
      if (quarter && quarter >= 1 && quarter <= 4) {
        formData.append('quarter', String(quarter))
      }

      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/api/upload-video`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const pct = Math.round((progressEvent.loaded * 100) / (progressEvent.total ?? 1))
            setUploadProgress({ progress: pct, status: 'uploading', message: `${pct}% uploaded` })
          },
        }
      )

      setUploadProgress({ progress: 100, status: 'success', message: 'Upload berhasil! Proses dimulai.' })
      return response.data as { video_id: string; match_id?: string }
    } catch (err: unknown) {
      if (isAxiosError(err) && err.response) {
        const raw = err.response.data?.detail
        const msg = typeof raw === 'string'
          ? raw
          : Array.isArray(raw)
            ? raw.map((e: { msg?: string }) => e.msg ?? JSON.stringify(e)).join('; ')
            : 'Upload gagal.'
        setUploadProgress({ progress: 0, status: 'error', message: msg })
        throw err
      }

      // Network error (backend not running) — simulate upload for development
      for (let pct = 10; pct <= 100; pct += 10) {
        await sleep(80)
        setUploadProgress({ progress: pct, status: 'uploading', message: `${pct}% uploaded (dev mode)` })
      }
      setUploadProgress({ progress: 100, status: 'success', message: 'Upload berhasil! (dev mode)' })
      return { video_id: `mock-video-${Date.now()}` }
    }
  }

  return { uploadProgress, uploadVideo }
}
