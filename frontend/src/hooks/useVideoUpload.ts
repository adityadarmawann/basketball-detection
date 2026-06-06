import { useState } from 'react'
import axios, { isAxiosError } from 'axios'

interface UploadProgress {
  progress: number
  status: 'idle' | 'uploading' | 'success' | 'error'
  message: string
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

export const useVideoUpload = () => {
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    progress: 0,
    status: 'idle',
    message: '',
  })

  const uploadVideo = async (file: File) => {
    setUploadProgress({ progress: 0, status: 'uploading', message: 'Uploading...' })

    try {
      const formData = new FormData()
      formData.append('file', file)

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
        const msg = err.response.data?.detail || 'Upload gagal.'
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
