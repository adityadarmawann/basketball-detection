import { useState, useRef } from 'react'
import { Upload, AlertCircle } from 'lucide-react'
import { useVideoUpload } from '../../hooks/useVideoUpload'

interface VideoUploadProps {
  onUploadComplete:  (result: { videoId: string; videoUrl: string }) => void
  initialQuarter?:   number   // 1–4: pre-select a quarter clip (e.g. when continuing from live view)
}

export default function VideoUpload({ onUploadComplete, initialQuarter }: VideoUploadProps) {
  const [dragActive, setDragActive]     = useState(false)
  const [videoMode, setVideoMode]       = useState<'full' | 'clip'>(initialQuarter ? 'clip' : 'full')
  const [clipQuarter, setClipQuarter]   = useState<1 | 2 | 3 | 4>(
    (Math.min(Math.max(initialQuarter ?? 1, 1), 4)) as 1 | 2 | 3 | 4
  )
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { uploadProgress, uploadVideo } = useVideoUpload()

  const allowedFormats = ['.mp4', '.mov', '.avi', '.mkv']

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFile = async (file: File) => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()

    if (!allowedFormats.includes(ext)) {
      alert(`Format tidak didukung. Gunakan: ${allowedFormats.join(', ')}`)
      return
    }

    try {
      const quarter = videoMode === 'clip' ? clipQuarter : undefined
      const result  = await uploadVideo(file, quarter)
      const videoUrl = URL.createObjectURL(file)
      onUploadComplete({ videoId: result.video_id, videoUrl })
    } catch (error) {
      console.error('Upload error:', error)
    }
  }

  const isUploading = uploadProgress.status === 'uploading'
  const isSuccess = uploadProgress.status === 'success'
  const isError = uploadProgress.status === 'error'

  return (
    <div className="bg-surface rounded-lg p-8 shadow-sm">
      <h2 className="font-display text-2xl font-bold mb-6">Upload Video Pertandingan</h2>

      {/* Quarter / video-type selector */}
      {!isSuccess && (
        <div className="mb-6">
          <p className="text-sm font-bold text-text-secondary uppercase tracking-wide mb-3">
            Jenis Video
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setVideoMode('full')}
              className={`flex-1 py-2.5 px-4 rounded-lg border-2 text-sm font-bold transition-all ${
                videoMode === 'full'
                  ? 'border-primary bg-primary/5 text-primary'
                  : 'border-gray-200 text-text-secondary hover:border-gray-300'
              }`}
            >
              Full Game
              <span className="block text-xs font-normal opacity-70 mt-0.5">
                Auto-deteksi Q1–Q4 dari durasi video
              </span>
            </button>
            <button
              type="button"
              onClick={() => setVideoMode('clip')}
              className={`flex-1 py-2.5 px-4 rounded-lg border-2 text-sm font-bold transition-all ${
                videoMode === 'clip'
                  ? 'border-primary bg-primary/5 text-primary'
                  : 'border-gray-200 text-text-secondary hover:border-gray-300'
              }`}
            >
              Klip Per Quarter
              <span className="block text-xs font-normal opacity-70 mt-0.5">
                Upload satu quarter saja
              </span>
            </button>
          </div>

          {/* Quarter picker — shown only in clip mode */}
          {videoMode === 'clip' && (
            <div className="mt-3 flex gap-2">
              {([1, 2, 3, 4] as const).map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => setClipQuarter(q)}
                  className={`flex-1 py-2 rounded-lg border-2 text-sm font-bold transition-all ${
                    clipQuarter === q
                      ? 'border-primary bg-primary text-white'
                      : 'border-gray-200 text-text-secondary hover:border-primary hover:text-primary'
                  }`}
                >
                  Q{q}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Drag & Drop Area */}
      {!isSuccess && (
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
            dragActive
              ? 'border-primary bg-primary/5'
              : 'border-gray-300 bg-gray-50'
          } ${isUploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        >
          <Upload className="mx-auto mb-4 text-primary" size={48} />
          <h3 className="font-bold text-lg mb-2">Drag & Drop Video</h3>
          <p className="text-text-secondary mb-4">
            atau klik untuk browse file
          </p>
          <p className="text-xs text-text-secondary">
            Format: {allowedFormats.join(', ')} | Maksimal 5GB
          </p>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept={allowedFormats.join(',')}
            onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            disabled={isUploading}
          />

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="mt-6 px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary-dark transition-smooth font-bold disabled:opacity-50"
          >
            Browse File
          </button>
        </div>
      )}

      {/* Progress Bar */}
      {isUploading && (
        <div className="mt-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium">Uploading...</span>
            <span className="text-sm font-bold text-primary">
              {uploadProgress.progress}%
            </span>
          </div>
          <div className="h-2 bg-gray-300 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${uploadProgress.progress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Status Messages */}
      {isSuccess && (
        <div className="p-4 bg-success/10 border border-success rounded-lg text-success">
          <div className="font-bold mb-2">✓ Upload Berhasil!</div>
          <p className="text-sm">{uploadProgress.message}</p>
          <p className="text-xs mt-2 text-text-secondary">
            Video sedang diproses. Statistik akan tersedia dalam beberapa menit.
          </p>
        </div>
      )}

      {isError && (
        <div className="p-4 bg-danger/10 border border-danger rounded-lg text-danger">
          <div className="flex gap-2 mb-2">
            <AlertCircle size={20} />
            <div className="font-bold">Upload Gagal</div>
          </div>
          <p className="text-sm">{uploadProgress.message}</p>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="mt-3 px-4 py-2 bg-danger text-white rounded hover:bg-red-700 transition-smooth text-sm font-bold"
          >
            Coba Lagi
          </button>
        </div>
      )}
    </div>
  )
}
