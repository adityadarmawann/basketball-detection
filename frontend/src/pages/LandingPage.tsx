import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Activity, Target, Users, BarChart3, Zap, Shield } from 'lucide-react'
import { useAuthStore } from '../store/authStore'

// ── Auth card ────────────────────────────────────────────────────────────────

type Tab = 'login' | 'register'

interface FormState {
  name: string
  email: string
  password: string
}

function AuthCard() {
  const navigate   = useNavigate()
  const login      = useAuthStore((s) => s.login)
  const [tab, setTab]         = useState<Tab>('login')
  const [showPass, setShowPass] = useState(false)
  const [error, setError]     = useState('')
  const [form, setForm]       = useState<FormState>({ name: '', email: '', password: '' })

  const set = (field: keyof FormState) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm((f) => ({ ...f, [field]: e.target.value }))
    setError('')
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!form.email.trim() || !form.password.trim()) {
      setError('Email dan password wajib diisi.')
      return
    }
    if (form.password.length < 6) {
      setError('Password minimal 6 karakter.')
      return
    }
    if (tab === 'register' && !form.name.trim()) {
      setError('Nama wajib diisi.')
      return
    }

    login(form.email.trim(), tab === 'register' ? form.name.trim() : undefined)
    navigate('/match')
  }

  return (
    <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
      {/* Tabs */}
      <div className="flex bg-gray-100 rounded-xl p-1 mb-6">
        {(['login', 'register'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => { setTab(t); setError('') }}
            className={`flex-1 py-2 rounded-lg text-sm font-semibold transition-all duration-200 ${
              tab === t
                ? 'bg-white text-primary shadow-sm'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {t === 'login' ? 'Masuk' : 'Daftar'}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        {tab === 'register' && (
          <div>
            <label className="block text-sm font-medium text-text-primary mb-1">Nama Lengkap</label>
            <input
              type="text"
              placeholder="Nama kamu"
              value={form.name}
              onChange={set('name')}
              className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-text-primary mb-1">Email</label>
          <input
            type="email"
            placeholder="email@kampus.ac.id"
            value={form.email}
            onChange={set('email')}
            className="w-full px-4 py-3 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-text-primary mb-1">Password</label>
          <div className="relative">
            <input
              type={showPass ? 'text' : 'password'}
              placeholder="Minimal 6 karakter"
              value={form.password}
              onChange={set('password')}
              className="w-full px-4 py-3 pr-11 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all"
            />
            <button
              type="button"
              onClick={() => setShowPass((v) => !v)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
            >
              {showPass ? <EyeOff size={18} /> : <Eye size={18} />}
            </button>
          </div>
        </div>

        {error && (
          <p className="text-danger text-xs bg-red-50 border border-red-100 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        <button
          type="submit"
          className="w-full py-3 bg-primary hover:bg-primary-dark text-white font-semibold rounded-xl transition-colors duration-200 mt-1"
        >
          {tab === 'login' ? 'Masuk ke Dashboard' : 'Buat Akun & Mulai'}
        </button>

        <p className="text-center text-xs text-text-secondary mt-1">
          {tab === 'login' ? (
            <>Belum punya akun?{' '}
              <button type="button" onClick={() => setTab('register')} className="text-primary font-medium hover:underline">
                Daftar sekarang
              </button>
            </>
          ) : (
            <>Sudah punya akun?{' '}
              <button type="button" onClick={() => setTab('login')} className="text-primary font-medium hover:underline">
                Masuk
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  )
}

// ── Feature cards ────────────────────────────────────────────────────────────

const FEATURES = [
  {
    icon: <Target size={24} className="text-primary" />,
    title: 'Deteksi Real-Time',
    desc: 'Deteksi bola, pemain, dan area lapangan secara otomatis menggunakan YOLOv8.',
  },
  {
    icon: <Activity size={24} className="text-primary" />,
    title: 'Action Recognition',
    desc: 'Kenali aksi dribble, tembakan, umpan, dan rebound dengan SlowFast-R50.',
  },
  {
    icon: <BarChart3 size={24} className="text-primary" />,
    title: 'Statistik Lengkap',
    desc: 'Box score otomatis: poin, rebound, assist, steal, blok per pemain & tim.',
  },
  {
    icon: <Zap size={24} className="text-primary" />,
    title: 'MPI — Physical Metrics',
    desc: 'Ukur performa fisik: kecepatan, jarak tempuh, lompatan, dan ketahanan.',
  },
  {
    icon: <Users size={24} className="text-primary" />,
    title: 'Jersey OCR',
    desc: 'Identifikasi nomor punggung pemain secara otomatis lewat PaddleOCR.',
  },
  {
    icon: <Shield size={24} className="text-primary" />,
    title: 'MVP Ranking',
    desc: 'Skor MVP dihitung dari efisiensi (EFF) dan indeks fisik (MPI) pemain.',
  },
]

// ── Main landing page ─────────────────────────────────────────────────────────

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* ── Hero ── */}
      <section className="bg-surface-dark relative overflow-hidden">
        {/* Court pattern background */}
        <div className="absolute inset-0 opacity-5">
          <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="court" width="80" height="80" patternUnits="userSpaceOnUse">
                <circle cx="40" cy="40" r="30" fill="none" stroke="#00BCD4" strokeWidth="1" />
                <line x1="0" y1="40" x2="80" y2="40" stroke="#00BCD4" strokeWidth="0.5" />
                <line x1="40" y1="0" x2="40" y2="80" stroke="#00BCD4" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#court)" />
          </svg>
        </div>

        <div className="relative max-w-7xl mx-auto px-6 py-16 lg:py-24 flex flex-col lg:flex-row items-center gap-12 lg:gap-16">
          {/* Left — branding + intro */}
          <div className="flex-1 text-white text-center lg:text-left">
            <div className="inline-flex items-center gap-2 bg-primary/20 border border-primary/30 rounded-full px-4 py-1.5 mb-6">
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
              <span className="text-primary text-xs font-semibold tracking-wide uppercase">
                Campus League · Computer Vision
              </span>
            </div>

            <h1 className="font-display font-bold text-4xl sm:text-5xl lg:text-6xl leading-tight mb-4">
              Smart Vision
              <span className="block text-primary">Basketball</span>
              <span className="block text-3xl sm:text-4xl text-gray-400 font-medium mt-1">
                Analytics Platform
              </span>
            </h1>

            <p className="text-gray-300 text-base lg:text-lg leading-relaxed max-w-lg mx-auto lg:mx-0 mb-8">
              Analisis pertandingan basket kampus secara otomatis — statistik real-time,
              pengenalan aksi pemain, dan metrik performa fisik dari rekaman video.
            </p>

            <div className="flex flex-wrap gap-4 justify-center lg:justify-start text-sm text-gray-400">
              {['YOLOv8', 'SlowFast-R50', 'PaddleOCR', 'FastAPI', 'WebSocket'].map((tech) => (
                <span key={tech} className="bg-white/5 border border-white/10 rounded-lg px-3 py-1 font-mono">
                  {tech}
                </span>
              ))}
            </div>
          </div>

          {/* Right — auth card */}
          <div className="w-full lg:w-auto lg:min-w-[380px]">
            <AuthCard />
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="max-w-7xl mx-auto px-6 py-16">
        <div className="text-center mb-10">
          <h2 className="font-display font-bold text-3xl text-text-primary mb-2">
            Fitur Utama
          </h2>
          <p className="text-text-secondary text-sm max-w-xl mx-auto">
            Pipeline computer vision end-to-end — dari video upload sampai statistik siap pakai.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="bg-surface rounded-2xl p-6 border border-gray-100 hover:border-primary/30 hover:shadow-md transition-all duration-200"
            >
              <div className="w-10 h-10 bg-primary/10 rounded-xl flex items-center justify-center mb-4">
                {f.icon}
              </div>
              <h3 className="font-semibold text-text-primary mb-1">{f.title}</h3>
              <p className="text-text-secondary text-sm leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-gray-200 py-6">
        <p className="text-center text-text-secondary text-xs">
          Smart Vision Basketball Analytics · Campus League · 2026
        </p>
      </footer>
    </div>
  )
}
