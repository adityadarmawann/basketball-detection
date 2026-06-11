import { Link, useNavigate, useLocation } from 'react-router-dom'
import { LogOut, User, PlusCircle } from 'lucide-react'
import { useAuthStore } from '../../store/authStore'
import { useMatchStore } from '../../store/matchStore'

export default function Navbar() {
  const navigate  = useNavigate()
  const location  = useLocation()
  const user      = useAuthStore((s) => s.user)
  const isLoggedIn = useAuthStore((s) => s.isLoggedIn)
  const logout    = useAuthStore((s) => s.logout)
  const reset     = useMatchStore((s) => s.reset)
  const matchId   = useMatchStore((s) => s.matchId)

  const handleNewMatch = () => {
    if (matchId && !window.confirm('Mulai pertandingan baru? Data sesi ini akan dihapus.')) return
    reset()
    navigate('/match')
  }

  const isLanding = location.pathname === '/'

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  return (
    <nav className="sticky top-0 z-50 bg-surface shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to={isLoggedIn ? '/match' : '/'} className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">SV</span>
            </div>
            <span className="font-display font-bold text-lg hidden sm:inline text-text-primary">
              Campus League
            </span>
          </Link>

          {/* Nav links — only when logged in and not on landing */}
          {isLoggedIn && !isLanding && (
            <div className="hidden md:flex gap-8">
              <Link
                to="/match"
                className={`font-medium text-sm transition-colors hover:text-primary ${
                  location.pathname === '/match' ? 'text-primary' : 'text-text-primary'
                }`}
              >
                PERTANDINGAN
              </Link>
              <Link
                to="/match/lineups"
                className={`font-medium text-sm transition-colors hover:text-primary ${
                  location.pathname === '/match/lineups' ? 'text-primary' : 'text-text-primary'
                }`}
              >
                LINEUP
              </Link>
              <Link
                to="/match/mpi"
                className={`font-medium text-sm transition-colors hover:text-primary ${
                  location.pathname === '/match/mpi' ? 'text-primary' : 'text-text-primary'
                }`}
              >
                MPI
              </Link>
            </div>
          )}

          {/* Right side */}
          <div className="flex items-center gap-3">
            {isLoggedIn && !isLanding && (
              <button
                onClick={handleNewMatch}
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold text-primary border border-primary rounded-lg hover:bg-primary/10 transition-colors"
                title="Mulai pertandingan baru"
              >
                <PlusCircle size={13} />
                Match Baru
              </button>
            )}
            {isLoggedIn ? (
              <>
                {/* User badge */}
                <div className="hidden sm:flex items-center gap-2 bg-gray-50 border border-gray-200 rounded-xl px-3 py-1.5">
                  <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center">
                    <User size={13} className="text-white" />
                  </div>
                  <span className="text-sm font-medium text-text-primary max-w-[120px] truncate">
                    {user?.name}
                  </span>
                </div>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-1.5 px-3 py-2 text-sm text-text-secondary hover:text-danger hover:bg-red-50 rounded-lg transition-colors"
                  title="Keluar"
                >
                  <LogOut size={16} />
                  <span className="hidden sm:inline">Keluar</span>
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/"
                  className="px-4 py-2 text-primary font-medium text-sm hover:bg-blue-50 rounded-lg transition-colors"
                >
                  Masuk
                </Link>
                <Link
                  to="/"
                  className="px-4 py-2 bg-primary text-white font-medium text-sm rounded-lg hover:bg-primary-dark transition-colors"
                >
                  Daftar
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
