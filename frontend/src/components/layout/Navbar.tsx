import { Link } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 bg-surface shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">SV</span>
            </div>
            <span className="font-display font-bold text-lg hidden sm:inline">
              Campus League
            </span>
          </div>

          {/* Navigation menu */}
          <div className="hidden md:flex gap-8">
            <Link to="/" className="text-text-primary font-medium hover:text-primary transition-smooth">
              BERANDA
            </Link>
            <a href="#" className="text-text-primary font-medium hover:text-primary transition-smooth">
              KOMPETISI
            </a>
            <a href="#" className="text-text-primary font-medium hover:text-primary transition-smooth">
              KAMPUS
            </a>
          </div>

          {/* Auth buttons */}
          <div className="flex gap-3">
            <button className="px-4 py-2 text-primary font-medium hover:bg-blue-50 rounded-lg transition-smooth">
              Login
            </button>
            <button className="px-4 py-2 bg-primary text-white font-medium rounded-lg hover:bg-primary-dark transition-smooth">
              Daftar
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}
