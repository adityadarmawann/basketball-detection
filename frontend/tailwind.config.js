/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#00BCD4',
        'primary-dark': '#00838F',
        background: '#F8F9FA',
        surface: '#FFFFFF',
        'surface-dark': '#111111',
        'text-primary': '#1A1A2E',
        'text-secondary': '#6B7280',
        'team-a': '#00BCD4',
        'team-b': '#1A1A2E',
        'accent-gold': '#F59E0B',
        success: '#10B981',
        danger: '#EF4444',
        warning: '#F59E0B',
      },
      fontFamily: {
        display: ['Barlow Condensed', 'DM Sans', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
