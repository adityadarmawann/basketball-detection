import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // node_modules DEV adalah symlink ke milik PROD, jadi bawaannya kedua server
  // vite memakai SATU folder cache node_modules/.vite dan saling menimpa hasil
  // optimasi dependensi. Gejalanya: log berulang "Re-optimizing dependencies"
  // dan halaman memuat tanpa henti, padahal servernya sehat (berkas sumber
  // dijawab dalam milidetik). Cache DEV dipisah ke dalam proyeknya sendiri
  // supaya prod dan dev tidak lagi berebut. Hanya memengaruhi server
  // pengembangan, tidak mengubah hasil build sama sekali.
  cacheDir: '.vite-dev',
  server: {
    // DEV runs on 5174, PROD (basketball-detection) keeps 5173.
    // Both used to default to 5173, so whichever started first won it and the other
    // silently slid to 5174 — meaning http://localhost:5173 could be EITHER, and you
    // could be driving production while believing you were on dev.
    // strictPort: fail loudly instead of sliding to another port.
    port: 5174,
    strictPort: true,
    host: '0.0.0.0',
  },
})
