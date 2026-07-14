import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
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
