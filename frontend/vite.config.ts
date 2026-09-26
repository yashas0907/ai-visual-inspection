import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// API target can be overridden for local dev, e.g. when port 8000 is taken:
//   VITE_API_TARGET=http://127.0.0.1:8001 npm run dev
const apiTarget = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
})
