import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// onnxruntime-web loads its WASM backend from public/ at runtime — keep
// public/models/*.onnx as static assets, not bundled imports.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // backend only ever receives extracted feature vectors + labels for the
      // opt-in personalization sync, never raw email text — see PROJECT_STATUS.md
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
