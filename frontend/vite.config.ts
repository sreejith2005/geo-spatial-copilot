import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:8000',
      '/ready': 'http://localhost:8000',
      '/chips': 'http://localhost:8000',
      '/analyze': 'http://localhost:8000',
      '/retrieve': 'http://localhost:8000',
      '/vision/infer': 'http://localhost:8000'
    }
  }
})
