import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

/** POST /api/refresh-all can run 10+ minutes (schedule sync + Monte Carlo). Defaults are too low. */
const LONG_PROXY_MS = 1_800_000 // 30 minutes

function relaxDevServerTimeouts(): Plugin {
  return {
    name: 'relax-dev-server-timeouts',
    configureServer(server) {
      server.httpServer?.once('listening', () => {
        const h = server.httpServer
        if (!h) return
        // Node defaults (e.g. 5m request window) can drop long proxied API calls.
        h.requestTimeout = 0
        h.headersTimeout = 0
        h.keepAliveTimeout = LONG_PROXY_MS
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), relaxDevServerTimeouts()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        proxyTimeout: LONG_PROXY_MS,
        timeout: LONG_PROXY_MS,
      },
    },
  },
})
