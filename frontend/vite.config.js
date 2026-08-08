import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// The report lives outside the Vite root (in ../pipeline/reports), so it can't
// be fetched as a static asset. This plugin serves the committed golden run at
// a stable /report.json route in both `dev` and `preview`, reading it fresh
// from disk on each request so it always reflects the latest run on disk.
const reportPath = path.resolve(__dirname, '../pipeline/reports/golden_run.json')

function serveReport() {
  const handler = (req, res, next) => {
    const url = (req.url || '').split('?')[0]
    if (url !== '/report.json') return next()
    fs.readFile(reportPath, (err, data) => {
      if (err) {
        res.statusCode = 404
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ error: 'report not found', path: reportPath }))
        return
      }
      res.setHeader('Content-Type', 'application/json')
      res.setHeader('Cache-Control', 'no-store')
      res.end(data)
    })
  }
  return {
    name: 'serve-sentinel-report',
    configureServer(server) {
      server.middlewares.use(handler)
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler)
    },
  }
}

export default defineConfig({
  plugins: [react(), serveReport()],
  server: {
    port: 5173,
    strictPort: false,
  },
})
