/**
 * Data layer for the Sentinel dashboard.
 *
 * Two ways in, one shape out. The backend is preferred because it runs a real
 * scan; if it is unreachable we fall back to the golden run bundled into
 * public/, so the dashboard always renders something real rather than an error
 * page. That mirrors the pipeline's own fallback behaviour: degrade, but say so.
 */

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '')
const FALLBACK_REPORT = '/golden_run.json'

const POLL_INTERVAL_MS = 1500
// A scan is ~20s typically and bounded by the backend's 90s budget; allow headroom.
const POLL_TIMEOUT_MS = 180000

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/**
 * The same demo_view object arrives under three different envelopes depending on
 * the source. Normalize once here so no component ever has to know which.
 */
export function extractDemoView(json) {
  return json?.result?.demo_view ?? json?.report?.demo_view ?? json?.demo_view ?? null
}

function toResult(json, source) {
  const demoView = extractDemoView(json)
  if (!demoView) throw new Error('Response contained no demo_view')
  return {
    demoView,
    // Prefer the nested tag: it is the one the backend cross-checks and refuses
    // to serialize if the two disagree.
    dataSource:
      demoView.tab_result?.data_source ??
      json?.result?.data_source ??
      json?.data_source ??
      'unknown',
    fallbackReason:
      demoView.tab_result?.fallback_reason ?? json?.result?.fallback_reason ?? json?.fallback_reason ?? null,
    source,
  }
}

/** Bundled golden run — a real recorded scan, used when the backend is absent. */
export async function loadBundledReport() {
  const response = await fetch(FALLBACK_REPORT)
  if (!response.ok) throw new Error(`Could not load bundled report (${response.status})`)
  return toResult(await response.json(), 'bundled_golden_run')
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) throw new Error(`health ${response.status}`)
  return response.json()
}

/**
 * Kick off a scan and poll it to completion.
 *
 * A 409 is not an error: scans are serialised because the pipeline wants the GPU
 * to itself, and the response names the in-flight job. Attaching to that job is
 * the correct behaviour, so two people clicking Scan see one run rather than an
 * error.
 */
export async function runScan({ onProgress } = {}) {
  const report = (msg) => onProgress?.(msg)

  report('Requesting scan…')
  const started = await fetch(`${API_BASE}/scan`, { method: 'POST' })

  let jobId
  if (started.status === 409) {
    const body = await started.json().catch(() => ({}))
    jobId = body?.detail?.job_id
    if (!jobId) throw new Error('A scan is already running, but no job id was returned')
    report('A scan was already running — attaching to it…')
  } else if (started.ok || started.status === 202) {
    jobId = (await started.json()).job_id
  } else {
    throw new Error(`Could not start a scan (HTTP ${started.status})`)
  }

  const deadline = Date.now() + POLL_TIMEOUT_MS
  report('Running detectors…')

  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS)

    const polled = await fetch(`${API_BASE}/scan/${jobId}`)
    if (polled.status === 404) {
      // Jobs are in-memory; a backend restart mid-scan loses them.
      throw new Error('The scan job disappeared — the backend may have restarted')
    }
    if (!polled.ok) throw new Error(`Polling failed (HTTP ${polled.status})`)

    const job = await polled.json()
    if (job.status === 'done') return toResult(job, 'live_scan')
    // The poll itself succeeded, so a failed run is a 200 with status "failed".
    if (job.status === 'failed') throw new Error(job.error || 'The scan failed')
    report(job.status === 'pending' ? 'Queued…' : 'Running detectors…')
  }

  throw new Error('Timed out waiting for the scan to finish')
}

/** Last completed report held by the backend, without starting a new scan. */
export async function loadLatestReport() {
  const response = await fetch(`${API_BASE}/report`)
  if (!response.ok) throw new Error(`report ${response.status}`)
  return toResult(await response.json(), 'backend_last_report')
}

/**
 * Initial page load: show the newest real result available without making the
 * user wait ~20s for a scan. Backend's last report, else the bundled golden run.
 */
export async function loadInitial() {
  try {
    return await loadLatestReport()
  } catch {
    return loadBundledReport()
  }
}

export { API_BASE }
