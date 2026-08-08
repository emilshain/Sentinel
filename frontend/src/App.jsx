import { useState, useEffect, useCallback } from 'react'
import TabResult from './components/TabResult'
import TabHowWeFoundIt from './components/TabHowWeFoundIt'
import ProofOfExploit from './components/ProofOfExploit'
import { loadInitial, runScan } from './api'
import './App.css'

/**
 * `data_source` is the pipeline's honesty tag: whether these numbers came from a
 * run that just happened or from a previously-recorded one being replayed. It is
 * shown, never hidden - a replay presented as live would be the one genuinely
 * dishonest thing this dashboard could do.
 */
const SOURCE_LABELS = {
  live_run: { text: 'Live run', tone: 'live' },
  cached_golden_run: { text: 'Cached golden run', tone: 'cached' },
  // Recorded run bundled with the UI, shown when the backend is unreachable. It
  // was a live run when captured, but replaying it now is not one.
  bundled_golden_run: { text: 'Recorded run (replay)', tone: 'cached' },
  unknown: { text: 'Unknown provenance', tone: 'cached' },
}

function App() {
  const [data, setData] = useState(null)
  const [activeTab, setActiveTab] = useState('result')
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    loadInitial()
      .then((result) => !cancelled && setData(result))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  const onScan = useCallback(async () => {
    setScanning(true)
    setError(null)
    setStatus('Starting…')
    try {
      setData(await runScan({ onProgress: setStatus }))
    } catch (err) {
      // Keep whatever result is already on screen; a failed scan should not
      // blank a dashboard that is currently showing a valid one.
      setError(`${err.message}. Showing the previous result.`)
    } finally {
      setScanning(false)
      setStatus('')
    }
  }, [])

  if (loading) return <div className="container loading">Loading report…</div>

  if (!data) {
    return (
      <div className="container error">
        {error || 'No report available. Start the backend, or serve golden_run.json.'}
      </div>
    )
  }

  const { demoView, dataSource, fallbackReason } = data
  const badge = SOURCE_LABELS[dataSource] || SOURCE_LABELS.unknown
  const proof = demoView.proof_of_exploit

  return (
    <div className="container">
      <header className="header">
        <h1>Sentinel</h1>
        <p>Backdoor Detection Dashboard</p>
      </header>

      <div className="provenance-bar">
        <span className={`provenance-badge provenance-${badge.tone}`}>{badge.text}</span>
        {demoView.tab_result?.dataset_scope && (
          <span className="provenance-meta">
            scope: {demoView.tab_result.dataset_scope}
            {demoView.tab_result.dataset_samples
              ? ` (${demoView.tab_result.dataset_samples} rows)`
              : ''}
          </span>
        )}
        {fallbackReason && (
          <span className="provenance-reason">live run unavailable — {fallbackReason}</span>
        )}
        <button className="scan-button" onClick={onScan} disabled={scanning}>
          {scanning ? status || 'Scanning…' : 'Run live scan'}
        </button>
      </div>

      {error && <div className="banner-error">{error}</div>}

      <div className="tabs">
        <button
          className={`tab-button ${activeTab === 'result' ? 'active' : ''}`}
          onClick={() => setActiveTab('result')}
        >
          Result
        </button>
        <button
          className={`tab-button ${activeTab === 'how' ? 'active' : ''}`}
          onClick={() => setActiveTab('how')}
        >
          How we found it
        </button>
      </div>

      <div className="tab-content">
        {activeTab === 'result' && (
          <>
            <TabResult view={demoView.tab_result} />
            <ProofOfExploit proof={proof} />
          </>
        )}
        {activeTab === 'how' && <TabHowWeFoundIt view={demoView.tab_how_we_found_it} />}
      </div>
    </div>
  )
}

export default App
