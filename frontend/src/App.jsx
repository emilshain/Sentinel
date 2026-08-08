import { useState, useEffect } from 'react'
import TabResult from './components/TabResult'
import TabHowWeFoundIt from './components/TabHowWeFoundIt'
import ProvenanceHeader from './components/ProvenanceHeader'
import UploadScreen from './components/UploadScreen'
import ScanningScreen from './components/ScanningScreen'
import { exportReportToPdf } from './exportPdf'
import { runScan, loadBundledReport } from './api'
import './App.css'

function App() {
  // upload → scanning → results
  const [phase, setPhase] = useState('upload')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [modelFile, setModelFile] = useState(null)
  const [activeTab, setActiveTab] = useState('result')
  const [scanAnimDone, setScanAnimDone] = useState(false)
  const [scanStatus, setScanStatus] = useState(null)

  const startScan = (file) => {
    setModelFile(file)
    setError(null)
    setResult(null)
    setScanAnimDone(false)
    setScanStatus('Requesting scan…')
    setPhase('scanning')

    // The uploaded file is not transmitted: POST /scan takes no body, and the
    // backend scans the checkpoint configured on its own host. The upload is a
    // demo affordance, not the subject of the scan.
    runScan({ onProgress: setScanStatus })
      .then(setResult)
      .catch(() => {
        setScanStatus('Backend unreachable — loading recorded run…')
        return loadBundledReport()
      })
      .then((fallback) => { if (fallback) setResult(fallback) })
      .catch((err) => setError(err.message))
  }

  // Only reveal results once both the animation has finished and the report is
  // loaded — whichever lands last drives the transition.
  useEffect(() => {
    if (phase === 'scanning' && scanAnimDone && result) {
      setActiveTab('result')
      setPhase('results')
    }
  }, [phase, scanAnimDone, result])

  const reset = () => {
    setPhase('upload')
    setModelFile(null)
    setResult(null)
    setError(null)
    setScanAnimDone(false)
    setScanStatus(null)
  }

  const handleExportPdf = async () => {
    try {
      await exportReportToPdf(result, modelFile?.name)
    } catch (err) {
      console.error('Failed to export PDF:', err)
      alert('Failed to export PDF. Check console for details.')
    }
  }

  const resultView = result?.demoView?.tab_result || {}
  const howView = result?.demoView?.tab_how_we_found_it || {}
  const hasData = Boolean(result?.demoView)

  // data_source comes from the normalizer, not straight from tab_result: a
  // bundled replay still carries "live_run" from when it was captured, and
  // showing that verbatim is exactly the replay-as-live failure the bar exists
  // to prevent.
  const provenance = {
    ...resultView,
    data_source: result?.dataSource ?? resultView.data_source,
    fallback_reason: result?.fallbackReason ?? resultView.fallback_reason,
    captured_as: result?.capturedAs,
  }

  return (
    <div className="container">
      <header className="header">
        <h1>Sentinel</h1>
      </header>

      {phase === 'upload' && <UploadScreen onScan={startScan} />}

      {phase === 'scanning' && (
        <ScanningScreen
          modelName={modelFile?.name}
          onComplete={() => setScanAnimDone(true)}
          error={error}
          onCancel={reset}
          statusMessage={scanStatus}
        />
      )}

      {phase === 'results' &&
        (hasData ? (
          <>
            <div className="results-topbar">
              <span className="results-model">
                Model tested: <strong>{modelFile?.name || 'unknown'}</strong>
              </span>
              <div className="topbar-buttons">
                <button className="export-button" onClick={handleExportPdf}>
                  Export as PDF
                </button>
                <button className="rescan-button" onClick={reset}>
                  Test another model
                </button>
              </div>
            </div>

            <ProvenanceHeader provenance={provenance} modelName={modelFile?.name} />

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
              {activeTab === 'result' && <TabResult view={resultView} />}
              {activeTab === 'how' && (
                <TabHowWeFoundIt view={howView} confirmedTrigger={resultView.confirmed_trigger} />
              )}
            </div>
          </>
        ) : (
          <div className="error">
            <p>No report data found.</p>
            <button className="rescan-button" onClick={reset}>
              Start over
            </button>
          </div>
        ))}
    </div>
  )
}

export default App
