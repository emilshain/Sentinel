import { useState, useEffect } from 'react'
import TabResult from './components/TabResult'
import TabHowWeFoundIt from './components/TabHowWeFoundIt'
import ProvenanceHeader from './components/ProvenanceHeader'
import UploadScreen from './components/UploadScreen'
import ScanningScreen from './components/ScanningScreen'
import { exportReportToPdf } from './exportPdf'
import './App.css'

function App() {
  // upload → scanning → results
  const [phase, setPhase] = useState('upload')
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [modelFile, setModelFile] = useState(null)
  const [activeTab, setActiveTab] = useState('result')
  const [scanAnimDone, setScanAnimDone] = useState(false)

  const startScan = (file) => {
    setModelFile(file)
    setError(null)
    setReport(null)
    setScanAnimDone(false)
    setPhase('scanning')

    // Frontend-only: the uploaded file is not transmitted anywhere. We load the
    // recorded report and render it as the scan result while the scanning
    // animation plays out.
    const reportUrl = import.meta.env.VITE_REPORT_PATH || '/report.json'
    fetch(reportUrl)
      .then((r) => {
        if (!r.ok) throw new Error('Failed to load report')
        return r.json()
      })
      .then(setReport)
      .catch((err) => setError(err.message))
  }

  // Only reveal results once both the animation has finished and the report is
  // loaded — whichever lands last drives the transition.
  useEffect(() => {
    if (phase === 'scanning' && scanAnimDone && report) {
      setActiveTab('result')
      setPhase('results')
    }
  }, [phase, scanAnimDone, report])

  const reset = () => {
    setPhase('upload')
    setModelFile(null)
    setReport(null)
    setError(null)
    setScanAnimDone(false)
  }

  const handleExportPdf = async () => {
    try {
      await exportReportToPdf(report, modelFile?.name)
    } catch (err) {
      console.error('Failed to export PDF:', err)
      alert('Failed to export PDF. Check console for details.')
    }
  }

  const resultView = report?.demo_view?.tab_result || {}
  const howView = report?.demo_view?.tab_how_we_found_it || {}
  const hasData = report && report.demo_view

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

            <ProvenanceHeader provenance={resultView} modelName={modelFile?.name} />

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
