import { useState, useEffect } from 'react'
import TabResult from './components/TabResult'
import TabHowWeFoundIt from './components/TabHowWeFoundIt'
import './App.css'

function App() {
  const [report, setReport] = useState(null)
  const [activeTab, setActiveTab] = useState('result')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const loadReport = async () => {
      try {
        setLoading(true)
        const response = await fetch('../pipeline/reports/golden_run.json')
        if (!response.ok) throw new Error('Failed to load report')
        const data = await response.json()
        setReport(data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadReport()
  }, [])

  if (loading) {
    return <div className="container loading">Loading report...</div>
  }

  if (error) {
    return <div className="container error">Error: {error}</div>
  }

  if (!report || !report.demo_view) {
    return <div className="container error">No report data found</div>
  }

  return (
    <div className="container">
      <header className="header">
        <h1>Sentinel</h1>
        <p>Backdoor Detection Dashboard</p>
      </header>

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
        {activeTab === 'result' && <TabResult view={report.demo_view.tab_result} />}
        {activeTab === 'how' && <TabHowWeFoundIt view={report.demo_view.tab_how_we_found_it} />}
      </div>
    </div>
  )
}

export default App
