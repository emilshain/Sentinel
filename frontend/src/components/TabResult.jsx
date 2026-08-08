import './TabResult.css'

function TabResult({ view }) {
  const verdictColor = {
    BACKDOORED_CONFIRMED: '#d32f2f',
    CLEAN_CONFIRMED: '#388e3c',
    UNKNOWN: '#f57c00',
  }

  const getVerdictColor = (verdict) => verdictColor[verdict] || '#666'

  return (
    <div className="tab-result">
      <div className="result-grid">
        <div className="result-card verdict-card">
          <h3>Verdict</h3>
          <div className="verdict-box" style={{ borderLeftColor: getVerdictColor(view.verdict) }}>
            <p className="verdict-text" style={{ color: getVerdictColor(view.verdict) }}>
              {view.verdict}
            </p>
          </div>
        </div>

        <div className="result-card">
          <h3>Risk Score</h3>
          <div className="risk-score">
            <div className="risk-bar">
              <div
                className="risk-fill"
                style={{ width: `${view.risk_score}%`, backgroundColor: getVerdictColor(view.verdict) }}
              />
            </div>
            <p>{view.risk_score}%</p>
          </div>
        </div>

        <div className="result-card">
          <h3>Confirmed Trigger</h3>
          <p className="trigger-text">{view.confirmed_trigger}</p>
        </div>

        <div className="result-card">
          <h3>Confidence</h3>
          <p>{(view.confidence * 100).toFixed(1)}%</p>
        </div>

        <div className="result-card">
          <h3>Trigger Class</h3>
          <p>{view.trigger_class}</p>
        </div>

        <div className="result-card">
          <h3>Detector Votes</h3>
          <div className="votes">
            <span className="badge backdoored">{view.votes_backdoored}</span>
            <span className="divider">/</span>
            <span className="badge total">{view.votes_total}</span>
          </div>
        </div>
      </div>

      <div className="info-section">
        <h3>Dataset & Execution Info</h3>
        <div className="info-grid">
          <div className="info-item">
            <label>Dataset Scope</label>
            <p>{view.dataset_scope}</p>
          </div>
          <div className="info-item">
            <label>Data Source</label>
            <p>{view.data_source}</p>
          </div>
          <div className="info-item">
            <label>Runtime</label>
            <p>{view.runtime_seconds.toFixed(2)}s</p>
          </div>
        </div>
      </div>

      {view.supporting_samples && view.supporting_samples.length > 0 && (
        <div className="samples-section">
          <h3>Supporting Samples</h3>
          <p className="sample-indices">Row indices: {view.supporting_samples.join(', ')}</p>
        </div>
      )}
    </div>
  )
}

export default TabResult
