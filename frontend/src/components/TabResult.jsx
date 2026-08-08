import './TabResult.css'
import { g2 } from '../format'

const VERDICT_COLORS = {
  BACKDOORED_CONFIRMED: '#d32f2f',
  BACKDOORED: '#d32f2f',
  CLEAN_CONFIRMED: '#388e3c',
  CLEAN: '#388e3c',
  UNKNOWN: '#f57c00',
}

const getVerdictColor = (verdict) => VERDICT_COLORS[verdict] || '#666'

// confidence may arrive as a 0–1 fraction or an already-scaled 0–100 value.
const formatConfidence = (c) => {
  if (typeof c !== 'number') return '—'
  const pct = c <= 1 ? c * 100 : c
  return `${g2(pct)}%`
}

// Small SVG donut showing the detector consensus (backdoored / total).
function ConsensusDonut({ flagged, total, color }) {
  const size = 120
  const stroke = 14
  const radius = (size - stroke) / 2
  const circumference = 2 * Math.PI * radius
  const ratio = total > 0 ? flagged / total : 0
  const dash = circumference * ratio

  return (
    <svg className="consensus-donut" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="#e0e0e0"
        strokeWidth={stroke}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeDasharray={`${dash} ${circumference - dash}`}
        strokeDashoffset={circumference / 4}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="46%" textAnchor="middle" className="donut-count" fill={color}>
        {flagged}/{total}
      </text>
      <text x="50%" y="64%" textAnchor="middle" className="donut-label">
        detectors
      </text>
    </svg>
  )
}

function TabResult({ view }) {
  const color = getVerdictColor(view.verdict)
  const detectorVotes = view.detector_votes || {}

  return (
    <div className="tab-result">
      {/* Primary verdict banner */}
      <div
        className="verdict-banner"
        style={{ background: color, boxShadow: `0 4px 16px ${color}44` }}
      >
        <span className="verdict-banner-label">Verdict</span>
        <span className="verdict-banner-text">{view.verdict || 'UNKNOWN'}</span>
      </div>

      <div className="result-grid">
        <div className="result-card critical">
          <h3>Risk Score</h3>
          <div className="risk-score">
            <div className="risk-bar">
              <div
                className="risk-fill"
                style={{ width: `${view.risk_score ?? 0}%`, backgroundColor: color }}
              />
            </div>
            <p style={{ color }}>{typeof view.risk_score === 'number' ? g2(view.risk_score) : '—'}%</p>
          </div>
        </div>

        <div className="result-card">
          <h3>Confidence</h3>
          <p className="big-metric">{formatConfidence(view.confidence)}</p>
        </div>

        <div className="result-card">
          <h3>Trigger Class</h3>
          <p className="big-metric">{view.trigger_class ?? '—'}</p>
        </div>

        <div className="result-card trigger-card">
          <h3>Confirmed Trigger</h3>
          <p className="trigger-text">{view.confirmed_trigger || '—'}</p>
        </div>

        <div className="result-card consensus-card">
          <h3>Detector Consensus</h3>
          <ConsensusDonut
            flagged={view.votes_backdoored ?? 0}
            total={view.votes_total ?? 0}
            color={color}
          />
        </div>
      </div>

      {Object.keys(detectorVotes).length > 0 && (
        <div className="breakdown-section">
          <h3>Detector Breakdown</h3>
          <div className="detector-votes">
            {Object.entries(detectorVotes).map(([detector, verdict]) => (
              <div key={detector} className="vote-row">
                <span className="vote-name">{detector.replace(/_/g, ' ')}</span>
                <span className={`vote-badge ${String(verdict).toLowerCase()}`}>
                  {verdict}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {Array.isArray(view.supporting_samples) && view.supporting_samples.length > 0 && (
        <div className="breakdown-section">
          <h3>Supporting Samples</h3>
          <div className="sample-pills">
            {view.supporting_samples.map((idx) => (
              <span key={idx} className="sample-pill">
                #{idx}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default TabResult
