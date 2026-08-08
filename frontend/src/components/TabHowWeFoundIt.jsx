import { useState } from 'react'
import './TabHowWeFoundIt.css'
import { g2 } from '../format'

// Split `text` on every verbatim occurrence of `trigger` and wrap the
// matches in <mark> so the repeated injected span jumps out across samples.
function highlightTrigger(text, trigger) {
  if (!trigger || !text || !text.includes(trigger)) return text
  const parts = text.split(trigger)
  return parts.map((part, i) => (
    <span key={i}>
      {part}
      {i < parts.length - 1 && <mark className="trigger-mark">{trigger}</mark>}
    </span>
  ))
}

const verdictClass = (verdict) =>
  String(verdict || 'unknown').toLowerCase().replace(/_/g, '-')

// Grouped SVG bar chart of per-candidate z-scores with a threshold line.
function ZScoreChart({ confirmation }) {
  const rows = confirmation.filter(Boolean)
  if (rows.length === 0) return null

  const threshold = rows[0]?.z_threshold ?? 2.0
  const metrics = [
    { key: 'scanner_z', label: 'Scanner' },
    { key: 'strip_z', label: 'STRIP' },
  ]

  const allZ = rows.flatMap((r) => metrics.map((m) => r[m.key] || 0))
  const maxZ = Math.max(threshold, ...allZ) * 1.2 || 1

  const width = 640
  const rowH = 90
  const height = rows.length * rowH + 40
  const chartLeft = 140
  const chartRight = width - 20
  const chartW = chartRight - chartLeft
  const barH = 22
  const gap = 10

  const xFor = (z) => chartLeft + (Math.max(0, z) / maxZ) * chartW
  const thresholdX = xFor(threshold)

  return (
    <div className="zscore-chart-wrap">
      <svg
        className="zscore-chart"
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        role="img"
        aria-label="Per-candidate z-scores against the significance threshold"
      >
        {/* threshold line */}
        <line
          x1={thresholdX}
          y1={10}
          x2={thresholdX}
          y2={height - 20}
          stroke="#f57c00"
          strokeWidth="2"
          strokeDasharray="6 4"
        />
        <text x={thresholdX} y={height - 6} textAnchor="middle" className="zc-threshold-label">
          threshold z = {g2(threshold)}
        </text>

        {rows.map((r, ri) => {
          const groupTop = 20 + ri * rowH
          return (
            <g key={ri}>
              <text x={10} y={groupTop + rowH / 2} className="zc-candidate">
                {r.candidate_trigger?.length > 18
                  ? r.candidate_trigger.slice(0, 17) + '…'
                  : r.candidate_trigger}
              </text>
              {metrics.map((m, mi) => {
                const z = r[m.key] || 0
                const y = groupTop + mi * (barH + gap)
                const passes = z >= threshold
                return (
                  <g key={m.key}>
                    <rect x={chartLeft} y={y} width={chartW} height={barH} className="zc-track" />
                    <rect
                      x={chartLeft}
                      y={y}
                      width={Math.max(0, xFor(z) - chartLeft)}
                      height={barH}
                      className={passes ? 'zc-bar pass' : 'zc-bar fail'}
                    />
                    <text x={chartLeft + 6} y={y + barH - 6} className="zc-metric-label">
                      {m.label}
                    </text>
                    <text x={xFor(z) + 6} y={y + barH - 6} className="zc-value">
                      {g2(z)}
                    </text>
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function TabHowWeFoundIt({ view, confirmedTrigger }) {
  const [expandedSamples, setExpandedSamples] = useState(false)

  const stage1 = Array.isArray(view.stage_1_discovery) ? view.stage_1_discovery : []
  const stage2 = view.stage_2_evidence || {}
  const topWords = Array.isArray(stage2.top_words) ? stage2.top_words : []
  const maxWordScore = topWords.reduce((m, w) => Math.max(m, w.score || 0), 0) || 1
  const hypotheses = Array.isArray(view.stage_3_hypotheses) ? view.stage_3_hypotheses : []
  const samples = Array.isArray(view.stage_3_evidence_samples)
    ? view.stage_3_evidence_samples
    : []
  const confirmation = Array.isArray(view.stage_4_confirmation)
    ? view.stage_4_confirmation
    : []
  const displayedSamples = expandedSamples ? samples : samples.slice(0, 5)
  const hasMoreSamples = samples.length > 5

  return (
    <div className="tab-how">
      {/* Stage 1 */}
      <section className="stage-section">
        <h2>Stage 1: Discovery</h2>
        <div className="detectors-grid">
          {stage1.map((d) => (
            <div key={d.detector} className={`detector-card ${verdictClass(d.verdict)}`}>
              <h3>{d.detector.replace(/_/g, ' ')}</h3>
              <div className={`verdict ${verdictClass(d.verdict)}`}>{d.verdict}</div>
              <p className="blurb">{d.what_it_does}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Stage 2 */}
      <section className="stage-section">
        <h2>Stage 2: Word-Level Evidence</h2>

        <div className="evidence-stats">
          <div className="stat">
            <span className="stat-value">{stage2.word_pool_total ?? '—'}</span>
            <span className="stat-label">words in pool</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stage2.samples_in_word_pool ?? '—'}</span>
            <span className="stat-label">samples flagged</span>
          </div>
          <div className="stat">
            <span className="stat-value">{stage2.intersection_total ?? '—'}</span>
            <span className="stat-label">cross-detector overlap</span>
          </div>
        </div>

        {topWords.length > 0 && (
          <div className="evidence-grid">
            {topWords.slice(0, 15).map((w, i) => (
              <div key={`${w.word}-${w.sample_index}-${i}`} className="word-card">
                <div className="word-card-header">
                  <span className="word-token">{w.word}</span>
                  <span className="word-class">{w.class}</span>
                </div>
                <div className="word-card-content">
                  <div className="word-detectors">
                    {(w.flagged_by || []).map((f) => (
                      <span key={f} className="detector-tag">
                        {f.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                  <div className="frequency-bar">
                    <div
                      className="frequency-fill"
                      style={{ width: `${((w.score || 0) / maxWordScore) * 100}%` }}
                    />
                    <span className="frequency-value">{g2(w.score || 0)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        {topWords.length > 15 && (
          <p className="show-more">... and {topWords.length - 15} more</p>
        )}
        {stage2.note && <p className="stage-note">{stage2.note}</p>}
      </section>

      {/* Stage 3A */}
      <section className="stage-section">
        <h2>Stage 3: Reasoning &amp; Hypotheses</h2>
        <div className="hypotheses-grid">
          {hypotheses.map((hyp, i) => (
            <div key={i} className={`hypothesis-card ${typeof hyp.score === 'number' && hyp.score > 0.6 ? 'high-confidence' : ''}`}>
              <div className="hypothesis-head">
                <span className="hyp-class">class {hyp.class}</span>
                {typeof hyp.score === 'number' && (
                  <span className="hyp-score">score {g2(hyp.score)}</span>
                )}
              </div>
              <p className="candidate-text">{hyp.candidate_trigger}</p>
              <p className="reasoning">{hyp.reasoning}</p>
              {Array.isArray(hyp.source_samples) && hyp.source_samples.length > 0 && (
                <p className="source-samples">
                  Seen in rows: {hyp.source_samples.join(', ')}
                </p>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Stage 3B */}
      <section className="stage-section evidence-samples">
        <h2>Stage 3: Evidence Samples</h2>
        <div className={`samples-list ${expandedSamples ? 'expanded' : ''}`}>
          {displayedSamples.map((sample, i) => (
            <div key={sample.index ?? i} className="sample-item">
              <p className="sample-index">Row #{sample.index}</p>
              <p className="sample-text">
                {highlightTrigger(sample.text, confirmedTrigger)}
              </p>
            </div>
          ))}
        </div>
        {hasMoreSamples && (
          <div className="samples-show-more-container">
            <button
              className="samples-show-more-btn"
              onClick={() => setExpandedSamples(!expandedSamples)}
            >
              {expandedSamples
                ? 'Show less'
                : `Show more (${samples.length - 5} more)`}
            </button>
          </div>
        )}
      </section>

      {/* Stage 4 */}
      <section className="stage-section">
        <h2>Stage 4: Statistical Confirmation</h2>
        <ZScoreChart confirmation={confirmation} />

        {view.confirmation_note && (
          <div className="confirmation-note">
            <p>{view.confirmation_note}</p>
          </div>
        )}
      </section>
    </div>
  )
}

export default TabHowWeFoundIt
