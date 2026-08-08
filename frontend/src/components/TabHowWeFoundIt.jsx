import './TabHowWeFoundIt.css'

function TabHowWeFoundIt({ view }) {
  return (
    <div className="tab-how">
      <section className="stage-section">
        <h2>Stage 1: Discovery</h2>
        <p className="stage-description">Four detectors, each with verdict and reasoning</p>

        {view.stage_1_discovery && (
          <div className="detectors-grid">
            {Object.entries(view.stage_1_discovery).map(([detector, data]) => (
              <div key={detector} className="detector-card">
                <h3>{detector.replace(/_/g, ' ')}</h3>
                <div className={`verdict ${data.verdict.toLowerCase().replace(/_/g, '-')}`}>
                  {data.verdict}
                </div>
                <p className="blurb">{data.blurb}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="stage-section">
        <h2>Stage 2: Word-Level Evidence</h2>
        <p className="stage-description">Flagged words and counts from all samples</p>

        {view.stage_2_evidence && (
          <div className="evidence-table">
            <table>
              <thead>
                <tr>
                  <th>Word / Token</th>
                  <th>Count</th>
                  <th>Frequency</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(view.stage_2_evidence)
                  .slice(0, 15)
                  .map(([word, count]) => (
                    <tr key={word}>
                      <td className="word-token">{word}</td>
                      <td>{count}</td>
                      <td>
                        <div className="frequency-bar">
                          <div
                            className="frequency-fill"
                            style={{ width: `${Math.min(100, (count / 10) * 100)}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {Object.keys(view.stage_2_evidence).length > 15 && (
              <p className="show-more">... and {Object.keys(view.stage_2_evidence).length - 15} more</p>
            )}
          </div>
        )}
      </section>

      <section className="stage-section">
        <h2>Stage 3: Reasoning & Hypotheses</h2>
        <p className="stage-description">Model's reasoning about repeated phrases in flagged samples</p>

        {view.stage_3_hypotheses && (
          <div className="hypotheses-grid">
            {view.stage_3_hypotheses.map((hyp, i) => (
              <div key={i} className="hypothesis-card">
                <h3>Candidate {i + 1}</h3>
                <p className="candidate-text">{hyp.candidate}</p>
                <p className="reasoning">{hyp.reasoning}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="stage-section evidence-samples">
        <h2>Stage 3: Evidence Samples</h2>
        <p className="stage-description">Raw sample texts with the repeated span highlighted</p>

        {view.stage_3_evidence_samples && (
          <div className="samples-list">
            {view.stage_3_evidence_samples.map((sample, i) => (
              <div key={i} className="sample-item">
                <p className="sample-index">Sample {i + 1}</p>
                <p className="sample-text">{sample}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="stage-section">
        <h2>Stage 4: Confirmation</h2>
        <p className="stage-description">Z-score confirmation against threshold</p>

        {view.stage_4_confirmation && (
          <div className="confirmation-grid">
            {Object.entries(view.stage_4_confirmation).map(([candidate, scores]) => (
              <div key={candidate} className="confirmation-card">
                <h3>{candidate}</h3>
                <div className="score-item">
                  <label>Scanner Z-Score</label>
                  <p className={`score ${Math.abs(scores.scanner_z) > 2 ? 'confirmed' : 'rejected'}`}>
                    {scores.scanner_z.toFixed(2)}
                  </p>
                </div>
                <div className="score-item">
                  <label>Strip Z-Score</label>
                  <p className={`score ${Math.abs(scores.strip_z) > 2 ? 'confirmed' : 'rejected'}`}>
                    {scores.strip_z.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

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
