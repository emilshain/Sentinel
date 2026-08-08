import { useEffect, useState } from 'react'
import './ScanningScreen.css'

// Cosmetic staged progress that mirrors the real pipeline's four stages.
// Frontend-only: no scan actually runs here — this paces the reveal of the
// recorded report so the flow reads like a live scan.
const STEPS = [
  'Loading model & dataset',
  'Running 4 detectors',
  'Generating trigger hypotheses',
  'Statistical confirmation',
]

const STEP_MS = 800

function ScanningScreen({ modelName, onComplete, error, onCancel, statusMessage }) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (error) return undefined
    if (step >= STEPS.length) {
      onComplete?.()
      return undefined
    }
    const t = setTimeout(() => setStep((s) => s + 1), STEP_MS)
    return () => clearTimeout(t)
  }, [step, error, onComplete])

  if (error) {
    return (
      <div className="scanning-screen">
        <div className="scan-error">
          <h2>Scan failed</h2>
          <p>{error}</p>
          <button className="primary-button" onClick={onCancel}>
            Back
          </button>
        </div>
      </div>
    )
  }

  const pct = Math.min(100, Math.round((step / STEPS.length) * 100))

  return (
    <div className="scanning-screen">
      <div className="spinner" />
      <h2 className="scan-title">Scanning {modelName ? <span>{modelName}</span> : 'model'}</h2>

      <div className="scan-progress-bar">
        <div className="scan-progress-fill" style={{ width: `${pct}%` }} />
      </div>

      {/* The staged list above is cosmetic and finishes in ~3s; this line is the
          real backend state, which is what the user is actually waiting on. */}
      {statusMessage && <p className="scan-status">{statusMessage}</p>}

      <ul className="scan-steps">
        {STEPS.map((label, i) => {
          const state = i < step ? 'done' : i === step ? 'active' : 'pending'
          return (
            <li key={label} className={`scan-step ${state}`}>
              <span className="step-marker">
                {state === 'done' ? '✓' : i + 1}
              </span>
              {label}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

export default ScanningScreen
