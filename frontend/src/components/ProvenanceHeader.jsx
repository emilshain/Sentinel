import './ProvenanceHeader.css'
import { g2 } from '../format'

// Persistent provenance bar shown across both tabs so run context
// (where the data came from, scope, runtime, generator) is always visible.
function ProvenanceHeader({ provenance, modelName }) {
  if (!provenance) return null

  const {
    data_source,
    fallback_reason,
    dataset_scope,
    dataset_samples,
    runtime_seconds,
    hypothesis_generator,
    hypothesis_is_mock,
  } = provenance

  const isLive = data_source === 'live_run'
  const runtime =
    typeof runtime_seconds === 'number' ? `${g2(runtime_seconds)}s` : '—'

  const scopeLabel =
    dataset_scope
      ? dataset_scope.replace(/_/g, ' ')
      : dataset_samples != null
        ? `${dataset_samples} rows`
        : '—'

  return (
    <div className="provenance-bar">
      <div className={`prov-badge ${isLive ? 'prov-live' : 'prov-cached'}`}>
        <span className="prov-dot" />
        {data_source ? data_source.replace(/_/g, ' ') : 'unknown source'}
      </div>

      {modelName && (
        <div className="prov-item">
          <span className="prov-label">Model</span>
          <span className="prov-value prov-model">{modelName}</span>
        </div>
      )}

      <div className="prov-item">
        <span className="prov-label">Scope</span>
        <span className="prov-value">{scopeLabel}</span>
      </div>

      <div className="prov-item">
        <span className="prov-label">Runtime</span>
        <span className="prov-value">{runtime}</span>
      </div>

      <div className="prov-item">
        <span className="prov-label">Hypothesis generator</span>
        <span className="prov-value">
          {hypothesis_generator
            ? hypothesis_generator.replace(/_/g, ' ')
            : '—'}
        </span>
      </div>

      {hypothesis_is_mock && (
        <div className="prov-badge prov-mock" title="Hypotheses came from mock data, not a live model call">
          MOCK DATA
        </div>
      )}

      {/* Why this is not a live run. Stating the reason is what makes the
          fallback honest degradation rather than a silent substitution. */}
      {!isLive && fallback_reason && (
        <div className="prov-reason" title={fallback_reason}>
          <span className="prov-label">Why not live</span>
          <span className="prov-value prov-reason-text">{fallback_reason}</span>
        </div>
      )}
    </div>
  )
}

export default ProvenanceHeader
