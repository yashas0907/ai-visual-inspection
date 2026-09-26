import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import {
  getInspection,
  submitFeedback,
  type Inspection,
  ALLOWED_LABELS,
} from '../api/client'
import {
  ConfidenceBar,
  ErrorBox,
  Probabilities,
  SeverityBadge,
  SeverityBreakdown,
  Spinner,
  TierBadge,
} from '../components/ui'

function FeedbackPanel({ inspection, onDone }: { inspection: Inspection; onDone: () => void }) {
  const [mode, setMode] = useState<'idle' | 'correct' | 'incorrect'>(inspection.feedback ? 'idle' : 'idle')
  const [label, setLabel] = useState('')
  const [notes, setNotes] = useState('')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  if (inspection.feedback) {
    const fb = inspection.feedback
    return (
      <div className="feedback-panel">
        <h3 style={{ fontSize: 14, marginBottom: 8 }}>Inspector feedback</h3>
        <dl className="kv">
          <dt>Verdict</dt>
          <dd>{fb.is_correct ? 'Prediction confirmed' : `Corrected to "${fb.corrected_label}"`}</dd>
          <dt>Inspector</dt>
          <dd>{fb.inspector_name ?? '—'}</dd>
          <dt>Notes</dt>
          <dd>{fb.notes ?? '—'}</dd>
          <dt>Review status</dt>
          <dd>
            <span className={`badge ${fb.review_status}`}>{fb.review_status}</span>
          </dd>
        </dl>
        <p className="small muted" style={{ marginTop: 10 }}>
          Feedback is stored with the model version and must be approved by a maintainer before it can be used as training data.
        </p>
      </div>
    )
  }

  async function send(isCorrect: boolean, corrected?: string) {
    setBusy(true)
    setError('')
    try {
      await submitFeedback(inspection.id, {
        is_correct: isCorrect,
        corrected_label: corrected ?? null,
        notes: notes || null,
        inspector_name: name || null,
      })
      onDone()
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  return (
    <div className="feedback-panel">
      <h3 style={{ fontSize: 14, marginBottom: 8 }}>Inspector review</h3>
      {error && <ErrorBox>{error}</ErrorBox>}
      <div className="feedback-controls" style={{ marginTop: 10 }}>
        {mode === 'idle' ? (
          <>
            <button onClick={() => setMode('correct')} disabled={busy}>
              Prediction is correct
            </button>
            <button className="danger" onClick={() => setMode('incorrect')} disabled={busy}>
              Prediction is incorrect
            </button>
          </>
        ) : mode === 'correct' ? (
          <>
            <span className="small">Confirm that <strong>{inspection.prediction?.predicted_label}</strong> is the correct defect type?</span>
            <button onClick={() => send(true)} disabled={busy}>
              Confirm
            </button>
            <button className="secondary" onClick={() => setMode('idle')} disabled={busy}>
              Cancel
            </button>
          </>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%' }}>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <select value={label} onChange={(e) => setLabel(e.target.value)} defaultValue="">
                <option value="" disabled>
                  Select correct defect type…
                </option>
                {ALLOWED_LABELS.map((l) => (
                  <option key={l} value={l}>
                    {l}
                  </option>
                ))}
              </select>
              <button onClick={() => send(false, label)} disabled={busy || !label}>
                Submit correction
              </button>
              <button className="secondary" onClick={() => setMode('idle')} disabled={busy}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 10, marginTop: 12, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Inspector name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          style={{ width: 200 }}
        />
        <input
          type="text"
          placeholder="Notes (optional)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          style={{ flex: 1, minWidth: 220 }}
        />
      </div>
    </div>
  )
}

export default function Result() {
  const { id } = useParams()
  const [inspection, setInspection] = useState<Inspection | null>(null)
  const [error, setError] = useState('')

  async function load() {
    try {
      setInspection(await getInspection(Number(id)))
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  if (error) return <ErrorBox>{error}</ErrorBox>
  if (!inspection || !inspection.prediction) return <Spinner label="Loading inspection…" />

  const p = inspection.prediction
  const coveragePct = (p.evidence_coverage * 100).toFixed(1)

  return (
    <div>
      <div className="page-header">
        <h2>Inspection #{inspection.id}</h2>
        <p className="muted">
          {inspection.original_filename} — {inspection.image_width}×{inspection.image_height} —{' '}
          {new Date(inspection.created_at).toLocaleString()} — model <code>{p.model_version}</code>
        </p>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="image-compare">
            <figure>
              <img className="inspection-img" src={inspection.image_url ?? ''} alt="original" />
              <figcaption>Original (sanitized grayscale)</figcaption>
            </figure>
            <figure>
              <img className="inspection-img" src={inspection.overlay_url ?? ''} alt="explainability overlay" />
              <figcaption>Grad-CAM evidence overlay</figcaption>
            </figure>
          </div>
          {p.evidence_bbox && (
            <p className="small muted" style={{ marginTop: 10 }}>
              Evidence region (approx.): x={p.evidence_bbox.x}, y={p.evidence_bbox.y}, w={p.evidence_bbox.w}, h={p.evidence_bbox.h}
            </p>
          )}
        </div>

        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <h3 style={{ fontSize: 16 }}>
              {p.predicted_label.replace('_', ' ')}
            </h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <SeverityBadge severity={p.severity} />
              <TierBadge tier={p.confidence_tier} />
            </div>
          </div>

          <ConfidenceBar value={p.confidence} />

          <p className="small muted" style={{ margin: '14px 0 8px' }}>
            Class probabilities (softmax — not calibrated probabilities):
          </p>
          <Probabilities prediction={p} />

          <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '14px 0' }} />

          <h3 style={{ fontSize: 14, marginBottom: 8 }}>Severity derivation</h3>
          <SeverityBreakdown prediction={p} />
          <p className="small muted" style={{ marginTop: 10 }}>
            Severity is computed by a deterministic rule engine from defect risk, confidence tier and
            evidence coverage ({coveragePct}% of surface). See the model page for the full rubric.
          </p>

          {p.needs_review && (
            <div className="error-box" style={{ marginTop: 12 }}>
              Low-confidence prediction — human review strongly recommended.
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <FeedbackPanel inspection={inspection} onDone={load} />
        <p className="small muted" style={{ marginTop: 12 }}>
          Inference latency: {p.latency_ms.toFixed(1)} ms (CPU).
        </p>
      </div>
    </div>
  )
}
