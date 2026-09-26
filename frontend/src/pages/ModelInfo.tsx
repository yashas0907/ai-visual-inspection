import { useEffect, useState } from 'react'
import { getModelInfo, type ModelInfo } from '../api/client'
import { ErrorBox, Spinner } from '../components/ui'

function MetricsTable({ metrics }: { metrics: Record<string, unknown> }) {
  const skip = new Set([
    'per_class',
    'confusion_matrix',
    'confusion_matrix_classes',
    'images_measured',
  ])
  const rows = Object.entries(metrics).filter(([k]) => !skip.has(k))
  return (
    <table>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ width: '45%' }}>{k.replace(/_/g, ' ')}</td>
            <td className="mono">{String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function PerClassTable({ metrics }: { metrics: Record<string, unknown> }) {
  const perClass = metrics.per_class as
    | { class: string; precision: number; recall: number; f1: number }[]
    | undefined
  if (!perClass) return null
  return (
    <table style={{ marginTop: 10 }}>
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        {perClass.map((c) => (
          <tr key={c.class}>
            <td>{c.class.replace('_', ' ')}</td>
            <td className="mono">{c.precision.toFixed(3)}</td>
            <td className="mono">{c.recall.toFixed(3)}</td>
            <td className="mono">{c.f1.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export default function ModelInfoPage() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError((e as Error).message))
  }, [])

  if (error) return <ErrorBox>{error}</ErrorBox>
  if (!info) return <Spinner label="Loading model card…" />

  const hasMetrics = Object.keys(info.test_metrics).length > 0
  const env = info.training_summary.environment as Record<string, string> | undefined

  return (
    <div>
      <div className="page-header">
        <h2>Model Information</h2>
        <p>Active artifact, training provenance and genuine evaluation metrics.</p>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Artifact</h3>
          <dl className="kv">
            <dt>Model version</dt>
            <dd><code>{info.model_version}</code></dd>
            <dt>Experiment</dt>
            <dd><code>{info.experiment}</code></dd>
            <dt>Architecture</dt>
            <dd>{info.arch} (transfer learning, ImageNet init, 1-channel stem)</dd>
            <dt>Input</dt>
            <dd>{info.image_size}×{info.image_size} grayscale, normalized</dd>
            <dt>Parameters</dt>
            <dd>~11.2M (fp32, {info.model_size_mb || '—'} MB)</dd>
            <dt>Classes</dt>
            <dd>{info.classes.map((c) => c.replace('_', ' ')).join(', ')}</dd>
          </dl>
        </div>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Training summary</h3>
          <dl className="kv">
            <dt>Epochs run</dt>
            <dd>{info.training_summary.epochs_run ?? '—'}</dd>
            <dt>Best epoch</dt>
            <dd>{info.training_summary.best_epoch ?? '—'}</dd>
            <dt>Best val macro-F1</dt>
            <dd>{info.training_summary.best_val_f1_macro ?? '—'}</dd>
            <dt>Training time</dt>
            <dd>{info.training_summary.train_seconds ? `${info.training_summary.train_seconds}s (CPU)` : '—'}</dd>
            <dt>Environment</dt>
            <dd className="small">
              {env ? `python ${env.python ?? '—'} · torch ${env.torch ?? '—'} · ${env.device ?? '—'}` : '—'}
            </dd>
          </dl>
          <h3 style={{ fontSize: 15, margin: '16px 0 8px' }}>Confidence policy</h3>
          <dl className="kv">
            <dt>High tier</dt>
            <dd>≥ {(info.confidence_policy.high_min * 100).toFixed(0)}%</dd>
            <dt>Medium tier</dt>
            <dd>{(info.confidence_policy.medium_min * 100).toFixed(0)}% – {((info.confidence_policy.high_min * 100) - 1).toFixed(0)}%</dd>
            <dt>Low / review</dt>
            <dd>&lt; {(info.confidence_policy.medium_min * 100).toFixed(0)}%</dd>
          </dl>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: 15, marginBottom: 10 }}>Test-set metrics (held-out 144 images)</h3>
        {hasMetrics ? (
          <>
            <MetricsTable metrics={info.test_metrics} />
            <PerClassTable metrics={info.test_metrics} />
            <p className="small muted" style={{ marginTop: 10 }}>
              Produced by <code>ml/evaluation/evaluate.py</code> on the untouched test split. See
              the confusion matrix artifact in the experiment directory.
            </p>
          </>
        ) : (
          <p className="muted">
            Evaluation has not been run for this experiment yet. Run{' '}
            <code>python ml/evaluation/evaluate.py --experiment {info.experiment}</code> — metrics
            will appear here automatically once produced.
          </p>
        )}
      </div>

      <div className="card">
        <h3 style={{ fontSize: 15, marginBottom: 10 }}>Severity rule rubric</h3>
        <p className="small" style={{ marginBottom: 10 }}>
          Severity is a business-layer decision computed from three transparent inputs. Points sum
          to a score: ≤3 minor, 4–5 major, ≥6 critical.
        </p>
        <table>
          <thead>
            <tr>
              <th>Component</th>
              <th>Values</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Defect risk</td>
              <td className="small">
                patches 1 · rolled-in scale / scratches 2 · crazing / pitted surface / inclusion 3
              </td>
            </tr>
            <tr>
              <td>Confidence tier</td>
              <td className="small">high +0 · medium +1 · low +2</td>
            </tr>
            <tr>
              <td>Evidence coverage</td>
              <td className="small">≤10% of surface +0 · 10–30% +1 · &gt;30% +2</td>
            </tr>
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 10 }}>
          Configurable in <code>backend/app/services/severity.py</code>; unit-tested in{' '}
          <code>backend/tests/test_severity_confidence.py</code>.
        </p>
      </div>
    </div>
  )
}
