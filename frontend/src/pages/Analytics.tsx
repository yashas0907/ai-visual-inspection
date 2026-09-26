import { useEffect, useState } from 'react'
import { getAnalytics, type Analytics } from '../api/client'
import { ErrorBox, Spinner } from '../components/ui'

function DistributionBars({ data, colorMap }: { data: Record<string, number>; colorMap?: Record<string, string> }) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const max = Math.max(1, ...entries.map(([, v]) => v))
  if (entries.length === 0) return <p className="muted small">No data yet.</p>
  return (
    <div className="bar-chart">
      {entries.map(([k, v]) => (
        <div className="bar-row" key={k}>
          <span>{k.replace('_', ' ')}</span>
          <div className="bar-track">
            <div
              className={`bar-fill ${colorMap?.[k] ?? ''}`}
              style={{ width: `${(v / max) * 100}%` }}
            />
          </div>
          <span className="num">{v}</span>
        </div>
      ))}
    </div>
  )
}

function ConfidenceHistogram({ hist }: { hist: Analytics['confidence_histogram'] }) {
  if (hist.length === 0) return <p className="muted small">No predictions yet.</p>
  const max = Math.max(...hist.map((b) => b.count))
  return (
    <>
      <div className="hist">
        {hist.map((b) => (
          <div
            key={b.bin_start}
            className="bin"
            style={{ height: `${Math.max(3, (b.count / max) * 100)}%` }}
            title={`${(b.bin_start * 100).toFixed(0)}–${(b.bin_end * 100).toFixed(0)}%: ${b.count}`}
          />
        ))}
      </div>
      <div className="hist-labels">
        {hist.map((b) => (
          <span key={b.bin_start}>{(b.bin_start * 100).toFixed(0)}</span>
        ))}
      </div>
    </>
  )
}

export default function Analytics() {
  const [data, setData] = useState<Analytics | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getAnalytics()
      .then(setData)
      .catch((e) => setError((e as Error).message))
  }, [])

  if (error) return <ErrorBox>{error}</ErrorBox>
  if (!data) return <Spinner label="Computing analytics…" />

  const sevColors: Record<string, string> = { minor: 'ok', major: 'warn', critical: 'danger' }

  return (
    <div>
      <div className="page-header">
        <h2>Analytics</h2>
        <p>All charts are computed from stored inspection rows — nothing is simulated.</p>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Defect distribution (model predictions)</h3>
          <DistributionBars data={data.label_distribution} />
        </div>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Severity distribution</h3>
          <DistributionBars data={data.severity_distribution} colorMap={sevColors} />
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Model confidence distribution</h3>
          <ConfidenceHistogram hist={data.confidence_histogram} />
          <p className="small muted" style={{ marginTop: 8 }}>
            Softmax confidence per inspection. Tiers: ≥80% high, 60–79% medium, &lt;60% low.
          </p>
        </div>
        <div className="card">
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Confidence tiers</h3>
          <DistributionBars data={data.confidence_tier_distribution} />
          <h3 style={{ fontSize: 15, margin: '18px 0 12px' }}>Human corrections by true label</h3>
          {Object.keys(data.corrections_by_true_label).length === 0 ? (
            <p className="muted small">No corrective feedback submitted yet.</p>
          ) : (
            <DistributionBars data={data.corrections_by_true_label} />
          )}
        </div>
      </div>

      <div className="grid cols-3">
        <div className="card">
          <div className="stat-label">Total inspections</div>
          <div className="stat-value">{data.total_inspections}</div>
        </div>
        <div className="card">
          <div className="stat-label">Human correction rate</div>
          <div className="stat-value">
            {data.correction_rate === null ? '—' : `${(data.correction_rate * 100).toFixed(1)}%`}
          </div>
          <div className="stat-sub">{data.total_with_feedback} reviewed</div>
        </div>
        <div className="card">
          <div className="stat-label">Mean latency</div>
          <div className="stat-value">
            {data.mean_latency_ms === null ? '—' : `${data.mean_latency_ms.toFixed(0)} ms`}
          </div>
          <div className="stat-sub">CPU per image</div>
        </div>
      </div>

      {data.per_day_counts.length > 1 && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3 style={{ fontSize: 15, marginBottom: 12 }}>Inspections per day</h3>
          <DistributionBars
            data={Object.fromEntries(data.per_day_counts.map((d) => [d.date, d.count]))}
          />
        </div>
      )}
    </div>
  )
}
