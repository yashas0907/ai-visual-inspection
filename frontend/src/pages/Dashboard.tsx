import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getAnalytics, getHealth, type Analytics } from '../api/client'

function fmtPct(v: number | null): string {
  return v === null ? '—' : `${(v * 100).toFixed(1)}%`
}

export default function Dashboard() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [health, setHealth] = useState<{ status: string; model_ready: boolean; database: boolean; model_version: string | null } | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getAnalytics(), getHealth()])
      .then(([a, h]) => {
        setAnalytics(a)
        setHealth(h)
      })
      .catch((e) => setError(String(e.message)))
  }, [])

  if (error) return <div className="error-box">{error}</div>
  if (!analytics) return <div className="spinner" />

  const sev = analytics.severity_distribution
  const maxSev = Math.max(1, ...Object.values(sev))

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Hot-rolled steel strip surface inspection — live platform status and throughput.</p>
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <div className="card">
          <div className="stat-label">Total inspections</div>
          <div className="stat-value">{analytics.total_inspections}</div>
        </div>
        <div className="card">
          <div className="stat-label">Flagged defect rate</div>
          <div className="stat-value">{fmtPct(analytics.defect_rate)}</div>
          <div className="stat-sub">≥ medium-confidence predictions</div>
        </div>
        <div className="card">
          <div className="stat-label">Human correction rate</div>
          <div className="stat-value">{fmtPct(analytics.correction_rate)}</div>
          <div className="stat-sub">{analytics.total_with_feedback} inspections reviewed</div>
        </div>
        <div className="card">
          <div className="stat-label">Mean latency</div>
          <div className="stat-value">
            {analytics.mean_latency_ms === null ? '—' : `${analytics.mean_latency_ms.toFixed(0)} ms`}
          </div>
          <div className="stat-sub">CPU, per image</div>
        </div>
      </div>

      <div className="grid cols-2" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3 style={{ marginBottom: 12, fontSize: 15 }}>Severity distribution</h3>
          <div className="bar-chart">
            {(['minor', 'major', 'critical'] as const).map((s) => (
              <div className="bar-row" key={s}>
                <span>{s}</span>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${s === 'minor' ? 'ok' : s === 'major' ? 'warn' : 'danger'}`}
                    style={{ width: `${((sev[s] ?? 0) / maxSev) * 100}%` }}
                  />
                </div>
                <span className="num">{sev[s] ?? 0}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 12, fontSize: 15 }}>System health</h3>
          <dl className="kv">
            <dt>API status</dt>
            <dd>{health?.status ?? '—'}</dd>
            <dt>Model</dt>
            <dd>{health?.model_ready ? `ready (${health.model_version})` : 'not loaded'}</dd>
            <dt>Database</dt>
            <dd>{health?.database ? 'connected' : 'unavailable'}</dd>
          </dl>
          <p className="small muted" style={{ marginTop: 12 }}>
            Model not trained yet? See the <Link to="/model">model page</Link> and
            the training workflow in the README.
          </p>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 10, fontSize: 15 }}>Recent inspections</h3>
        {analytics.recent_inspections.length === 0 ? (
          <p className="muted">No inspections yet. Run one from the inspection page.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Filename</th>
                <th>Defect</th>
                <th>Confidence</th>
                <th>Severity</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {analytics.recent_inspections.map((i) => (
                <tr className="clickable" key={i.id} onClick={() => (window.location.href = `/inspections/${i.id}`)}>
                  <td className="mono">#{i.id}</td>
                  <td>{i.original_filename}</td>
                  <td>{i.prediction?.predicted_label ?? '—'}</td>
                  <td>{i.prediction ? `${(i.prediction.confidence * 100).toFixed(0)}%` : '—'}</td>
                  <td>{i.prediction && <span className={`badge ${i.prediction.severity}`}>{i.prediction.severity}</span>}</td>
                  <td>
                    {i.feedback ? (
                      <span className={`badge ${i.feedback.review_status}`}>{i.feedback.is_correct ? 'confirmed' : 'corrected'}</span>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
