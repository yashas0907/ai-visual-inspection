import { useEffect, useState } from 'react'
import { listInspections, type InspectionPage } from '../api/client'
import { ErrorBox, Spinner } from '../components/ui'

const PAGE_SIZE = 20

export default function History() {
  const [data, setData] = useState<InspectionPage | null>(null)
  const [page, setPage] = useState(1)
  const [label, setLabel] = useState('')
  const [severity, setSeverity] = useState('')
  const [tier, setTier] = useState('')
  const [feedbackFilter, setFeedbackFilter] = useState('')
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    listInspections({
      page,
      page_size: PAGE_SIZE,
      label: label || undefined,
      severity: severity || undefined,
      tier: tier || undefined,
      has_feedback: feedbackFilter === '' ? undefined : feedbackFilter === 'yes',
      search: search || undefined,
    })
      .then(setData)
      .catch((e) => setError((e as Error).message))
  }, [page, label, severity, tier, feedbackFilter, search])

  if (error) return <ErrorBox>{error}</ErrorBox>
  if (!data) return <Spinner label="Loading history…" />

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE))

  return (
    <div>
      <div className="page-header">
        <h2>Inspection History</h2>
        <p>All stored inspections with predictions, severity and inspector feedback.</p>
      </div>

      <div className="filter-bar">
        <select value={label} onChange={(e) => { setLabel(e.target.value); setPage(1) }}>
          <option value="">All defect types</option>
          {['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches'].map((l) => (
            <option key={l}>{l}</option>
          ))}
        </select>
        <select value={severity} onChange={(e) => { setSeverity(e.target.value); setPage(1) }}>
          <option value="">All severities</option>
          <option>minor</option>
          <option>major</option>
          <option>critical</option>
        </select>
        <select value={tier} onChange={(e) => { setTier(e.target.value); setPage(1) }}>
          <option value="">All confidences</option>
          <option value="high">high</option>
          <option value="medium">medium</option>
          <option value="low">low</option>
        </select>
        <select value={feedbackFilter} onChange={(e) => { setFeedbackFilter(e.target.value); setPage(1) }}>
          <option value="">Any feedback state</option>
          <option value="yes">Has feedback</option>
          <option value="no">Awaiting review</option>
        </select>
        <input
          type="text"
          placeholder="Search filename…"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          style={{ width: 180 }}
        />
      </div>

      <div className="card" style={{ padding: 6 }}>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Thumb</th>
              <th>File</th>
              <th>Defect</th>
              <th>Conf.</th>
              <th>Severity</th>
              <th>Feedback</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {data.items.length === 0 && (
              <tr>
                <td colSpan={8} className="muted" style={{ textAlign: 'center', padding: 30 }}>
                  No inspections match the filters.
                </td>
              </tr>
            )}
            {data.items.map((i) => (
              <tr className="clickable" key={i.id} onClick={() => (window.location.href = `/inspections/${i.id}`)}>
                <td className="mono">#{i.id}</td>
                <td>
                  {i.image_url && (
                    <img src={i.image_url} alt="" style={{ width: 44, height: 44, objectFit: 'cover', borderRadius: 4, border: '1px solid var(--border)' }} />
                  )}
                </td>
                <td style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {i.original_filename}
                </td>
                <td>{i.prediction?.predicted_label ?? '—'}</td>
                <td>{i.prediction ? `${(i.prediction.confidence * 100).toFixed(0)}%` : '—'}</td>
                <td>{i.prediction && <span className={`badge ${i.prediction.severity}`}>{i.prediction.severity}</span>}</td>
                <td>
                  {i.feedback ? (
                    <span className={`badge ${i.feedback.review_status}`}>
                      {i.feedback.is_correct ? 'confirmed' : 'corrected'}
                    </span>
                  ) : (
                    <span className="muted small">pending</span>
                  )}
                </td>
                <td className="small muted">{new Date(i.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button className="secondary" onClick={() => setPage(Math.max(1, page - 1))} disabled={page <= 1}>
          ← Prev
        </button>
        <span className="muted">
          Page {page} of {totalPages} — {data.total} inspections
        </span>
        <button className="secondary" onClick={() => setPage(Math.min(totalPages, page + 1))} disabled={page >= totalPages}>
          Next →
        </button>
      </div>
    </div>
  )
}
