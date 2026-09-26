import { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { createInspection } from '../api/client'
import { ErrorBox, Spinner } from '../components/ui'

const MAX_MB = 10

export default function Inspect() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  function pick(f: File | null | undefined) {
    setError('')
    if (!f) return
    if (!/\.(jpe?g|png|bmp|webp)$/i.test(f.name)) {
      setError(`Unsupported file type "${f.name.split('.').pop()}". Allowed: jpg, jpeg, png, bmp, webp.`)
      return
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`File is ${(f.size / 1e6).toFixed(1)} MB; limit is ${MAX_MB} MB.`)
      return
    }
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  async function run() {
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const inspection = await createInspection(file)
      navigate(`/inspections/${inspection.id}`)
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <h2>New Inspection</h2>
        <p>Upload a surface image of a hot-rolled steel strip. The model classifies the defect type, localizes the evidence and derives a severity.</p>
      </div>

      {error && <div style={{ marginBottom: 14 }}><ErrorBox>{error}</ErrorBox></div>}

      {!preview ? (
        <div
          className={`dropzone ${dragOver ? 'dragover' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragOver(false)
            pick(e.dataTransfer.files?.[0])
          }}
          onClick={() => inputRef.current?.click()}
        >
          <h3>Drop a surface image here</h3>
          <p>or click to browse</p>
          <p className="hint">jpg / jpeg / png / bmp / webp — up to {MAX_MB} MB — grayscale steel surface (200×200 or larger)</p>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.webp"
            hidden
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>
      ) : (
        <div className="grid cols-2">
          <div className="card">
            <img className="inspection-img" src={preview} alt="upload preview" />
            <p className="small muted" style={{ marginTop: 8 }}>
              {file?.name} — {file ? (file.size / 1024).toFixed(0) : 0} KB
            </p>
          </div>
          <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 14 }}>
            <p className="small">
              The inspection pipeline will:
            </p>
            <ol className="small muted" style={{ paddingLeft: 20, lineHeight: 1.9 }}>
              <li>Validate and sanitize the upload</li>
              <li>Classify the defect (6 NEU defect classes)</li>
              <li>Generate a Grad-CAM evidence heatmap</li>
              <li>Apply the confidence-tier policy</li>
              <li>Compute severity via transparent business rules</li>
            </ol>
            <div style={{ display: 'flex', gap: 10 }}>
              <button onClick={run} disabled={busy}>
                {busy ? 'Inspecting…' : 'Run inspection'}
              </button>
              <button
                className="secondary"
                onClick={() => {
                  setFile(null)
                  setPreview(null)
                }}
                disabled={busy}
              >
                Choose another
              </button>
            </div>
            {busy && <Spinner label="Model inference + explainability…" />}
          </div>
        </div>
      )}
    </div>
  )
}
