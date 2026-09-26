export const API_BASE = '/api'

export const ALLOWED_LABELS = [
  'crazing',
  'inclusion',
  'patches',
  'pitted_surface',
  'rolled-in_scale',
  'scratches',
] as const

export interface Probabilities {
  [label: string]: number
}

export interface Prediction {
  predicted_label: string
  predicted_index: number
  confidence: number
  confidence_tier: 'high' | 'medium' | 'low'
  probabilities: Probabilities
  needs_review: boolean
  evidence_coverage: number
  evidence_bbox: { x: number; y: number; w: number; h: number } | null
  severity: 'minor' | 'major' | 'critical'
  severity_score: number
  severity_components: Record<string, number>
  recommended_action: string
  latency_ms: number
  model_version: string
}

export interface Feedback {
  id: number
  inspection_id: number
  created_at: string
  is_correct: boolean
  corrected_label: string | null
  notes: string | null
  inspector_name: string | null
  model_version_at_feedback: string
  review_status: 'pending' | 'approved' | 'rejected'
}

export interface Inspection {
  id: number
  created_at: string
  original_filename: string
  image_width: number
  image_height: number
  image_url: string | null
  overlay_url: string | null
  prediction: Prediction | null
  feedback: Feedback | null
}

export interface InspectionPage {
  total: number
  page: number
  page_size: number
  items: Inspection[]
}

export interface Analytics {
  total_inspections: number
  total_with_feedback: number
  defect_rate: number | null
  severity_distribution: Record<string, number>
  label_distribution: Record<string, number>
  confidence_tier_distribution: Record<string, number>
  confidence_histogram: { bin_start: number; bin_end: number; count: number }[]
  correction_rate: number | null
  corrections_by_true_label: Record<string, number>
  mean_latency_ms: number | null
  per_day_counts: { date: string; count: number }[]
  recent_inspections: Inspection[]
}

export interface ModelInfo {
  model_version: string
  experiment: string
  arch: string
  classes: string[]
  image_size: number
  training_summary: {
    epochs_run: number | null
    best_epoch: number | null
    best_val_f1_macro: number | null
    train_seconds: number | null
    environment: Record<string, unknown>
  }
  test_metrics: Record<string, unknown>
  model_size_mb: number
  confidence_policy: { high_min: number; medium_min: number; needs_review_below: number }
  severity_policy: Record<string, unknown>
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* ignore parse errors */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export async function createInspection(file: File): Promise<Inspection> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/inspections`, { method: 'POST', body: form })
  return handle<Inspection>(res)
}

export async function listInspections(params: Record<string, string | number | boolean | undefined>): Promise<InspectionPage> {
  const q = new URLSearchParams()
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== '') q.set(k, String(v))
  })
  const res = await fetch(`${API_BASE}/inspections?${q}`)
  return handle<InspectionPage>(res)
}

export async function getInspection(id: number): Promise<Inspection> {
  const res = await fetch(`${API_BASE}/inspections/${id}`)
  return handle<Inspection>(res)
}

export async function submitFeedback(
  id: number,
  payload: { is_correct: boolean; corrected_label?: string | null; notes?: string | null; inspector_name?: string | null },
): Promise<Feedback> {
  const res = await fetch(`${API_BASE}/inspections/${id}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handle<Feedback>(res)
}

export async function getAnalytics(): Promise<Analytics> {
  const res = await fetch(`${API_BASE}/analytics`)
  return handle<Analytics>(res)
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model/info`)
  return handle<ModelInfo>(res)
}

export async function getHealth(): Promise<{ status: string; database: boolean; model_ready: boolean; model_version: string | null }> {
  const res = await fetch(`${API_BASE}/health`)
  return handle(res)
}
