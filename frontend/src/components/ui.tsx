import type { Prediction } from '../api/client'

import React from 'react'

export function SeverityBadge({ severity }: { severity: string }) {
  return <span className={`badge ${severity}`}>{severity}</span>
}

export function TierBadge({ tier }: { tier: string }) {
  return <span className={`badge ${tier}`}>{tier} confidence</span>
}

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const cls = value >= 0.8 ? 'ok' : value >= 0.6 ? 'warn' : 'danger'
  return (
    <div className="bar-row">
      <span>Confidence</span>
      <div className="bar-track">
        <div className={`bar-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="num">{pct}%</span>
    </div>
  )
}

export function Probabilities({ prediction }: { prediction: Prediction }) {
  const entries = Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
  return (
    <div className="prob-list">
      {entries.map(([label, p], i) => (
        <div className="prob-row" key={label}>
          <span className={i === 0 ? '' : 'muted'}>{label}</span>
          <div className="prob-track">
            <div className={`prob-fill ${i === 0 ? 'top' : ''}`} style={{ width: `${p * 100}%` }} />
          </div>
          <span className="num muted">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export function SeverityBreakdown({ prediction }: { prediction: Prediction }) {
  const names: Record<string, string> = {
    defect_risk: 'Defect type risk',
    confidence_tier: 'Confidence tier',
    evidence_coverage: 'Evidence coverage',
  }
  return (
    <dl className="kv">
      {Object.entries(prediction.severity_components).map(([k, v]) => (
        <React.Fragment key={k}>
          <dt>{names[k] ?? k}</dt>
          <dd>
            +{v} point{v === 1 ? '' : 's'}
          </dd>
        </React.Fragment>
      ))}
      <dt>Total score</dt>
      <dd>
        <strong>{prediction.severity_score}</strong>
      </dd>
      <dt>Recommended action</dt>
      <dd>{prediction.recommended_action}</dd>
    </dl>
  )
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span className="spinner" />
      {label && <span className="muted small">{label}</span>}
    </div>
  )
}

export function ErrorBox({ children }: { children: React.ReactNode }) {
  return <div className="error-box">{children}</div>
}
