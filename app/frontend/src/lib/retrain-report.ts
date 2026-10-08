import type { PersonalizableModel } from '../types'
import type { LastAttempt } from './personalization-api'

// Mirrors app/backend/src/personalization/gate.py. The backend decides; these
// only explain the decision it already stored.
export const OWN_CORRECTION_MIN = 0.8
export const MAX_GATE_DROP = 0.03
export const MAX_CUSTOM_LABEL_FRACTION = 0.1

export interface ReportCheck {
  label: string
  value: string
  caption: string
  passed: boolean
  // Fill and pass mark (both 0–1) for checks drawn as a bar.
  bar?: { fill: number; mark: number }
}

interface GateMetrics {
  own_correction_accuracy: number
  gate_accuracy: number
  base_gate_accuracy: number
  custom_label_fraction: number
}

function gateMetrics(metrics: LastAttempt['metrics']): GateMetrics | null {
  const keys = ['own_correction_accuracy', 'gate_accuracy', 'base_gate_accuracy', 'custom_label_fraction'] as const
  return keys.every((k) => typeof metrics[k] === 'number') ? (metrics as unknown as GateMetrics) : null
}

const percent = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`

// The gate's checks for one attempt, or null when the job failed before the
// gate ran (its metrics then only hold an error).
export function reportChecks(model: PersonalizableModel, attempt: LastAttempt): ReportCheck[] | null {
  const m = gateMetrics(attempt.metrics)
  if (!m) return null
  const total = attempt.correction_count
  const checks: ReportCheck[] = [
    {
      label: 'Learned your corrections',
      value: `${Math.round(m.own_correction_accuracy * total)} of ${total}`,
      caption: `Needs ${Math.ceil(OWN_CORRECTION_MIN * total - 1e-9)} of ${total}`,
      passed: m.own_correction_accuracy >= OWN_CORRECTION_MIN,
      bar: { fill: m.own_correction_accuracy, mark: OWN_CORRECTION_MIN },
    },
    {
      label: 'Still sorts other mail well',
      value: percent(m.gate_accuracy),
      caption: `${percent(m.base_gate_accuracy)} before · may drop 3 points at most`,
      passed: m.gate_accuracy >= m.base_gate_accuracy - MAX_GATE_DROP,
    },
  ]
  // Only the category model can learn labels of the user's own.
  if (model === 'category') {
    checks.push({
      label: 'Your labels stay specific',
      value: percent(m.custom_label_fraction, 0),
      caption: 'of test mail went to your labels · 10% at most',
      passed: m.custom_label_fraction <= MAX_CUSTOM_LABEL_FRACTION,
    })
  }
  return checks
}
