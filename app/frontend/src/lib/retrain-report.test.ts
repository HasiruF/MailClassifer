import { describe, expect, it } from 'vitest'
import type { LastAttempt } from './personalization-api'
import { reportChecks } from './retrain-report'

function attempt(metrics: LastAttempt['metrics'], overrides: Partial<LastAttempt> = {}): LastAttempt {
  return { version: 2, status: 'active', correction_count: 10, metrics, created_at: '2026-10-08T10:00:00Z', ...overrides }
}

const passed = {
  passed: true,
  own_correction_accuracy: 0.9,
  gate_accuracy: 0.776,
  base_gate_accuracy: 0.781,
  custom_label_fraction: 0.04,
  reason: null,
}

describe('reportChecks', () => {
  it('turns a passed attempt into three passing checks for the category model', () => {
    const checks = reportChecks('category', attempt(passed))!
    expect(checks.map((c) => [c.label, c.value, c.passed])).toEqual([
      ['Learned your corrections', '9 of 10', true],
      ['Still sorts other mail well', '77.6%', true],
      ['Your labels stay specific', '4%', true],
    ])
    expect(checks[0].caption).toBe('Needs 8 of 10')
    expect(checks[0].bar).toEqual({ fill: 0.9, mark: 0.8 })
    expect(checks[1].caption).toBe('78.1% before · may drop 3 points at most')
  })

  it('marks the check that failed', () => {
    const checks = reportChecks('category', attempt({ ...passed, own_correction_accuracy: 0.6 }, { status: 'rejected' }))!
    expect(checks[0]).toMatchObject({ value: '6 of 10', passed: false })
    expect(checks[1].passed).toBe(true)
  })

  it('fails the accuracy check only past a 3 point drop', () => {
    expect(reportChecks('category', attempt({ ...passed, gate_accuracy: 0.752 }))![1].passed).toBe(true)
    expect(reportChecks('category', attempt({ ...passed, gate_accuracy: 0.74 }))![1].passed).toBe(false)
  })

  it('leaves out the labels check for models without custom labels', () => {
    expect(reportChecks('priority', attempt(passed))!.map((c) => c.label)).toEqual([
      'Learned your corrections',
      'Still sorts other mail well',
    ])
  })

  it('returns null when the attempt errored before the gate ran', () => {
    expect(reportChecks('category', attempt({ error: 'boom' }, { status: 'rejected' }))).toBeNull()
  })
})
