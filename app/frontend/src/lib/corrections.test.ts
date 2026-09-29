import { describe, expect, it } from 'vitest'
import type { ClassificationResult } from '../types'
import { applyCorrection, applyCorrections, predictedFor } from './corrections'

const result: ClassificationResult = {
  spam: { label: 'ham', confidence: 0.2 },
  category: { label: 'Other', confidences: { Work: 0.3, Personal: 0.1, Other: 0.6 } },
  priority: { score: 0.1, bucket: 'low', note: 'classified as spam (conf=0.20) — priority suppressed' },
  inputs: {
    spamConf: 0.2,
    vaderCompound: -0.1,
    categoryLabel: 'Other',
    priorityBucket: 'medium',
    priorityConfidences: { high: 0.1, low: 0.3, medium: 0.6 },
  },
}

describe('predictedFor', () => {
  it('reports what each model actually predicted, before suppression or corrections', () => {
    expect(predictedFor('category', result)).toEqual({ label: 'Other', confidence: 0.6 })
    expect(predictedFor('priority', result)).toEqual({ label: 'medium', confidence: 0.6 })
    expect(predictedFor('spam', result)).toEqual({ label: 'ham', confidence: 0.8 })
  })
})

describe('applyCorrection', () => {
  it('overrides the category label, including a custom one', () => {
    expect(applyCorrection(result, 'category', 'Finance').category.label).toBe('Finance')
  })

  it('moves the priority score to the bucket representative and drops the suppression note', () => {
    expect(applyCorrection(result, 'priority', 'high').priority).toEqual({ score: 0.8, bucket: 'high' })
  })

  it('flips the spam label without touching its confidence', () => {
    expect(applyCorrection(result, 'spam', 'spam').spam).toEqual({ label: 'spam', confidence: 0.2 })
  })

  it('never mutates the model result', () => {
    applyCorrection(result, 'category', 'Work')
    expect(result.category.label).toBe('Other')
  })
})

describe('applyCorrections', () => {
  it('applies every stored correction', () => {
    const shown = applyCorrections(result, { category: 'Work', priority: 'high' })
    expect(shown.category.label).toBe('Work')
    expect(shown.priority.bucket).toBe('high')
    expect(shown.spam.label).toBe('ham')
  })
})
