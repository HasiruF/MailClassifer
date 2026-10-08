import { describe, expect, it } from 'vitest'
import type { ClassificationResult } from '../types'
import { resortedCategories } from './resort'

function result(category: string): ClassificationResult {
  return {
    spam: { label: 'ham', confidence: 0.1 },
    category: { label: category, confidences: { [category]: 0.9 } },
    priority: { score: 0.3, bucket: 'low' },
    inputs: { spamConf: 0.1, vaderCompound: 0, categoryLabel: category, priorityBucket: 'low', priorityConfidences: {} },
  }
}

describe('resortedCategories', () => {
  const before = new Map([
    ['a', result('Other')],
    ['b', result('Work')],
    ['c', result('Other')],
  ])

  it('maps each email whose category changed to the category it had', () => {
    const after = new Map([
      ['a', result('LINKEDIN')],
      ['b', result('Work')],
    ])
    expect(resortedCategories(before, after, new Set())).toEqual({ a: 'Other' })
  })

  it('leaves out emails the user corrected', () => {
    const after = new Map([
      ['a', result('LINKEDIN')],
      ['c', result('Personal')],
    ])
    expect(resortedCategories(before, after, new Set(['c']))).toEqual({ a: 'Other' })
  })

  it('leaves out emails with no earlier result', () => {
    expect(resortedCategories(before, new Map([['new', result('Work')]]), new Set())).toEqual({})
  })
})
