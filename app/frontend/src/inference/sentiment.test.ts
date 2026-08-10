import { describe, it, expect } from 'vitest'
import { vaderCompound } from './sentiment'

describe('vaderCompound', () => {
  it('matches Python vaderSentiment on a positive-urgency sentence', () => {
    expect(vaderCompound('urgent approval needed please sign off before end of day')).toBeCloseTo(
      0.7351,
      4,
    )
  })

  it('matches Python vaderSentiment on a casual sentence', () => {
    expect(
      vaderCompound('lunch on friday hey are you free for lunch this friday let me know'),
    ).toBeCloseTo(0.5106, 4)
  })
})
