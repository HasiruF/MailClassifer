import { afterEach, describe, expect, it, vi } from 'vitest'
import { getCorrectionDetail, submitCorrection, type CorrectionPayload } from './personalization-api'

const payload: CorrectionPayload = {
  model: 'category',
  provider_message_id: 'm1',
  feature_vector: { dim: 2, indices: [0], values: [1] },
  predicted_label: 'Other',
  predicted_confidence: 0.6,
  corrected_label: 'Work',
}

describe('submitCorrection', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('posts the correction with credentials and returns the parsed body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ retrain_scheduled: false, corrections_until_retrain: 4 }), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await expect(submitCorrection(payload)).resolves.toEqual({ retrain_scheduled: false, corrections_until_retrain: 4 })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:3011/personalization/corrections')
    expect(init.method).toBe('POST')
    expect(init.credentials).toBe('include')
    expect(JSON.parse(init.body)).toEqual(payload)
  })

  it('throws with the status code when the backend refuses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 403 })))
    await expect(submitCorrection(payload)).rejects.toThrow('403')
  })
})

describe('getCorrectionDetail', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('asks for one message, encoded, with credentials', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('[]', { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
    await expect(getCorrectionDetail('a/b')).resolves.toEqual([])
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:3011/personalization/corrections/a%2Fb')
    expect(init.credentials).toBe('include')
  })
})
