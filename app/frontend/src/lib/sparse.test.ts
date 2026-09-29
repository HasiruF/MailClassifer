import { describe, expect, it } from 'vitest'
import { toDense, toSparse } from './sparse'

describe('toSparse', () => {
  it('keeps only non-zero entries, including negative ones', () => {
    expect(toSparse(new Float32Array([0, 0.5, 0, -1.25]))).toEqual({ dim: 4, indices: [1, 3], values: [0.5, -1.25] })
  })

  it('round-trips through toDense exactly', () => {
    const dense = new Float32Array([0.1, 0, 0, 3.7, 0, -0.2])
    expect(Array.from(toDense(toSparse(dense)))).toEqual(Array.from(dense))
  })

  it('encodes an all-zero vector as empty', () => {
    expect(toSparse(new Float32Array(3))).toEqual({ dim: 3, indices: [], values: [] })
  })
})
