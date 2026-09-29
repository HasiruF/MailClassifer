// Dense model vectors are thousands of dimensions and nearly all zero, so
// corrections travel as (index, value) pairs. Lossless: toDense(toSparse(v))
// reproduces v exactly.
export interface SparseVector {
  dim: number
  indices: number[]
  values: number[]
}

export function toSparse(dense: Float32Array): SparseVector {
  const indices: number[] = []
  const values: number[] = []
  for (let i = 0; i < dense.length; i++) {
    if (dense[i] !== 0) {
      indices.push(i)
      values.push(dense[i])
    }
  }
  return { dim: dense.length, indices, values }
}

export function toDense(vector: SparseVector): Float32Array {
  const out = new Float32Array(vector.dim)
  vector.indices.forEach((index, k) => {
    out[index] = vector.values[k]
  })
  return out
}
