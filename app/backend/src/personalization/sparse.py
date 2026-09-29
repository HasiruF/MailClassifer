import numpy as np
import scipy.sparse as sp
from pydantic import BaseModel, model_validator


class SparseVector(BaseModel):
    dim: int
    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def _check_shape(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have the same length")
        if any(i < 0 or i >= self.dim for i in self.indices):
            raise ValueError("index out of range for dim")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("duplicate indices")
        return self


def to_csr(vectors: list[SparseVector], dim: int) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row, vector in enumerate(vectors):
        if vector.dim != dim:
            raise ValueError(f"vector dim {vector.dim} != expected dim {dim}")
        rows.extend([row] * len(vector.indices))
        cols.extend(vector.indices)
        vals.extend(vector.values)
    return sp.csr_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))),
        shape=(len(vectors), dim),
    )
