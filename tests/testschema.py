from __future__ import annotations
import numpy as np
import pytest
from aamemory.schema import SparseCode
def testsparsecodesortsnormalizesanddots() -> None:
    code = SparseCode(8, (5, 1), (4.0, 3.0)).normalized()
    assert code.indices == (1, 5)
    assert code.norm() == pytest.approx(1.0)
    other = SparseCode.frommapping(8, {1: 1.0, 7: 2.0})
    assert code.dot(other) == pytest.approx(3.0 / 5.0)
    dense = code.todense()
    assert dense.shape == (8,)
    assert np.count_nonzero(dense) == 2
def testsparsecoderejectsinvalidindicesandduplicates() -> None:
    with pytest.raises(ValueError):
        SparseCode(4, (0, 4), (1.0, 1.0))
    with pytest.raises(ValueError):
        SparseCode(4, (1, 1), (1.0, 2.0))
