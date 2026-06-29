from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.schema import SparseCode
class DenseTextEncoder(Protocol):
    @property
    def dimension(self) -> int: ...
    def encodedense(self, text: str) -> np.ndarray: ...
@dataclass
class GaussianRandomProjector:
    input_dimension: int
    outputdimension: int
    seed: int = 0
    density: float = 1.0
    def __post_init__(self) -> None:
        if self.input_dimension <= 0 or self.outputdimension <= 0:
            raise ValueError("projection dimensions must be positive")
        if not 0 < self.density <= 1:
            raise ValueError("density must be in (0, 1]")
        rng = np.random.default_rng(self.seed)
        matrix = rng.standard_normal((self.outputdimension, self.input_dimension), dtype=np.float32)
        if self.density < 1.0:
            mask = rng.random(matrix.shape) < self.density
            matrix *= mask
        row_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.maximum(row_norm, 1e-8)
    def __call__(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != self.input_dimension:
            raise ValueError(
                f"expected input dimension {self.input_dimension}, received {vector.size}"
            )
        return self.matrix @ vector
class ProjectedDenseEncoder(FeatureEncoder):
    def __init__(
        self,
        dense_encoder: DenseTextEncoder,
        *,
        outputdimension: int = 32768,
        topk: int = 128,
        seed: int = 0,
        positiveonly: bool = False,
        threshold: float = 0.0,
    ) -> None:
        self.dense_encoder = dense_encoder
        self.projector = GaussianRandomProjector(
            input_dimension=dense_encoder.dimension,
            outputdimension=outputdimension,
            seed=seed,
        )
        self.topk = topk
        self.positiveonly = positiveonly
        self.threshold = threshold
    @property
    def dimension(self) -> int:
        return self.projector.outputdimension
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        dense = self.dense_encoder.encodedense(text)
        projected = self.projector(dense)
        code = SparseCode.fromdense(
            projected,
            topk=self.topk,
            positiveonly=self.positiveonly,
            threshold=self.threshold,
            normalize=True,
        )
        return EncodingResult(
            code=code,
            diagnostics={
                "encoder": "projected_dense",
                "input_dimension": int(dense.size),
                "outputdimension": self.dimension,
            },
        )
