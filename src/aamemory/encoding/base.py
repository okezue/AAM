from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
import numpy as np
from aamemory.schema import SparseCode
@dataclass(frozen=True)
class EncodingResult:
    code: SparseCode
    payload: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
class DenseActivationProvider(Protocol):
    @property
    def featuredimension(self) -> int: ...
    def activations(self, text: str) -> np.ndarray: ...
class FeatureEncoder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        raise NotImplementedError
    def encodemany(
        self,
        texts: Sequence[str],
        *,
        metadata: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[EncodingResult]:
        metas = metadata or [None] * len(texts)
        if len(metas) != len(texts):
            raise ValueError("metadata and texts must have equal length")
        return [self.encode(text, metadata=meta) for text, meta in zip(texts, metas, strict=True)]
