from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import numpy as np
def utcnowiso() -> str:
    return datetime.now(timezone.utc).isoformat()
@dataclass(frozen=True)
class SparseCode:
    dimension: int
    indices: tuple[int, ...]
    values: tuple[float, ...]
    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have equal length")
        pairs = sorted(zip(self.indices, self.values, strict=True))
        if any(i < 0 or i >= self.dimension for i, _ in pairs):
            raise ValueError("sparse index outside ambient dimension")
        if len({i for i, _ in pairs}) != len(pairs):
            raise ValueError("duplicate sparse indices are not allowed")
        object.__setattr__(self, "indices", tuple(i for i, _ in pairs))
        object.__setattr__(self, "values", tuple(float(v) for _, v in pairs))
    @classmethod
    def empty(cls, dimension: int) -> SparseCode:
        return cls(dimension=dimension, indices=(), values=())
    @classmethod
    def fromdense(
        cls,
        vector: Sequence[float] | np.ndarray,
        *,
        topk: int,
        positiveonly: bool = False,
        normalize: bool = True,
        threshold: float = 0.0,
    ) -> SparseCode:
        arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        if positiveonly:
            arr = np.maximum(arr, 0.0)
        if threshold > 0:
            arr = np.where(np.abs(arr) >= threshold, arr, 0.0)
        if topk <= 0 or arr.size == 0:
            return cls.empty(int(arr.size))
        k = min(topk, int(np.count_nonzero(arr)))
        if k == 0:
            return cls.empty(int(arr.size))
        candidate = np.argpartition(np.abs(arr), -k)[-k:]
        candidate = candidate[np.argsort(candidate)]
        vals = arr[candidate]
        code = cls(int(arr.size), tuple(map(int, candidate)), tuple(map(float, vals)))
        return code.normalized() if normalize else code
    @classmethod
    def frommapping(cls, dimension: int, values: Mapping[int, float]) -> SparseCode:
        kept = {int(i): float(v) for i, v in values.items() if float(v) != 0.0}
        return cls(dimension, tuple(kept), tuple(kept.values()))
    def todense(self, dtype: np.dtype[Any] = np.float32) -> np.ndarray:
        out = np.zeros(self.dimension, dtype=dtype)
        if self.indices:
            out[np.asarray(self.indices)] = np.asarray(self.values, dtype=dtype)
        return out
    def asdict(self) -> dict[int, float]:
        return dict(zip(self.indices, self.values, strict=True))
    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.values))
    def normalized(self, eps: float = 1e-12) -> SparseCode:
        n = self.norm()
        if n <= eps:
            return self
        return SparseCode(self.dimension, self.indices, tuple(v / n for v in self.values))
    def topk(self, k: int, *, positiveonly: bool = False) -> SparseCode:
        if k >= len(self.indices):
            return self
        pairs = list(zip(self.indices, self.values, strict=True))
        if positiveonly:
            pairs = [(i, v) for i, v in pairs if v > 0]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        pairs = sorted(pairs[: max(0, k)])
        return SparseCode(self.dimension, tuple(i for i, _ in pairs), tuple(v for _, v in pairs))
    def dot(self, other: SparseCode) -> float:
        if self.dimension != other.dimension:
            raise ValueError("cannot dot sparse codes with different dimensions")
        i = j = 0
        total = 0.0
        while i < len(self.indices) and j < len(other.indices):
            a, b = self.indices[i], other.indices[j]
            if a == b:
                total += self.values[i] * other.values[j]
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        return float(total)
    def overlap(self, other: SparseCode) -> int:
        return len(set(self.indices).intersection(other.indices))
    def scaled(self, scalar: float) -> SparseCode:
        return SparseCode(self.dimension, self.indices, tuple(scalar * v for v in self.values))
    def tojsonable(self) -> dict[str, Any]:
        return {"dimension": self.dimension, "indices": list(self.indices), "values": list(self.values)}
    @classmethod
    def fromjsonable(cls, value: Mapping[str, Any]) -> SparseCode:
        return cls(
            dimension=int(value["dimension"]),
            indices=tuple(int(x) for x in value["indices"]),
            values=tuple(float(x) for x in value["values"]),
        )
@dataclass(frozen=True)
class SourceRef:
    uri: str | None = None
    document_id: str | None = None
    start: int | None = None
    end: int | None = None
    checksum: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    @classmethod
    def fortext(
        cls,
        text: str,
        *,
        uri: str | None = None,
        document_id: str | None = None,
        start: int | None = None,
        end: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SourceRef:
        checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(uri, document_id, start, end, checksum, metadata or {})
@dataclass
class Episode:
    episode_id: str
    text: str
    code: SparseCode
    timestamp: str = field(default_factory=utcnowiso)
    salience: float = 1.0
    confidence: float = 1.0
    source: SourceRef = field(default_factory=SourceRef)
    payload: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    def bytesizeestimate(self) -> int:
        code_bytes = len(self.code.indices) * (8 + 4)
        return len(self.text.encode("utf-8")) + code_bytes + len(json.dumps(self.payload))
@dataclass(frozen=True)
class QueryResult:
    episode_id: str
    score: float
    exact_score: float
    associative_score: float
    temporal_score: float
    recency_score: float
    episode: Episode
    trace: Mapping[str, Any] = field(default_factory=dict)
@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    text: str
    timestamp: str | None = None
    source: SourceRef = field(default_factory=SourceRef)
    metadata: Mapping[str, Any] = field(default_factory=dict)
@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    task: str
    events: tuple[MemoryEvent, ...]
    query: str
    answers: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()
    negative_evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    @classmethod
    def build(
        cls,
        *,
        example_id: str,
        task: str,
        events: Iterable[MemoryEvent],
        query: str,
        answers: Iterable[str],
        evidence_ids: Iterable[str] = (),
        negative_evidence_ids: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> BenchmarkExample:
        return cls(
            example_id=example_id,
            task=task,
            events=tuple(events),
            query=query,
            answers=tuple(str(a) for a in answers),
            evidence_ids=tuple(str(x) for x in evidence_ids),
            negative_evidence_ids=tuple(str(x) for x in negative_evidence_ids),
            metadata=metadata or {},
        )
