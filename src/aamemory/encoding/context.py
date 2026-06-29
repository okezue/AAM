from __future__ import annotations
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from aamemory.schema import SparseCode
def stablehash(text: str, seed: int = 0) -> int:
    key = int(seed).to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
def buckettimestamp(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return [f"time:raw:{text[:32]}"]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return [
        f"time:year:{dt.year}",
        f"time:month:{dt.year}-{dt.month:02d}",
        f"time:weekday:{dt.weekday()}",
        f"time:hour:{dt.hour}",
    ]
def flatten(value: Any, prefix: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"{prefix}:{str(value).lower()}"]
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            return [f"{prefix}:nan"]
        x = float(value)
        sign = "neg" if x < 0 else "pos"
        mag = math.floor(math.log10(abs(x) + 1e-9)) if abs(x) > 0 else -9
        rounded = round(x, 3)
        return [f"{prefix}:num:{sign}:1e{mag}", f"{prefix}:num_rounded:{rounded}"]
    if isinstance(value, str):
        if prefix.endswith("timestamp") or prefix in {"timestamp", "time"}:
            return buckettimestamp(value)
        clipped = value.strip().lower()[:160]
        return [f"{prefix}:{clipped}"] if clipped else []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key in sorted(value):
            out.extend(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        out = [f"{prefix}:len:{len(value)}"]
        for i, item in enumerate(value[:64]):
            out.extend(flatten(item, f"{prefix}[{i}]")[:2])
        return out
    return [f"{prefix}:{str(value)[:80].lower()}"]
@dataclass(frozen=True)
class ContextEncodingResult:
    code: SparseCode
    features: tuple[str, ...]
    diagnostics: Mapping[str, Any]
class ContextEncoder:
    def __init__(
        self,
        *,
        dimension: int = 8192,
        topk: int = 64,
        seed: int = 13,
        normalize: bool = True,
        include_keys: tuple[str, ...] | None = None,
        exclude_keys: tuple[str, ...] = ("encoding_diagnostics",),
    ) -> None:
        if dimension <= 0:
            raise ValueError("context dimension must be positive")
        self.dimension = int(dimension)
        self.topk = int(topk)
        self.seed = int(seed)
        self.normalize = bool(normalize)
        self.include_keys = tuple(include_keys) if include_keys is not None else None
        self.exclude_keys = tuple(exclude_keys)
    def encode(self, metadata: Mapping[str, Any] | None = None) -> ContextEncodingResult:
        metadata = dict(metadata or {})
        if self.include_keys is not None:
            metadata = {k: metadata[k] for k in self.include_keys if k in metadata}
        for key in self.exclude_keys:
            metadata.pop(key, None)
        features: list[str] = []
        for key in sorted(metadata):
            features.extend(flatten(metadata[key], str(key)))
        counts = Counter(features)
        dense: dict[int, float] = {}
        for feature, count in counts.items():
            idx = stablehash(f"ctx:{feature}", self.seed) % self.dimension
            dense[idx] = dense.get(idx, 0.0) + 1.0 + math.log(max(1, count))
        if not dense:
            code = SparseCode.empty(self.dimension)
        else:
            kept = sorted(dense.items(), key=lambda item: abs(item[1]), reverse=True)[: self.topk]
            code = SparseCode.frommapping(self.dimension, dict(kept))
            if self.normalize:
                code = code.normalized()
        return ContextEncodingResult(
            code=code,
            features=tuple(sorted(counts)),
            diagnostics={"encoder": "context", "unique_features": len(counts), "nonzero": len(code.indices)},
        )
