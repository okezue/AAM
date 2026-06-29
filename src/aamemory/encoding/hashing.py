from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.schema import SparseCode
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_DEFAULT_ALIASES: dict[str, str] = {
    "automobile": "car",
    "vehicle": "car",
    "colour": "color",
    "hue": "color",
    "resides": "lives",
    "dwells": "lives",
    "enjoys": "likes",
    "prefers": "likes",
    "favourite": "favorite",
    "favorite": "likes",
    "purchased": "bought",
    "acquired": "bought",
}
def stablehash(text: str, seed: int) -> int:
    key = seed.to_bytes(8, byteorder="little", signed=False)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, key=key).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
class HashingFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        dimension: int = 65536,
        topk: int = 128,
        wordngrams: tuple[int, ...] = (1, 2),
        charngrams: tuple[int, ...] = (),
        signed: bool = False,
        sublineartf: bool = True,
        positiveonly: bool = True,
        normalize: bool = True,
        seed: int = 0,
        aliases: Mapping[str, str] | None = None,
        include_boundaries: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if topk <= 0:
            raise ValueError("topk must be positive")
        self._dimension = int(dimension)
        self.topk = int(topk)
        self.wordngrams = tuple(sorted(set(int(n) for n in wordngrams if int(n) > 0)))
        self.charngrams = tuple(sorted(set(int(n) for n in charngrams if int(n) > 0)))
        self.signed = bool(signed)
        self.sublineartf = bool(sublineartf)
        self.positiveonly = bool(positiveonly)
        self.normalize = bool(normalize)
        self.seed = int(seed)
        self.aliases = {**_DEFAULT_ALIASES, **dict(aliases or {})}
        self.include_boundaries = include_boundaries
    @property
    def dimension(self) -> int:
        return self._dimension
    def tokens(self, text: str) -> list[str]:
        tokens = [self.aliases.get(tok, tok) for tok in _TOKEN_RE.findall(text.lower())]
        if self.include_boundaries:
            return ["<bos>", *tokens, "<eos>"]
        return tokens
    def features(self, text: str) -> Iterable[str]:
        tokens = self.tokens(text)
        for n in self.wordngrams:
            for i in range(max(0, len(tokens) - n + 1)):
                yield f"w{n}:" + "_".join(tokens[i : i + n])
        if self.charngrams:
            compact = " ".join(tokens)
            for n in self.charngrams:
                for i in range(max(0, len(compact) - n + 1)):
                    yield f"c{n}:" + compact[i : i + n]
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        counts = Counter(self.features(text))
        dense: dict[int, float] = {}
        collisions = 0
        for feature, count in counts.items():
            raw_hash = stablehash(feature, self.seed)
            index = raw_hash % self.dimension
            sign = -1.0 if self.signed and ((raw_hash >> 63) & 1) else 1.0
            weight = 1.0 + math.log(count) if self.sublineartf and count > 0 else float(count)
            if index in dense:
                collisions += 1
            dense[index] = dense.get(index, 0.0) + sign * weight
        if not dense:
            code = SparseCode.empty(self.dimension)
        else:
            indices = np.fromiter(dense.keys(), dtype=np.int64)
            values = np.fromiter((dense[i] for i in indices), dtype=np.float64)
            order = np.argsort(np.abs(values))[::-1][: self.topk]
            kept = {int(indices[i]): float(values[i]) for i in order}
            code = SparseCode.frommapping(self.dimension, kept)
            if self.positiveonly:
                code = SparseCode.frommapping(
                    self.dimension, {i: max(0.0, v) for i, v in code.asdict().items()}
                )
            if self.normalize:
                code = code.normalized()
        return EncodingResult(
            code=code,
            diagnostics={
                "encoder": "hashing",
                "unique_features": len(counts),
                "collisions": collisions,
                "nonzero": len(code.indices),
                "metadata_seen": bool(metadata),
            },
        )
