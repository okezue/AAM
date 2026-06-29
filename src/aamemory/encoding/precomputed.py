from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.schema import SparseCode
def textsha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
class PrecomputedFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        path: str | Path,
        dimension: int | None = None,
        verifytext: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.verifytext = bool(verifytext)
        self._records: dict[str, Mapping[str, Any]] = {}
        inferred_dimension: int | None = None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("textsha256") or "")
            if not key and "text" in row:
                key = textsha256(str(row["text"]))
            if not key:
                raise ValueError(f"missing textsha256 in {self.path}:{line_number}")
            row_dimension = int(row["dimension"])
            if inferred_dimension is None:
                inferred_dimension = row_dimension
            elif row_dimension != inferred_dimension:
                raise ValueError(
                    f"inconsistent dimensions in {self.path}: {inferred_dimension} vs {row_dimension}"
                )
            if key in self._records:
                raise ValueError(f"duplicate textsha256 {key} in {self.path}:{line_number}")
            self._records[key] = row
        self._dimension = int(dimension or inferred_dimension or 0)
        if self._dimension <= 0:
            raise ValueError("precomputed feature file is empty or has no valid dimension")
        if inferred_dimension is not None and dimension is not None and inferred_dimension != dimension:
            raise ValueError(
                f"configured dimension {dimension} does not match artifact dimension {inferred_dimension}"
            )
    @property
    def dimension(self) -> int:
        return self._dimension
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        key = textsha256(text)
        try:
            row = self._records[key]
        except KeyError as exc:
            raise KeyError(
                f"no precomputed feature record for SHA-256 {key}; exact input text must match export"
            ) from exc
        if self.verifytext and "text" in row and str(row["text"]) != text:
            raise ValueError(f"text hash collision or corrupted record for {key}")
        code = SparseCode(
            dimension=int(row["dimension"]),
            indices=tuple(int(value) for value in row.get("indices", [])),
            values=tuple(float(value) for value in row.get("values", [])),
        )
        return EncodingResult(
            code=code,
            payload=dict(row.get("payload", {})),
            diagnostics={
                "encoder": "precomputed",
                "artifact": str(self.path),
                "textsha256": key,
                **dict(row.get("diagnostics", {})),
            },
        )
