from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from aamemory.schema import BenchmarkExample
class BenchmarkDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[BenchmarkExample]:
        raise NotImplementedError
    def materialize(self, limit: int | None = None) -> list[BenchmarkExample]:
        out: list[BenchmarkExample] = []
        for example in self:
            out.append(example)
            if limit is not None and len(out) >= limit:
                break
        return out
    def prepare(self, outputdir: str | Path) -> Mapping[str, Any]:
        Path(outputdir).mkdir(parents=True, exist_ok=True)
        return {"prepared": False, "reason": "dataset loads lazily"}
