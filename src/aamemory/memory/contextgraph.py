from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from aamemory.schema import SparseCode
@dataclass(frozen=True)
class ContextGraphStats:
    context_nodes: int
    feature_edges: int
class ContextAssociationGraph:
    def __init__(self, *, learningrate: float = 0.10, decay: float = 0.0005, maxdegree: int = 128) -> None:
        self.learningrate = float(learningrate)
        self.decay_rate = float(decay)
        self.maxdegree = int(maxdegree)
        self.links: dict[int, dict[int, float]] = defaultdict(dict)
        self.write_count = 0
    @classmethod
    def fromconfig(cls, config: Mapping[str, Any] | None = None) -> ContextAssociationGraph:
        config = dict(config or {})
        return cls(
            learningrate=float(config.get("learningrate", 0.10)),
            decay=float(config.get("decay", 0.0005)),
            maxdegree=int(config.get("maxdegree", 128)),
        )
    def write(self, context: SparseCode, address: SparseCode, *, salience: float = 1.0) -> None:
        self.write_count += 1
        if not context.indices or not address.indices:
            return
        context_pairs = list(zip(context.indices, context.values, strict=True))[:64]
        address_pairs = list(zip(address.indices, address.values, strict=True))[:128]
        eta = self.learningrate * float(salience)
        for ci, cv in context_pairs:
            row = self.links[ci]
            for ai, av in address_pairs:
                row[ai] = row.get(ai, 0.0) + eta * cv * av
            if len(row) > self.maxdegree:
                keep = sorted(row, key=lambda j: abs(row[j]), reverse=True)[: self.maxdegree]
                self.links[ci] = {j: row[j] for j in keep}
    def messages(self, context: SparseCode, *, normalize: bool = True) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for ci, cv in zip(context.indices, context.values, strict=True):
            neighbors = self.links.get(ci, {})
            denom = math.sqrt(sum(v * v for v in neighbors.values())) if normalize else 1.0
            denom = max(denom, 1e-12)
            for ai, w in neighbors.items():
                out[ai] += cv * w / denom
        return dict(out)
    def decay(self, rate: float | None = None) -> None:
        multiplier = max(0.0, 1.0 - (self.decay_rate if rate is None else float(rate)))
        for ci in list(self.links):
            for ai in list(self.links[ci]):
                self.links[ci][ai] *= multiplier
                if abs(self.links[ci][ai]) < 1e-9:
                    del self.links[ci][ai]
            if not self.links[ci]:
                del self.links[ci]
    def stats(self) -> ContextGraphStats:
        return ContextGraphStats(len(self.links), sum(len(x) for x in self.links.values()))
    def statedict(self) -> dict[str, Any]:
        return {"links": {str(i): {str(j): v for j, v in n.items()} for i, n in self.links.items()}, "write_count": self.write_count}
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        self.links = defaultdict(
            dict,
            {int(i): {int(j): float(v) for j, v in n.items()} for i, n in state.get("links", {}).items()},
        )
        self.write_count = int(state.get("write_count", 0))
