from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.schema import SparseCode
@dataclass(frozen=True)
class AllocationDecision:
    selected_features: tuple[int, ...]
    feature_gain: Mapping[int, float]
    diagnostics: Mapping[str, float | int]
class EligibilityAllocator:
    def __init__(
        self,
        *,
        eligibility_decay: float = 0.95,
        noveltygain: float = 0.30,
        utilitygain: float = 0.20,
        hubpenalty: float = 0.50,
        max_gain: float = 2.0,
        min_gain: float = 0.15,
        select_top_k: int | None = None,
    ) -> None:
        self.eligibility_decay = float(eligibility_decay)
        self.noveltygain = float(noveltygain)
        self.utilitygain = float(utilitygain)
        self.hubpenalty = float(hubpenalty)
        self.max_gain = float(max_gain)
        self.min_gain = float(min_gain)
        self.select_top_k = select_top_k
        self.eligibility: dict[int, float] = defaultdict(float)
        self.authority: dict[int, float] = defaultdict(lambda: 1.0)
        self.write_count = 0
    @classmethod
    def fromconfig(cls, config: Mapping[str, Any] | None = None) -> EligibilityAllocator:
        config = dict(config or {})
        return cls(
            eligibility_decay=float(config.get("eligibility_decay", 0.95)),
            noveltygain=float(config.get("noveltygain", 0.30)),
            utilitygain=float(config.get("utilitygain", 0.20)),
            hubpenalty=float(config.get("hubpenalty", 0.50)),
            max_gain=float(config.get("max_gain", 2.0)),
            min_gain=float(config.get("min_gain", 0.15)),
            select_top_k=config.get("select_top_k"),
        )
    def degree(self, graph: SparseAssociationGraph, feature: int) -> float:
        return sum(abs(x) for x in graph.association.get(feature, {}).values())
    def allocate(
        self,
        code: SparseCode,
        *,
        graph: SparseAssociationGraph,
        novelty: float = 0.0,
        utility: float = 0.0,
    ) -> AllocationDecision:
        self.write_count += 1
        for feature in list(self.eligibility):
            self.eligibility[feature] *= self.eligibility_decay
            if self.eligibility[feature] < 1e-6:
                del self.eligibility[feature]
        feature_gain: dict[int, float] = {}
        pairs = list(zip(code.indices, code.values, strict=True))
        if self.select_top_k is not None:
            pairs = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)[: int(self.select_top_k)]
        for feature, value in pairs:
            degree = self.degree(graph, feature)
            self.eligibility[feature] += abs(value)
            raw = (
                1.0
                + self.noveltygain * novelty
                + self.utilitygain * utility
                + 0.05 * math.log1p(self.eligibility[feature])
            )
            inhibition = (1.0 + degree) ** self.hubpenalty
            gain = max(self.min_gain, min(self.max_gain, raw * self.authority[feature] / inhibition))
            feature_gain[feature] = gain
        return AllocationDecision(
            selected_features=tuple(feature_gain),
            feature_gain=feature_gain,
            diagnostics={
                "selected_features": len(feature_gain),
                "mean_gain": sum(feature_gain.values()) / max(1, len(feature_gain)),
                "eligibility_features": len(self.eligibility),
            },
        )
    def apply(self, code: SparseCode, decision: AllocationDecision) -> SparseCode:
        values = {
            i: v * float(decision.feature_gain.get(i, 1.0))
            for i, v in zip(code.indices, code.values, strict=True)
        }
        return SparseCode.frommapping(code.dimension, values).normalized()
    def statedict(self) -> dict[str, Any]:
        return {
            "eligibility": {str(i): v for i, v in self.eligibility.items()},
            "authority": {str(i): v for i, v in self.authority.items()},
            "write_count": self.write_count,
        }
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        self.eligibility = defaultdict(float, {int(i): float(v) for i, v in state.get("eligibility", {}).items()})
        self.authority = defaultdict(lambda: 1.0, {int(i): float(v) for i, v in state.get("authority", {}).items()})
        self.write_count = int(state.get("write_count", 0))
