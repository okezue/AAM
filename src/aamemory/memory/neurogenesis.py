from __future__ import annotations
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from aamemory.schema import SparseCode, utcnowiso
@dataclass
class FeatureBirthRecord:
    feature_id: int
    parent_features: tuple[int, ...]
    created_at: str = field(default_factory=utcnowiso)
    maturity: float = 0.0
    authority: float = 0.10
    writes: int = 0
    verified_hits: int = 0
    false_hits: int = 0
    status: str = "immature"
    def todict(self) -> dict[str, Any]:
        return asdict(self)
class NeurogenesisController:
    def __init__(
        self,
        *,
        reserved_start: int,
        reserved_size: int = 0,
        enabled: bool = False,
        birththreshold: float = 0.75,
        noveltyweight: float = 0.45,
        interference_weight: float = 0.30,
        retrieval_error_weight: float = 0.30,
        budget_pressure_weight: float = -0.40,
        immature_authority: float = 0.10,
        maturationhits: int = 3,
    ) -> None:
        self.reserved_start = int(reserved_start)
        self.reserved_size = int(reserved_size)
        self.enabled = bool(enabled and reserved_size > 0)
        self.birththreshold = float(birththreshold)
        self.noveltyweight = float(noveltyweight)
        self.interference_weight = float(interference_weight)
        self.retrieval_error_weight = float(retrieval_error_weight)
        self.budget_pressure_weight = float(budget_pressure_weight)
        self.immature_authority = float(immature_authority)
        self.maturationhits = int(maturationhits)
        self.records: dict[int, FeatureBirthRecord] = {}
        self.next_slot = 0
    @classmethod
    def fromconfig(
        cls,
        *,
        reserved_start: int,
        config: Mapping[str, Any] | None = None,
    ) -> NeurogenesisController:
        config = dict(config or {})
        return cls(
            reserved_start=reserved_start,
            reserved_size=int(config.get("reservedfeatures", 0)),
            enabled=bool(config.get("enabled", False)),
            birththreshold=float(config.get("birththreshold", 0.75)),
            noveltyweight=float(config.get("noveltyweight", 0.45)),
            interference_weight=float(config.get("interference_weight", 0.30)),
            retrieval_error_weight=float(config.get("retrieval_error_weight", 0.30)),
            budget_pressure_weight=float(config.get("budget_pressure_weight", -0.40)),
            immature_authority=float(config.get("immature_authority", 0.10)),
            maturationhits=int(config.get("maturationhits", 3)),
        )
    def birthscore(
        self,
        *,
        novelty: float,
        interference: float,
        retrieval_error: float,
        budget_pressure: float,
    ) -> float:
        raw = (
            self.noveltyweight * novelty
            + self.interference_weight * interference
            + self.retrieval_error_weight * retrieval_error
            + self.budget_pressure_weight * budget_pressure
        )
        return 1.0 / (1.0 + math.exp(-4.0 * (raw - 0.5)))
    def maybeaugment(
        self,
        code: SparseCode,
        *,
        novelty: float,
        interference: float = 0.0,
        retrieval_error: float = 0.0,
        budget_pressure: float = 0.0,
    ) -> tuple[SparseCode, FeatureBirthRecord | None, float]:
        if not self.enabled or self.next_slot >= self.reserved_size:
            return code, None, 0.0
        score = self.birthscore(
            novelty=novelty,
            interference=interference,
            retrieval_error=retrieval_error,
            budget_pressure=budget_pressure,
        )
        if score < self.birththreshold:
            return code, None, score
        feature_id = self.reserved_start + self.next_slot
        self.next_slot += 1
        parents = tuple(i for i, _ in sorted(zip(code.indices, code.values, strict=True), key=lambda item: abs(item[1]), reverse=True)[:8])
        record = FeatureBirthRecord(
            feature_id=feature_id,
            parent_features=parents,
            authority=self.immature_authority,
            writes=1,
        )
        self.records[feature_id] = record
        values = code.asdict()
        values[feature_id] = max(values.values(), default=1.0) * self.immature_authority
        return SparseCode.frommapping(code.dimension, values).normalized(), record, score
    def verifyhit(self, feature_id: int, *, helpful: bool) -> None:
        record = self.records.get(int(feature_id))
        if record is None:
            return
        if helpful:
            record.verified_hits += 1
            record.maturity = min(1.0, record.maturity + 1.0 / max(1, self.maturationhits))
            record.authority = min(1.0, record.authority + 0.25)
        else:
            record.false_hits += 1
            record.authority = max(0.0, record.authority - 0.25)
        if record.verified_hits >= self.maturationhits:
            record.status = "mature"
        if record.false_hits > record.verified_hits + 2:
            record.status = "pruned"
            record.authority = 0.0
    def stats(self) -> dict[str, int | float]:
        return {
            "births": len(self.records),
            "mature_features": sum(r.status == "mature" for r in self.records.values()),
            "immature_features": sum(r.status == "immature" for r in self.records.values()),
            "pruned_features": sum(r.status == "pruned" for r in self.records.values()),
            "reserved_size": self.reserved_size,
            "next_slot": self.next_slot,
        }
    def statedict(self) -> dict[str, Any]:
        return {
            "records": {str(i): record.todict() for i, record in self.records.items()},
            "next_slot": self.next_slot,
        }
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        self.records = {
            int(i): FeatureBirthRecord(
                feature_id=int(value["feature_id"]),
                parent_features=tuple(int(x) for x in value.get("parent_features", ())),
                created_at=str(value.get("created_at", utcnowiso())),
                maturity=float(value.get("maturity", 0.0)),
                authority=float(value.get("authority", self.immature_authority)),
                writes=int(value.get("writes", 0)),
                verified_hits=int(value.get("verified_hits", 0)),
                false_hits=int(value.get("false_hits", 0)),
                status=str(value.get("status", "immature")),
            )
            for i, value in state.get("records", {}).items()
        }
        self.next_slot = int(state.get("next_slot", 0))
