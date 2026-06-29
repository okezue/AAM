from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from aamemory.config import SalienceConfig
@dataclass(frozen=True)
class SalienceSignals:
    surprise: float = 0.0
    task_relevance: float = 0.0
    user_importance: float = 0.0
    novelty: float = 0.0
    redundancy: float = 0.0
    @classmethod
    def frommetadata(cls, metadata: Mapping[str, object] | None) -> SalienceSignals:
        metadata = metadata or {}
        return cls(
            surprise=float(metadata.get("surprise", 0.0)),
            task_relevance=float(metadata.get("task_relevance", 0.0)),
            user_importance=float(metadata.get("user_importance", 0.0)),
            novelty=float(metadata.get("novelty", 0.0)),
            redundancy=float(metadata.get("redundancy", 0.0)),
        )
class SalienceGate:
    def __init__(self, config: SalienceConfig | None = None) -> None:
        self.config = config or SalienceConfig()
    def score(self, signals: SalienceSignals) -> float:
        c = self.config
        value = (
            c.base
            + c.surpriseweight * signals.surprise
            + c.taskweight * signals.task_relevance
            + c.userweight * signals.user_importance
            + c.noveltyweight * signals.novelty
            - c.redundancyweight * signals.redundancy
        )
        return max(c.minimum, min(c.maximum, value))
