from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from aamemory.config import GraphConfig, MemoryConfig
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.memory.system import ActivationAssociativeMemory
from aamemory.schema import MemoryEvent, SparseCode
class MetadataEncoder(FeatureEncoder):
    @property
    def dimension(self) -> int:
        return 16
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        feature = int((metadata or {}).get("feature", 0))
        companion = int((metadata or {}).get("companion", feature + 1))
        return EncodingResult(
            SparseCode.frommapping(16, {feature: 1.0, companion: 1.0}).normalized()
        )
def testevictionrebuildremovesghostgraphfeatures() -> None:
    memory = ActivationAssociativeMemory(
        MemoryConfig(
            graph=GraphConfig(
                rule="hebb",
                learningrate=1.0,
                temporallearningrate=0.0,
                normalizeafterwrite=False,
                maxdegree=16,
            ),
            consolidation={
                "fixedcapacityepisodes": 1,
                "evictionpolicy": "fifo",
                "rebuildgraphaftermaintenance": True,
            },
        ),
        encoder=MetadataEncoder(),
    )
    memory.write(
        MemoryEvent(
            "old",
            "old",
            timestamp="2026-01-01T00:00:00+00:00",
            metadata={"feature": 1, "companion": 2},
        )
    )
    memory.write(
        MemoryEvent(
            "new",
            "new",
            timestamp="2026-01-02T00:00:00+00:00",
            metadata={"feature": 7, "companion": 8},
        )
    )
    assert memory.store.get("old") is None
    assert 1 not in memory.graph.association
    assert 2 not in memory.graph.association
    assert 7 in memory.graph.association
    memory.close()
