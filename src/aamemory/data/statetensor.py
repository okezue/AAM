from __future__ import annotations
import random
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def source(text: str, eid: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=eid, uri=f"state-tensor://{eid}")
class StateTensorDataset(BenchmarkDataset):
    def __init__(self, *, examplespertask: int = 25, seed: int = 0) -> None:
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        labels = ["accumulate", "liquidate", "hold", "rebalance"]
        for idx in range(self.examplespertask):
            rng = random.Random(self.seed + idx)
            state = [round(rng.uniform(-2, 2), 3) for _ in range(12)]
            label = labels[int(sum(x > 0 for x in state)) % len(labels)]
            eid = f"state:{idx}:support"
            text = f"State canvas {idx} produced latent macro-state label {label}."
            event = MemoryEvent(
                event_id=eid,
                text=text,
                source=source(text, eid),
                metadata={"modality": "state_tensor", "state_tensor": state, "state_id": f"canvas-{idx}"},
            )
            yield BenchmarkExample.build(
                example_id=f"state:{idx}",
                task="state_tensor",
                events=[event],
                query=f"For state canvas {idx}, what latent macro-state label was produced?",
                answers=[label],
                evidence_ids=[eid],
                metadata={"modality": "state_tensor", "state_tensor": state, "state_id": f"canvas-{idx}"},
            )
