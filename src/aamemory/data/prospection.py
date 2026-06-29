from __future__ import annotations
import random
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def source(text: str, eid: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=eid, uri=f"prospection://{eid}")
class ProspectionDataset(BenchmarkDataset):
    def __init__(self, *, examplespertask: int = 25, seed: int = 0) -> None:
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        for idx in range(self.examplespertask):
            rng = random.Random(self.seed + idx)
            project = f"project-{idx}"
            milestone = rng.choice(["alpha", "beta", "gamma", "delta"])
            answer = rng.choice(["notebook", "terminal", "calendar", "browser"])
            eid = f"future:{idx}:support"
            text = f"Planning trace: when {project} reaches milestone {milestone}, the next useful tool will be {answer}."
            event = MemoryEvent(
                event_id=eid,
                text=text,
                source=source(text, eid),
                metadata={"task_state": {"project": project, "milestone": milestone}, "future_relevant": True},
            )
            yield BenchmarkExample.build(
                example_id=f"future:{idx}",
                task="future_simulation",
                events=[event],
                query=f"{project} has reached milestone {milestone}. Which tool should be prefetched?",
                answers=[answer],
                evidence_ids=[eid],
                metadata={"topic": project, "milestone": milestone, "tests": "prospective_prefetch"},
            )
