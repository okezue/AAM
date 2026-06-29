from __future__ import annotations
import random
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def source(text: str, eid: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=eid, uri=f"neurogenesis://{eid}")
class NeurogenesisDataset(BenchmarkDataset):
    def __init__(self, *, examplespertask: int = 25, seed: int = 0, distractors: int = 10) -> None:
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
        self.distractors = int(distractors)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        domains = ["astrocyte", "ribosome", "microtubule", "synapse"]
        values = ["phase-red", "phase-blue", "phase-gold", "phase-green"]
        for idx in range(self.examplespertask):
            rng = random.Random(self.seed + idx)
            domain = rng.choice(domains)
            novel_token = f"{domain}-novel-{idx}-{rng.randrange(10000)}"
            answer = rng.choice(values)
            eid = f"neuro:{idx}:novel"
            events = [
                MemoryEvent(
                    event_id=f"neuro:{idx}:d{j}",
                    text=f"Common domain fragment {domain} repeats with distractor index {j} and value {rng.choice(values)}.",
                    source=source(f"d{j}", f"neuro:{idx}:d{j}"),
                    metadata={"domain": domain, "budget_pressure": 0.1},
                )
                for j in range(self.distractors)
            ]
            text = f"New high-novelty feature {novel_token} should mature with latent value {answer}."
            events.append(
                MemoryEvent(
                    event_id=eid,
                    text=text,
                    source=source(text, eid),
                    metadata={"domain": domain, "novelty": 1.0, "retrieval_error": 1.0, "interference": 0.8},
                )
            )
            rng.shuffle(events)
            yield BenchmarkExample.build(
                example_id=f"neuro:{idx}",
                task="neurogenesis",
                events=events,
                query=f"What latent value is associated with new high-novelty feature {novel_token}?",
                answers=[answer],
                evidence_ids=[eid],
                metadata={"domain": domain, "novelty": 1.0, "retrieval_error": 1.0},
            )
