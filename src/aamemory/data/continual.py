from __future__ import annotations
import random
import string
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def token(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))
def event(event_id: str, text: str, *, step: int, kind: str) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        text=text,
        timestamp=f"2026-01-01T00:{step // 60:02d}:{step % 60:02d}+00:00",
        source=SourceRef.fortext(text, document_id=event_id, uri=f"continual://{event_id}"),
        metadata={"stream_id": "continual-retention", "step": step, "kind": kind},
    )
class ContinualRetentionDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        steps: int = 200,
        seed: int = 0,
        distractorsperstep: int = 2,
        probelags: tuple[int, ...] | list[int] = (0, 1, 5, 20, 50, 100),
        payloadlength: int = 20,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not probelags:
            raise ValueError("probelags must not be empty")
        self.steps = int(steps)
        self.seed = int(seed)
        self.distractorsperstep = int(distractorsperstep)
        self.probelags = tuple(max(0, int(value)) for value in probelags)
        self.payloadlength = int(payloadlength)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        rng = random.Random(self.seed)
        facts: list[tuple[str, str, str]] = []
        for step in range(self.steps):
            key = token(rng, 10)
            value = token(rng, self.payloadlength)
            fact_id = f"continual:{step}:fact"
            fact_text = f"Verified registry fact: key {key} has exact payload {value}."
            events = [event(fact_id, fact_text, step=step, kind="fact")]
            facts.append((fact_id, key, value))
            for distractor_index in range(self.distractorsperstep):
                distractor_key = token(rng, 8)
                distractor_value = token(rng, 10)
                distractor_id = f"continual:{step}:distractor:{distractor_index}"
                events.append(
                    event(
                        distractor_id,
                        f"Background registry note: {distractor_key} maps to {distractor_value}.",
                        step=step,
                        kind="distractor",
                    )
                )
            lag_requested = self.probelags[step % len(self.probelags)]
            lag = min(step, lag_requested)
            target_id, target_key, target_value = facts[step - lag]
            yield BenchmarkExample.build(
                example_id=f"continual:probe:{step}",
                task="continual_retention",
                events=events,
                query=f"Repeat the exact payload stored for key {target_key}.",
                answers=[target_value],
                evidence_ids=[target_id],
                metadata={
                    "dataset": "continual_retention",
                    "step": step,
                    "lag": lag,
                    "lag_requested": lag_requested,
                    "cluster_id": target_id,
                },
            )
