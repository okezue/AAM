from __future__ import annotations
import random
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def source(text: str, eid: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=eid, uri=f"contextual://{eid}")
def event(eid: str, text: str, **metadata: object) -> MemoryEvent:
    return MemoryEvent(event_id=eid, text=text, source=source(text, eid), metadata=metadata)
class ContextualMemoryDataset(BenchmarkDataset):
    def __init__(self, *, examplespertask: int = 25, seed: int = 0) -> None:
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        speakers = ["alice", "bob", "carol", "dave"]
        tools = ["browser", "notebook", "calendar", "terminal"]
        colors = ["blue", "amber", "violet", "green"]
        for idx in range(self.examplespertask):
            rng = random.Random(self.seed + idx)
            object_name = f"artifact-{idx}"
            target_speaker = rng.choice(speakers)
            target_tool = rng.choice(tools)
            target_color = rng.choice(colors)
            events: list[MemoryEvent] = []
            evidence_id = ""
            for j, speaker in enumerate(speakers):
                color = target_color if speaker == target_speaker else rng.choice([c for c in colors if c != target_color])
                tool = target_tool if speaker == target_speaker else rng.choice(tools)
                eid = f"context:{idx}:{speaker}"
                if speaker == target_speaker:
                    evidence_id = eid
                events.append(
                    event(
                        eid,
                        f"The shared record says {object_name} has visible state color {color}.",
                        speaker=speaker,
                        tool_state={"tool": tool, "pane": f"pane-{j}"},
                        document_id=f"doc-{speaker}",
                        modality="text",
                    )
                )
            rng.shuffle(events)
            yield BenchmarkExample.build(
                example_id=f"context:{idx}",
                task="context_association",
                events=events,
                query=f"In {target_speaker}'s {target_tool} context, what color was recorded for {object_name}?",
                answers=[target_color],
                evidence_ids=[evidence_id],
                metadata={"speaker": target_speaker, "tool_state": {"tool": target_tool}, "object": object_name},
            )
