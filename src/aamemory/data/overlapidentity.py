from __future__ import annotations
import random
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
def source(text: str, eid: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=eid, uri=f"overlap://{eid}")
class OverlapIdentityDataset(BenchmarkDataset):
    def __init__(self, *, examplespertask: int = 25, seed: int = 0, overlapcount: int = 6) -> None:
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
        self.overlapcount = int(overlapcount)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        colors = ["blue", "amber", "violet", "green", "silver", "crimson"]
        rooms = ["lab", "studio", "archive", "hangar", "atrium", "vault"]
        for idx in range(self.examplespertask):
            rng = random.Random(self.seed + idx)
            base_entity = f"unit-{idx}"
            target = rng.randrange(self.overlapcount)
            target_color = colors[target % len(colors)]
            events: list[MemoryEvent] = []
            negative: list[str] = []
            evidence_id = f"overlap:{idx}:{target}"
            for j in range(self.overlapcount):
                color = colors[j % len(colors)]
                room = rooms[j % len(rooms)]
                eid = f"overlap:{idx}:{j}"
                text = (
                    f"Memory capsule for {base_entity} includes shared features: operator, handle, status, marker. "
                    f"In room {room}, slot {j} binds the hidden payload color {color}."
                )
                metadata = {
                    "room": room,
                    "slot": j,
                    "hidden_state_id": eid,
                    "modality": "pseudo_multimodal_state",
                    "state_tensor": [idx, j, len(color), len(room)],
                }
                events.append(MemoryEvent(event_id=eid, text=text, source=source(text, eid), metadata=metadata))
                if j != target:
                    negative.append(eid)
            rng.shuffle(events)
            yield BenchmarkExample.build(
                example_id=f"overlap:{idx}",
                task="overlap_identity",
                events=events,
                query=f"For {base_entity} in slot {target}, what hidden payload color is bound?",
                answers=[target_color],
                evidence_ids=[evidence_id],
                negative_evidence_ids=negative,
                metadata={"slot": target, "room": rooms[target % len(rooms)], "hidden_state_id": evidence_id},
            )
