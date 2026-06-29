from __future__ import annotations
import random
import string
from collections.abc import Callable, Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
_COLORS = ["blue", "amber", "violet", "green", "silver", "crimson"]
_NAMES = [
    "Aster",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fir",
    "Garnet",
    "Hazel",
    "Iris",
    "Juniper",
]
def source(text: str, document_id: str) -> SourceRef:
    return SourceRef.fortext(text, document_id=document_id, uri=f"synthetic://{document_id}")
def event(event_id: str, text: str, *, timestamp: str | None = None, **metadata: object) -> MemoryEvent:
    return MemoryEvent(
        event_id=event_id,
        text=text,
        timestamp=timestamp,
        source=source(text, event_id),
        metadata=metadata,
    )
def randomtoken(rng: random.Random, length: int = 16) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))
class SyntheticMemoryDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        tasks: tuple[str, ...] | list[str] = (
            "paired_associate",
            "paraphrase",
            "multi_hop",
            "temporal_update",
            "interference",
            "hub_distractor",
            "random_string",
            "replay_poison",
        ),
        examplespertask: int = 25,
        seed: int = 0,
        distractors: int = 8,
    ) -> None:
        self.tasks = tuple(tasks)
        self.examplespertask = int(examplespertask)
        self.seed = int(seed)
        self.distractors = int(distractors)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        generators: dict[str, Callable[[random.Random, int], BenchmarkExample]] = {
            "paired_associate": self.pairedassociate,
            "paraphrase": self.paraphrase,
            "multi_hop": self.multihop,
            "temporal_update": self.temporalupdate,
            "interference": self.interference,
            "hub_distractor": self.hubdistractor,
            "random_string": self.randomstring,
            "replay_poison": self.replaypoison,
        }
        for task_offset, task in enumerate(self.tasks):
            if task not in generators:
                raise ValueError(f"unknown synthetic task: {task}")
            for index in range(self.examplespertask):
                rng = random.Random(self.seed + task_offset * 1_000_003 + index)
                yield generators[task](rng, index)
    def distractorevents(self, rng: random.Random, prefix: str) -> list[MemoryEvent]:
        events: list[MemoryEvent] = []
        for index in range(self.distractors):
            name = rng.choice(_NAMES)
            item = rng.choice(["notebook", "lantern", "compass", "mug", "ticket"])
            number = rng.randrange(100, 999)
            text = f"{name} placed a {item} in locker {number}."
            events.append(event(f"{prefix}:d{index}", text, stream_id=prefix))
        return events
    def pairedassociate(self, rng: random.Random, index: int) -> BenchmarkExample:
        key = randomtoken(rng, 8)
        value = str(rng.randrange(10_000, 99_999))
        eid = f"paired:{index}:answer"
        text = f"The code-name {key} maps to the registry value {value}."
        events = self.distractorevents(rng, f"paired:{index}")
        events.insert(rng.randrange(len(events) + 1), event(eid, text, user_importance=0.5))
        return BenchmarkExample.build(
            example_id=f"paired:{index}",
            task="paired_associate",
            events=events,
            query=f"Which registry value is assigned to code-name {key}?",
            answers=[value],
            evidence_ids=[eid],
            metadata={"entropy_class": "low_to_medium"},
        )
    def paraphrase(self, rng: random.Random, index: int) -> BenchmarkExample:
        person = rng.choice(_NAMES)
        color = rng.choice(_COLORS)
        eid = f"paraphrase:{index}:answer"
        text = f"{person} enjoys the color {color} more than every other hue."
        events = self.distractorevents(rng, f"paraphrase:{index}") + [event(eid, text)]
        rng.shuffle(events)
        return BenchmarkExample.build(
            example_id=f"paraphrase:{index}",
            task="paraphrase",
            events=events,
            query=f"What is {person}'s favourite colour?",
            answers=[color],
            evidence_ids=[eid],
        )
    def multihop(self, rng: random.Random, index: int) -> BenchmarkExample:
        a, b, c = rng.sample(_NAMES, 3)
        answer = rng.choice(_COLORS)
        ids = [f"multihop:{index}:h{hop}" for hop in range(1, 4)]
        path = [
            event(ids[0], f"{a} carries the bridge-key that points to {b}."),
            event(ids[1], f"{b} carries the bridge-key that points to {c}."),
            event(ids[2], f"The final marker attached to {c} reads {answer}."),
        ]
        events = self.distractorevents(rng, f"multihop:{index}") + path
        distractors = events[:-3]
        rng.shuffle(distractors)
        events = distractors[: len(distractors) // 2] + path + distractors[len(distractors) // 2 :]
        return BenchmarkExample.build(
            example_id=f"multihop:{index}",
            task="multi_hop",
            events=events,
            query=f"Following all bridge-keys starting from {a}, what does the final marker read?",
            answers=[answer],
            evidence_ids=ids,
            metadata={"hops": 3},
        )
    def temporalupdate(self, rng: random.Random, index: int) -> BenchmarkExample:
        person = rng.choice(_NAMES)
        old, new = rng.sample(_COLORS, 2)
        old_id = f"temporal:{index}:old"
        new_id = f"temporal:{index}:new"
        events = [
            event(
                old_id,
                f"On 2025-01-01, {person}'s project status color was {old}.",
                timestamp="2025-01-01T12:00:00+00:00",
            ),
            *self.distractorevents(rng, f"temporal:{index}"),
            event(
                new_id,
                f"On 2025-06-01, {person}'s project status color changed to {new}; this supersedes earlier records.",
                timestamp="2025-06-01T12:00:00+00:00",
                task_relevance=0.5,
            ),
        ]
        return BenchmarkExample.build(
            example_id=f"temporal:{index}",
            task="temporal_update",
            events=events,
            query=f"According to the latest record, what is {person}'s project status color?",
            answers=[new],
            evidence_ids=[new_id],
            negative_evidence_ids=[old_id],
        )
    def interference(self, rng: random.Random, index: int) -> BenchmarkExample:
        entity = rng.choice(_NAMES)
        target = rng.choice(_COLORS)
        answer_id = f"interference:{index}:answer"
        events = []
        for j, color in enumerate(c for c in _COLORS if c != target):
            events.append(
                event(
                    f"interference:{index}:near{j}",
                    f"In the simulation copy {j}, {entity}-{j} has badge color {color}.",
                )
            )
        events.append(
            event(
                answer_id,
                f"In the canonical record, the exact entity {entity} has badge color {target}.",
                task_relevance=0.75,
            )
        )
        rng.shuffle(events)
        return BenchmarkExample.build(
            example_id=f"interference:{index}",
            task="interference",
            events=events,
            query=f"What badge color belongs to the exact canonical entity {entity}?",
            answers=[target],
            evidence_ids=[answer_id],
        )
    def hubdistractor(self, rng: random.Random, index: int) -> BenchmarkExample:
        rare = randomtoken(rng, 7)
        answer = rng.choice(_COLORS)
        answer_id = f"hub:{index}:answer"
        events = [
            event(
                f"hub:{index}:d{j}",
                f"The common hub concept relates item {j} to ordinary value {rng.randrange(100)}.",
            )
            for j in range(max(20, self.distractors * 3))
        ]
        events.append(
            event(answer_id, f"Rare key {rare} is associated with marker {answer}.")
        )
        rng.shuffle(events)
        return BenchmarkExample.build(
            example_id=f"hub:{index}",
            task="hub_distractor",
            events=events,
            query=f"Which marker is associated with rare key {rare}?",
            answers=[answer],
            evidence_ids=[answer_id],
            metadata={"tests": "degree_normalization_and_hub_penalty"},
        )
    def randomstring(self, rng: random.Random, index: int) -> BenchmarkExample:
        key = randomtoken(rng, 10)
        answer = randomtoken(rng, 32)
        answer_id = f"random:{index}:answer"
        events = self.distractorevents(rng, f"random:{index}")
        events.append(event(answer_id, f"The lossless payload for {key} is exactly {answer}."))
        return BenchmarkExample.build(
            example_id=f"random:{index}",
            task="random_string",
            events=events,
            query=f"Repeat the exact lossless payload for {key}.",
            answers=[answer],
            evidence_ids=[answer_id],
            metadata={"tests": "information_bound_and_source_pointer"},
        )
    def replaypoison(self, rng: random.Random, index: int) -> BenchmarkExample:
        key = randomtoken(rng, 8)
        true_value = rng.choice(_COLORS)
        false_value = rng.choice([c for c in _COLORS if c != true_value])
        true_id = f"poison:{index}:verified"
        false_id = f"poison:{index}:unverified"
        verified = event(
            true_id,
            f"Verified source: key {key} maps to {true_value}.",
            user_importance=0.5,
        )
        unverified = MemoryEvent(
            event_id=false_id,
            text=f"Generated speculation: key {key} might map to {false_value}.",
            metadata={"generated": True, "confidence": 0.1},
        )
        return BenchmarkExample.build(
            example_id=f"poison:{index}",
            task="replay_poison",
            events=[verified, unverified],
            query=f"What verified value maps to key {key}?",
            answers=[true_value],
            evidence_ids=[true_id],
            negative_evidence_ids=[false_id],
        )
