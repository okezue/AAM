from __future__ import annotations
from aamemory.data.chunked import ChunkedDataset
def testchunkeddatasetexpandseventsandevidence() -> None:
    dataset = ChunkedDataset(
        base={
            "type": "synthetic",
            "params": {
                "tasks": ["paired_associate"],
                "examplespertask": 1,
                "distractors": 0,
                "seed": 4,
            },
        },
        chunksize=20,
        overlap=5,
        unit="characters",
    )
    example = next(iter(dataset))
    assert len(example.events) > 1
    assert all(":chunk:" in event.event_id for event in example.events)
    assert example.evidence_ids
    assert all(":chunk:" in evidence_id for evidence_id in example.evidence_ids)
