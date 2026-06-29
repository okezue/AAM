from __future__ import annotations
import json
from aamemory.data.continual import ContinualRetentionDataset
from aamemory.data.locomo import LoCoMoDataset
from aamemory.data.longmemeval import LongMemEvalDataset
from aamemory.data.synthetic import SyntheticMemoryDataset
def testsyntheticdatasethasexpectedcountandevidence() -> None:
    examples = list(
        SyntheticMemoryDataset(tasks=["paired_associate", "multi_hop"], examplespertask=2, seed=3)
    )
    assert len(examples) == 4
    assert all(example.evidence_ids for example in examples)
def testlongmemevallocalschema(tmp_path) -> None:
    path = tmp_path / "longmemeval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "What did I buy?",
                    "answer": "a lamp",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2025-01-01"],
                    "haystack_sessions": [[{"role": "user", "content": "I bought a lamp."}]],
                    "answer_session_ids": ["s1"],
                }
            ]
        )
    )
    example = next(iter(LongMemEvalDataset(path=path)))
    assert example.evidence_ids == ("q1:s1",)
    assert example.events[0].event_id == "q1:s1"
    assert "lamp" in example.events[0].text
def testlocomolocalschema(tmp_path) -> None:
    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "c1",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1_date_time": "2025-01-01",
                        "session_1": [
                            {"speaker": "A", "text": "I like tea.", "dia_id": "D1:1"}
                        ],
                    },
                    "qa": [
                        {
                            "question": "What does A like?",
                            "answer": "tea",
                            "category": 1,
                            "evidence": ["D1:1"],
                        }
                    ],
                }
            ]
        )
    )
    example = next(iter(LoCoMoDataset(path=path)))
    assert example.evidence_ids == ("c1:D1:1",)
    assert example.answers == ("tea",)
def testcontinualretentionpointstopriorevents() -> None:
    examples = list(
        ContinualRetentionDataset(
            steps=4,
            seed=2,
            distractorsperstep=0,
            probelags=[0, 1],
        )
    )
    assert examples[1].metadata["lag"] == 1
    assert examples[1].evidence_ids == (examples[0].events[0].event_id,)
    all_event_ids = [event.event_id for example in examples for event in example.events]
    assert len(all_event_ids) == len(set(all_event_ids))
