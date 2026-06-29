from __future__ import annotations
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class LongMemEvalV2Dataset(BenchmarkDataset):
    def __init__(self, *, path: str | Path) -> None:
        self.path = Path(path)
    def __iter__(self) -> Iterator[BenchmarkExample]:
        records: list[dict[str, Any]] = []
        if self.path.suffix == ".jsonl":
            records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        else:
            raw = json.loads(self.path.read_text())
            records = raw if isinstance(raw, list) else raw.get("data", raw.get("examples", []))
        for index, record in enumerate(records):
            question_id = str(record.get("question_id", index))
            trajectories = record.get("trajectories", record.get("memory", record.get("events")))
            if trajectories is None:
                raise ValueError("LongMemEval-V2 export is missing trajectories/memory/events")
            events: list[MemoryEvent] = []
            for event_index, event in enumerate(trajectories):
                raw_event_id = str(
                    event.get("id", event.get("trajectory_id", f"event:{event_index}"))
                )
                event_id = f"{question_id}:{raw_event_id}"
                content = event.get("content", event.get("text", event.get("messages", "")))
                if isinstance(content, list):
                    content = "\n".join(
                        f"{item.get('role', 'unknown')}: {item.get('content', item)}"
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    )
                text = str(content)
                events.append(
                    MemoryEvent(
                        event_id=event_id,
                        text=text,
                        timestamp=event.get("timestamp", event.get("date")),
                        source=SourceRef.fortext(
                            text,
                            document_id=event_id,
                            uri=f"longmemeval-v2://{question_id}/{raw_event_id}",
                        ),
                        metadata={
                            "stream_id": question_id,
                            "raw_event_id": raw_event_id,
                        },
                    )
                )
            answer = record.get("answer", record.get("reference_answer", ""))
            evidence = record.get("evidence_ids", record.get("answer_trajectory_ids", []))
            yield BenchmarkExample.build(
                example_id=question_id,
                task=str(record.get("question_type", record.get("ability", "longmemeval_v2"))),
                events=events,
                query=str(record.get("question", record.get("query", ""))),
                answers=answer if isinstance(answer, list) else [answer],
                evidence_ids=[f"{question_id}:{event_id}" for event_id in evidence],
                metadata={"dataset": "LongMemEval-V2"},
            )
