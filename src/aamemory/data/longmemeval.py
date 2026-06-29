from __future__ import annotations
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class LongMemEvalDataset(BenchmarkDataset):
    OFFICIAL_FILES = {
        "oracle": "longmemeval_oracle.json",
        "s": "longmemeval_s_cleaned.json",
        "m": "longmemeval_m_cleaned.json",
    }
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        variant: str = "s",
        repoid: str = "xiaowu0162/LongMemEval",
        revision: str = "main",
        cachedir: str | Path | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.variant = variant
        self.repoid = repoid
        self.revision = revision
        self.cachedir = str(cachedir) if cachedir else None
    def resolve(self) -> Path:
        if self.path:
            return self.path
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("LongMemEval download requires `pip install -e .[hf]`") from exc
        filename = self.OFFICIAL_FILES.get(self.variant, self.variant)
        return Path(
            hf_hub_download(
                repo_id=self.repoid,
                filename=filename,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cachedir,
            )
        )
    @staticmethod
    def turntext(turn: dict[str, Any]) -> str:
        role = str(turn.get("role", "unknown"))
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        return f"{role}: {content}"
    def __iter__(self) -> Iterator[BenchmarkExample]:
        raw = json.loads(self.resolve().read_text())
        records = raw if isinstance(raw, list) else raw.get("data", raw.get("examples", []))
        for index, record in enumerate(records):
            question_id = str(record.get("question_id", index))
            session_ids = record.get("haystack_session_ids", [])
            dates = record.get("haystack_dates", [])
            sessions = record.get("haystack_sessions", [])
            events: list[MemoryEvent] = []
            for session_index, session in enumerate(sessions):
                raw_session_id = str(
                    session_ids[session_index]
                    if session_index < len(session_ids)
                    else f"session:{session_index}"
                )
                session_id = f"{question_id}:{raw_session_id}"
                date = dates[session_index] if session_index < len(dates) else None
                turns = session.get("turns", session) if isinstance(session, dict) else session
                text = "\n".join(self.turntext(turn) for turn in turns)
                events.append(
                    MemoryEvent(
                        event_id=session_id,
                        text=text,
                        timestamp=str(date) if date else None,
                        source=SourceRef.fortext(
                            text,
                            document_id=session_id,
                            uri=f"longmemeval://{question_id}/{raw_session_id}",
                        ),
                        metadata={
                            "session_id": raw_session_id,
                            "date": date,
                            "stream_id": question_id,
                        },
                    )
                )
            answer = record.get("answer", "")
            answers = answer if isinstance(answer, list) else [answer]
            yield BenchmarkExample.build(
                example_id=question_id,
                task=str(record.get("question_type", "longmemeval")),
                events=events,
                query=str(record.get("question", "")),
                answers=answers,
                evidence_ids=[
                    f"{question_id}:{session_id}"
                    for session_id in record.get("answer_session_ids", [])
                ],
                metadata={
                    "question_date": record.get("question_date"),
                    "dataset": "LongMemEval",
                    "variant": self.variant,
                },
            )
