from __future__ import annotations
import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class LoCoMoDataset(BenchmarkDataset):
    DEFAULT_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        cachedir: str | Path = "data/locomo",
        url: str = DEFAULT_URL,
    ) -> None:
        self.path = Path(path) if path else None
        self.cachedir = Path(cachedir)
        self.url = url
    def resolve(self) -> Path:
        if self.path:
            return self.path
        destination = self.cachedir / "locomo10.json"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.url, destination)
        return destination
    def __iter__(self) -> Iterator[BenchmarkExample]:
        samples = json.loads(self.resolve().read_text(encoding="utf-8"))
        for sample_index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", sample_index))
            conversation = sample.get("conversation", {})
            events: list[MemoryEvent] = []
            dialog_to_event: dict[str, str] = {}
            session_keys = sorted(
                (
                    key
                    for key, value in conversation.items()
                    if key.startswith("session_") and isinstance(value, list)
                ),
                key=lambda key: (
                    (0, int(key.split("_")[-1]))
                    if key.split("_")[-1].isdigit()
                    else (1, key)
                ),
            )
            for session_key in session_keys:
                date = conversation.get(f"{session_key}_date_time")
                for turn_index, turn in enumerate(conversation[session_key]):
                    dialog_id = str(turn.get("dia_id", f"{session_key}:{turn_index}"))
                    event_id = f"{sample_id}:{dialog_id}"
                    speaker = turn.get("speaker", turn.get("role", "speaker"))
                    text = str(turn.get("text", turn.get("content", "")))
                    rendered = f"{speaker}: {text}"
                    events.append(
                        MemoryEvent(
                            event_id=event_id,
                            text=rendered,
                            timestamp=str(date) if date else None,
                            source=SourceRef.fortext(
                                rendered,
                                document_id=event_id,
                                uri=f"locomo://{sample_id}/{dialog_id}",
                            ),
                            metadata={
                                "session": session_key,
                                "dialog_id": dialog_id,
                                "stream_id": str(sample.get("sample_id", sample_index)),
                            },
                        )
                    )
                    dialog_to_event[dialog_id] = event_id
            for qa_index, qa in enumerate(sample.get("qa", [])):
                evidence_raw = qa.get("evidence", [])
                if isinstance(evidence_raw, str):
                    evidence_raw = [evidence_raw]
                evidence_ids = [dialog_to_event.get(str(item), str(item)) for item in evidence_raw]
                answer = qa.get("answer", "")
                answers = answer if isinstance(answer, list) else [answer]
                yield BenchmarkExample.build(
                    example_id=f"{sample_id}:qa:{qa_index}",
                    task=f"locomo_category_{qa.get('category', 'unknown')}",
                    events=events,
                    query=str(qa.get("question", "")),
                    answers=answers,
                    evidence_ids=evidence_ids,
                    metadata={
                        "dataset": "LoCoMo",
                        "sample_id": sample_id,
                        "category": qa.get("category"),
                    },
                )
