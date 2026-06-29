from __future__ import annotations
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class RulerDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        repoid: str = "self-long/RULER-llama3-1M",
        configname: str = "4k",
        split: str = "train",
        revision: str | None = None,
        cachedir: str | None = None,
    ) -> None:
        self.repoid = repoid
        self.configname = configname
        self.split = split
        self.revision = revision
        self.cachedir = cachedir
    def __iter__(self) -> Iterator[BenchmarkExample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("RULER requires `pip install -e .[hf]`") from exc
        dataset = load_dataset(
            self.repoid,
            self.configname,
            split=self.split,
            revision=self.revision,
            cache_dir=self.cachedir,
        )
        for index, row in enumerate(dataset):
            prompt = str(row.get("input", row.get("prompt", row.get("context", ""))))
            query = str(row.get("query", row.get("question", "")))
            event_id = f"ruler:{self.configname}:{index}:context"
            outputs = row.get("outputs", row.get("output", row.get("answer", [])))
            if not isinstance(outputs, list):
                outputs = [outputs]
            yield BenchmarkExample.build(
                example_id=f"ruler:{self.configname}:{index}",
                task=str(row.get("task", row.get("category", "ruler"))),
                events=[
                    MemoryEvent(
                        event_id=event_id,
                        text=prompt,
                        source=SourceRef.fortext(
                            prompt, document_id=event_id, uri=f"ruler://{self.configname}/{index}"
                        ),
                    )
                ],
                query=query or "Answer the task contained in the stored RULER prompt.",
                answers=[str(value) for value in outputs],
                evidence_ids=[event_id],
                metadata={"dataset": self.repoid, "configname": self.configname},
            )
