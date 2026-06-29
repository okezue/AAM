from __future__ import annotations
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class LongBenchV2Dataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        repoid: str = "THUDM/LongBench-v2",
        split: str = "train",
        revision: str | None = None,
        cachedir: str | None = None,
    ) -> None:
        self.repoid = repoid
        self.split = split
        self.revision = revision
        self.cachedir = cachedir
    def __iter__(self) -> Iterator[BenchmarkExample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("LongBench-v2 requires `pip install -e .[hf]`") from exc
        dataset = load_dataset(
            self.repoid,
            split=self.split,
            revision=self.revision,
            cache_dir=self.cachedir,
        )
        for index, row in enumerate(dataset):
            context = str(row.get("context", ""))
            event_id = f"longbench-v2:{index}:context"
            options = [
                (letter, row.get(letter))
                for letter in ("A", "B", "C", "D")
                if row.get(letter) not in {None, ""}
            ]
            query = str(row.get("question", row.get("input", "")))
            if options:
                query += "\n" + "\n".join(f"{letter}. {value}" for letter, value in options)
            answer = row.get("answer", row.get("label", ""))
            yield BenchmarkExample.build(
                example_id=str(row.get("_id", f"longbench-v2:{index}")),
                task=str(row.get("domain", row.get("task", "longbench_v2"))),
                events=[
                    MemoryEvent(
                        event_id=event_id,
                        text=context,
                        source=SourceRef.fortext(
                            context, document_id=event_id, uri=f"longbench-v2://{index}"
                        ),
                    )
                ],
                query=query,
                answers=[answer],
                evidence_ids=[event_id],
                metadata={"dataset": self.repoid, "row": index},
            )
