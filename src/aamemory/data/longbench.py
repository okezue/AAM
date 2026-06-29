from __future__ import annotations
from collections.abc import Iterator
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class LongBenchDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        repoid: str = "THUDM/LongBench",
        subsets: list[str] | tuple[str, ...] = ("narrativeqa",),
        split: str = "test",
        revision: str | None = None,
        cachedir: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.repoid = repoid
        self.subsets = tuple(subsets)
        self.split = split
        self.revision = revision
        self.cachedir = cachedir
        self.trust_remote_code = trust_remote_code
    def __iter__(self) -> Iterator[BenchmarkExample]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("LongBench requires `pip install -e .[hf]`") from exc
        for subset in self.subsets:
            dataset = load_dataset(
                self.repoid,
                subset,
                split=self.split,
                revision=self.revision,
                cache_dir=self.cachedir,
                trust_remote_code=self.trust_remote_code,
            )
            for index, row in enumerate(dataset):
                context = str(row.get("context", ""))
                event_id = f"{subset}:{index}:context"
                answers = row.get("answers", row.get("answer", []))
                if not isinstance(answers, list):
                    answers = [answers]
                yield BenchmarkExample.build(
                    example_id=f"{subset}:{index}",
                    task=subset,
                    events=[
                        MemoryEvent(
                            event_id=event_id,
                            text=context,
                            source=SourceRef.fortext(
                                context,
                                document_id=event_id,
                                uri=f"longbench://{subset}/{index}",
                            ),
                        )
                    ],
                    query=str(row.get("input", row.get("question", ""))),
                    answers=answers,
                    evidence_ids=[event_id],
                    metadata={
                        "dataset": self.repoid,
                        "subset": subset,
                        "length": row.get("length"),
                        "all_classes": row.get("all_classes"),
                    },
                )
