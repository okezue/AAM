from __future__ import annotations
from collections.abc import Iterator, Mapping
from typing import Any
from aamemory.data.base import BenchmarkDataset
from aamemory.schema import BenchmarkExample, MemoryEvent, SourceRef
class ChunkedDataset(BenchmarkDataset):
    def __init__(
        self,
        *,
        base: Mapping[str, Any],
        chunksize: int = 8192,
        overlap: int = 0,
        unit: str = "characters",
        tokenizername: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        if chunksize <= 0:
            raise ValueError("chunksize must be positive")
        if overlap < 0 or overlap >= chunksize:
            raise ValueError("overlap must satisfy 0 <= overlap < chunksize")
        from aamemory.data.registry import builddataset
        self.base = builddataset(base)
        self.chunksize = int(chunksize)
        self.overlap = int(overlap)
        self.unit = unit.lower().replace("-", "_")
        self.tokenizer = None
        if self.unit in {"token", "tokens"}:
            if not tokenizername:
                raise ValueError("tokenizername is required for token chunking")
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError("token chunking requires `pip install -e .[hf]`") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizername, revision=tokenizer_revision
            )
        elif self.unit not in {"character", "characters", "word", "words"}:
            raise ValueError(f"unknown chunk unit: {unit}")
    def chunks(self, text: str) -> list[tuple[str, int, int]]:
        step = self.chunksize - self.overlap
        if self.unit in {"character", "characters"}:
            return [
                (text[start : start + self.chunksize], start, min(len(text), start + self.chunksize))
                for start in range(0, max(1, len(text)), step)
                if text[start : start + self.chunksize]
            ] or [("", 0, 0)]
        if self.unit in {"word", "words"}:
            words = text.split()
            chunks: list[tuple[str, int, int]] = []
            for start in range(0, max(1, len(words)), step):
                piece = words[start : start + self.chunksize]
                if piece:
                    chunks.append((" ".join(piece), start, start + len(piece)))
            return chunks or [("", 0, 0)]
        assert self.tokenizer is not None
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for start in range(0, max(1, len(token_ids)), step):
            ids = token_ids[start : start + self.chunksize]
            if ids:
                chunks.append((self.tokenizer.decode(ids), start, start + len(ids)))
        return chunks or [("", 0, 0)]
    def __iter__(self) -> Iterator[BenchmarkExample]:
        for example in self.base:
            events: list[MemoryEvent] = []
            expanded: dict[str, list[str]] = {}
            for event in example.events:
                pieces = self.chunks(event.text)
                expanded[event.event_id] = []
                for index, (text, start, end) in enumerate(pieces):
                    chunk_id = f"{event.event_id}:chunk:{index}"
                    expanded[event.event_id].append(chunk_id)
                    events.append(
                        MemoryEvent(
                            event_id=chunk_id,
                            text=text,
                            timestamp=event.timestamp,
                            source=SourceRef.fortext(
                                text,
                                uri=event.source.uri,
                                document_id=event.source.document_id or event.event_id,
                                start=start,
                                end=end,
                                metadata={
                                    **dict(event.source.metadata),
                                    "parent_event_id": event.event_id,
                                    "chunk_index": index,
                                    "chunk_unit": self.unit,
                                },
                            ),
                            metadata={
                                **dict(event.metadata),
                                "parent_event_id": event.event_id,
                                "chunk_index": index,
                                "chunk_start": start,
                                "chunk_end": end,
                                "chunk_unit": self.unit,
                            },
                        )
                    )
            evidence = [
                chunk_id
                for evidence_id in example.evidence_ids
                for chunk_id in expanded.get(evidence_id, [evidence_id])
            ]
            negatives = [
                chunk_id
                for evidence_id in example.negative_evidence_ids
                for chunk_id in expanded.get(evidence_id, [evidence_id])
            ]
            yield BenchmarkExample.build(
                example_id=example.example_id,
                task=example.task,
                events=events,
                query=example.query,
                answers=example.answers,
                evidence_ids=evidence,
                negative_evidence_ids=negatives,
                metadata={
                    **dict(example.metadata),
                    "chunked": True,
                    "chunksize": self.chunksize,
                    "chunk_overlap": self.overlap,
                    "chunk_unit": self.unit,
                },
            )
