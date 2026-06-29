from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from aamemory.config import MemoryConfig
from aamemory.memory.system import ActivationAssociativeMemory
from aamemory.schema import MemoryEvent, SourceRef
@dataclass
class AAMLongMemEvalV2Backend:
    memory_config: MemoryConfig
    def __post_init__(self) -> None:
        self.memory = ActivationAssociativeMemory(self.memory_config)
    def insert(
        self,
        memory_id: str,
        content: str | Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(
                f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in content
            )
        metadata = dict(metadata or {})
        episode = self.memory.write(
            MemoryEvent(
                event_id=str(memory_id),
                text=text,
                timestamp=metadata.get("timestamp", metadata.get("date")),
                source=SourceRef.fortext(
                    text,
                    document_id=str(memory_id),
                    uri=metadata.get("uri", f"longmemeval-v2://{memory_id}"),
                ),
                metadata=metadata,
            )
        )
        return {"id": episode.episode_id, "stored": True}
    def query(self, query: str, topk: int | None = None, **_: Any) -> list[dict[str, Any]]:
        old_top_k = self.memory.retriever.config.topk
        if topk is not None:
            self.memory.retriever.config.topk = int(topk)
        try:
            results = self.memory.query(query)
        finally:
            self.memory.retriever.config.topk = old_top_k
        return [
            {
                "id": result.episode_id,
                "text": result.episode.text,
                "score": result.score,
                "source": result.episode.source.uri,
                "metadata": result.episode.metadata,
            }
            for result in results
        ]
    def reset(self) -> None:
        self.memory.close()
        self.memory = ActivationAssociativeMemory(self.memory_config)
    def close(self) -> None:
        self.memory.close()
