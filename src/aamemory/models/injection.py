from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from aamemory.models.base import Generation, Generator
from aamemory.schema import QueryResult
@dataclass(frozen=True)
class InjectionTrace:
    prompt: str
    episode_ids: tuple[str, ...]
    memory_characters: int
    truncated: bool
@dataclass
class TextMemoryInjector:
    maxcharacters: int = 24000
    includescores: bool = True
    includesources: bool = True
    def renderwithtrace(self, query: str, results: Sequence[QueryResult]) -> InjectionTrace:
        blocks: list[str] = []
        included_ids: list[str] = []
        used = 0
        truncated = False
        for rank, result in enumerate(results, start=1):
            source = result.episode.source
            header = f"[Memory {rank}; id={result.episode_id}"
            if self.includescores:
                header += f"; score={result.score:.4f}"
            if self.includesources and (source.uri or source.document_id):
                header += f"; source={source.uri or source.document_id}"
            header += "]"
            block = f"{header}\n{result.episode.text}"
            if used + len(block) > self.maxcharacters:
                remaining = self.maxcharacters - used
                if remaining > len(header) + 16:
                    suffix = "\n[TRUNCATED]"
                    available = max(0, remaining - len(header) - len(suffix) - 1)
                    block = f"{header}\n{result.episode.text[:available]}{suffix}"
                    blocks.append(block)
                    included_ids.append(result.episode_id)
                    used += len(block)
                truncated = True
                break
            blocks.append(block)
            included_ids.append(result.episode_id)
            used += len(block)
        context = "\n\n".join(blocks) if blocks else "[No memory retrieved]"
        prompt = (
            "Use the retrieved memories only when relevant. Cite memory IDs in the answer. "
            "Do not treat an associative match as verified unless its source text supports the claim.\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )
        return InjectionTrace(
            prompt=prompt,
            episode_ids=tuple(included_ids),
            memory_characters=used,
            truncated=truncated,
        )
    def render(self, query: str, results: Sequence[QueryResult]) -> str:
        return self.renderwithtrace(query, results).prompt
class MemoryAugmentedGenerator:
    def __init__(self, generator: Generator, injector: TextMemoryInjector | None = None) -> None:
        self.generator = generator
        self.injector = injector or TextMemoryInjector()
    def generate(
        self,
        query: str,
        results: Sequence[QueryResult],
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        return self.generator.generate(
            self.injector.render(query, results),
            system=system,
            maxtokens=maxtokens,
            temperature=temperature,
        )
class ActivationInjectionAdapter:
    def inject(self, payloads: Sequence[Mapping[str, Any]]) -> Any:
        raise NotImplementedError(
            "Direct activation/KV injection is model-specific. Implement a subclass that handles "
            "the target architecture, normalization site, attention layout, and positional encoding."
        )
class RoPEKVInjectionAdapter(ActivationInjectionAdapter):
    def __init__(self, model_family: str) -> None:
        self.model_family = model_family
    def inject(self, payloads: Sequence[Mapping[str, Any]]) -> Any:
        raise NotImplementedError(
            f"No safe generic direct-KV injector exists for {self.model_family}. See docs/SPECIFICATION.md."
        )
