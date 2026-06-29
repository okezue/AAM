from __future__ import annotations
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from aamemory.schema import SparseCode, utcnowiso
@dataclass
class ProspectiveTrace:
    trace_id: str
    prompt: str
    query_address: SparseCode
    completed_address: SparseCode | None = None
    provenance: str = "self_simulated"
    factual_authority: float = 0.0
    retrieval_authority: float = 0.10
    created_at: str = field(default_factory=utcnowiso)
    promoted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def todict(self) -> dict[str, Any]:
        data = asdict(self)
        data["query_address"] = self.query_address.tojsonable()
        data["completed_address"] = self.completed_address.tojsonable() if self.completed_address else None
        data["metadata"] = dict(self.metadata)
        return data
class FuturePromptSimulator:
    def __init__(self, *, enabled: bool = False, maxprompts: int = 8, authority: float = 0.10) -> None:
        self.enabled = bool(enabled)
        self.maxprompts = int(maxprompts)
        self.authority = float(authority)
        self.traces: dict[str, ProspectiveTrace] = {}
        self.created = 0
        self.promoted = 0
    @classmethod
    def fromconfig(cls, config: Mapping[str, Any] | None = None) -> FuturePromptSimulator:
        config = dict(config or {})
        return cls(
            enabled=bool(config.get("enabled", False)),
            maxprompts=int(config.get("maxprompts", 8)),
            authority=float(config.get("authority", 0.10)),
        )
    def proposeprompts(self, *, recent_query: str | None = None, task_state: Mapping[str, Any] | None = None) -> list[str]:
        if not self.enabled:
            return []
        task_state = dict(task_state or {})
        prompts: list[str] = []
        if recent_query:
            prompts.append(f"Follow-up: {recent_query}")
            prompts.append(f"What evidence would verify the answer to: {recent_query}")
        entity = task_state.get("entity") or task_state.get("topic")
        if entity:
            prompts.append(f"What changed most recently about {entity}?")
            prompts.append(f"Which previous memory conflicts with {entity}?")
        prompts.append("Which future task is likely to need the most salient recent engram?")
        return prompts[: self.maxprompts]
    def simulate(
        self,
        prompts: Iterable[str],
        *,
        encodequery: Callable[[str, Mapping[str, Any] | None], SparseCode],
        complete: Callable[[SparseCode, Mapping[str, Any] | None], SparseCode | None] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[ProspectiveTrace]:
        if not self.enabled:
            return []
        traces: list[ProspectiveTrace] = []
        for prompt in list(prompts)[: self.maxprompts]:
            q = encodequery(prompt, {**dict(metadata or {}), "hypothetical": True})
            completed = complete(q, metadata) if complete else None
            trace_id = f"prospective:{self.created}"
            self.created += 1
            trace = ProspectiveTrace(
                trace_id=trace_id,
                prompt=prompt,
                query_address=q,
                completed_address=completed,
                retrieval_authority=self.authority,
                metadata={**dict(metadata or {}), "factual": False},
            )
            self.traces[trace_id] = trace
            traces.append(trace)
        return traces
    def promoteifmatched(self, prompt: str, *, threshold: float = 0.95) -> bool:
        for trace in self.traces.values():
            if trace.prompt == prompt and not trace.promoted and threshold <= 1.0:
                trace.promoted = True
                trace.factual_authority = 0.0
                self.promoted += 1
                return True
        return False
    def stats(self) -> dict[str, int | float]:
        return {
            "prospective_traces": len(self.traces),
            "created": self.created,
            "promoted": self.promoted,
            "authority": self.authority,
        }
    def statedict(self) -> dict[str, Any]:
        return {"traces": {tid: trace.todict() for tid, trace in self.traces.items()}, "created": self.created, "promoted": self.promoted}
