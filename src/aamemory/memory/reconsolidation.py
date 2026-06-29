from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from aamemory.memory.completion import EpisodeIndexGraph
from aamemory.memory.engram import EngramVersion
from aamemory.memory.store import EpisodeStore
from aamemory.schema import QueryResult, utcnowiso
@dataclass(frozen=True)
class ReconsolidationReport:
    updated: int = 0
    superseded: int = 0
    corrected: int = 0
    depressed: int = 0
    notes: tuple[str, ...] = ()
class ReconsolidationLedger:
    def __init__(self) -> None:
        self.versions: dict[str, EngramVersion] = {}
        self.children: dict[str, list[str]] = {}
    def register(
        self,
        episode_id: str,
        *,
        status: str = "observed",
        confidence: float = 1.0,
        authority: float = 1.0,
        source_checksum: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EngramVersion:
        version = EngramVersion(
            memory_id=episode_id,
            version_id=f"{episode_id}:v1",
            status=status,
            confidence=float(confidence),
            authority=float(authority),
            source_checksum=source_checksum,
            metadata=dict(metadata or {}),
        )
        self.versions[episode_id] = version
        return version
    def supersede(
        self,
        old_id: str,
        new_id: str,
        *,
        reason: str = "correction",
    ) -> ReconsolidationReport:
        now = utcnowiso()
        old = self.versions.get(old_id)
        if old is not None:
            old.status = "superseded"
            old.last_verified_at = now
            old.metadata = {**dict(old.metadata), "superseded_by": new_id, "reason": reason}
        new = self.versions.get(new_id)
        if new is not None:
            new.status = "corrected"
            new.parent_version_id = old.version_id if old is not None else None
            new.supersedes = (*new.supersedes, old_id)
            new.last_verified_at = now
        self.children.setdefault(old_id, []).append(new_id)
        return ReconsolidationReport(updated=int(old is not None) + int(new is not None), superseded=int(old is not None), corrected=int(new is not None))
    def afteruse(
        self,
        results: list[QueryResult],
        *,
        store: EpisodeStore,
        episodeindex: EpisodeIndexGraph,
        verified: bool | None = None,
    ) -> ReconsolidationReport:
        updated = depressed = 0
        now = utcnowiso()
        for rank, result in enumerate(results):
            episode = result.episode
            version = self.versions.get(result.episode_id)
            if version is None:
                version = self.register(result.episode_id, confidence=episode.confidence, source_checksum=episode.source.checksum)
            episode.metadata["access_count"] = int(episode.metadata.get("access_count", 0)) + 1
            episode.metadata["last_access"] = now
            episode.metadata["labile_after_recall"] = True
            if verified is True or (verified is None and result.episode.source.checksum):
                episode.metadata["last_verified_at"] = now
                episode.confidence = min(2.0, float(episode.confidence) + 0.02 / (rank + 1))
                version.last_verified_at = now
                version.confidence = episode.confidence
                episodeindex.potentiate(result.episode_id, 0.01 / (rank + 1))
            elif verified is False:
                episode.confidence = max(0.0, float(episode.confidence) - 0.10)
                version.confidence = episode.confidence
                episodeindex.depress(result.episode_id, 0.05)
                depressed += 1
            store.add(episode)
            updated += 1
        return ReconsolidationReport(updated=updated, depressed=depressed)
    def statedict(self) -> dict[str, Any]:
        return {
            "versions": {eid: version.todict() for eid, version in self.versions.items()},
            "children": self.children,
        }
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        self.versions = {
            str(eid): EngramVersion.fromdict(value)
            for eid, value in state.get("versions", {}).items()
        }
        self.children = {str(k): [str(x) for x in v] for k, v in state.get("children", {}).items()}
