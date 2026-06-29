from __future__ import annotations
from dataclasses import dataclass
from aamemory.memory.store import EpisodeStore
from aamemory.schema import Episode, SourceRef, SparseCode
@dataclass(frozen=True)
class ConsolidationReport:
    considered: int
    merged_groups: int
    removed_episodes: int
    created_prototypes: int
class PrototypeConsolidator:
    def __init__(self, store: EpisodeStore) -> None:
        self.store = store
    @staticmethod
    def averagecode(episodes: list[Episode], topk: int) -> SparseCode:
        dimension = episodes[0].code.dimension
        values: dict[int, float] = {}
        total_weight = 0.0
        for episode in episodes:
            weight = max(episode.salience * episode.confidence, 1e-6)
            total_weight += weight
            for index, value in zip(episode.code.indices, episode.code.values, strict=True):
                values[index] = values.get(index, 0.0) + weight * value
        averaged = {index: value / total_weight for index, value in values.items()}
        return SparseCode.frommapping(dimension, averaged).topk(topk).normalized()
    def run(
        self,
        *,
        similaritythreshold: float = 0.9,
        minimumgroupsize: int = 2,
        topk: int = 128,
        delete_members: bool = True,
    ) -> ConsolidationReport:
        episodes = list(self.store.all())
        remaining = set(e.episode_id for e in episodes)
        groups: list[list[Episode]] = []
        by_id = {e.episode_id: e for e in episodes}
        for episode in episodes:
            if episode.episode_id not in remaining:
                continue
            group = [episode]
            remaining.remove(episode.episode_id)
            for other_id in list(remaining):
                other = by_id[other_id]
                if episode.code.dot(other.code) >= similaritythreshold:
                    group.append(other)
                    remaining.remove(other_id)
            if len(group) >= minimumgroupsize:
                groups.append(group)
        removed = created = 0
        for group in groups:
            prototype_id = "prototype:" + "+".join(e.episode_id for e in group)
            prototype = Episode(
                episode_id=prototype_id,
                text="\n\n".join(e.text for e in group),
                code=self.averagecode(group, topk),
                timestamp=max(e.timestamp for e in group),
                salience=max(e.salience for e in group),
                confidence=sum(e.confidence for e in group) / len(group),
                source=SourceRef(
                    metadata={"member_source_checksums": [e.source.checksum for e in group]}
                ),
                metadata={
                    "kind": "consolidated_prototype",
                    "member_ids": [e.episode_id for e in group],
                },
            )
            self.store.add(prototype)
            created += 1
            if delete_members:
                for member in group:
                    removed += int(self.store.delete(member.episode_id))
        return ConsolidationReport(len(episodes), len(groups), removed, created)
