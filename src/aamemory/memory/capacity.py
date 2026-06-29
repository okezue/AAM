from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from aamemory.memory.store import EpisodeStore
from aamemory.schema import Episode
@dataclass(frozen=True)
class EvictionReport:
    policy: str
    before: int
    after: int
    evicted_ids: tuple[str, ...]
@dataclass(frozen=True)
class ByteEvictionReport:
    policy: str
    maximum_bytes: int
    before_bytes: int
    after_bytes: int
    before_episodes: int
    after_episodes: int
    evicted_ids: tuple[str, ...]
class CapacityManager:
    def __init__(self, store: EpisodeStore) -> None:
        self.store = store
    @staticmethod
    def timestamp(episode: Episode) -> str:
        return episode.timestamp or ""
    def ordered(self, episodes: list[Episode], policy: str) -> list[Episode]:
        policy = policy.lower()
        if policy == "fifo":
            return sorted(episodes, key=lambda e: (self.timestamp(e), e.episode_id))
        if policy == "lru":
            return sorted(
                episodes,
                key=lambda e: (
                    str(e.metadata.get("last_access", e.timestamp)),
                    int(e.metadata.get("access_count", 0)),
                    e.episode_id,
                ),
            )
        if policy in {"salience", "utility"}:
            return sorted(
                episodes,
                key=lambda e: (
                    e.salience * e.confidence * (1 + int(e.metadata.get("access_count", 0))),
                    self.timestamp(e),
                    e.episode_id,
                ),
            )
        raise ValueError(f"unknown capacity policy: {policy}")
    def enforce(self, maximum: int, *, policy: str = "salience") -> EvictionReport:
        before = len(self.store)
        if maximum <= 0 or before <= maximum:
            return EvictionReport(policy, before, before, ())
        episodes = list(self.store.all())
        policy = policy.lower()
        ordered = self.ordered(episodes, policy)
        remove = ordered[: before - maximum]
        removed: list[str] = []
        for episode in remove:
            if self.store.delete(episode.episode_id):
                removed.append(episode.episode_id)
        return EvictionReport(policy, before, len(self.store), tuple(removed))
    def enforcebytes(
        self,
        maximum_bytes: int,
        *,
        bytesize: Callable[[Episode], int],
        policy: str = "salience",
    ) -> ByteEvictionReport:
        episodes = list(self.store.all())
        sizes = {episode.episode_id: int(bytesize(episode)) for episode in episodes}
        before_bytes = sum(sizes.values())
        before_episodes = len(episodes)
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes must be non-negative")
        if before_bytes <= maximum_bytes:
            return ByteEvictionReport(
                policy, maximum_bytes, before_bytes, before_bytes, before_episodes, before_episodes, ()
            )
        ordered = self.ordered(episodes, policy)
        remaining = before_bytes
        removed: list[str] = []
        for episode in ordered:
            if remaining <= maximum_bytes:
                break
            if self.store.delete(episode.episode_id):
                removed.append(episode.episode_id)
                remaining -= sizes[episode.episode_id]
        return ByteEvictionReport(
            policy,
            maximum_bytes,
            before_bytes,
            max(0, remaining),
            before_episodes,
            len(self.store),
            tuple(removed),
        )
