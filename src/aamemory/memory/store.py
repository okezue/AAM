from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any
from aamemory.schema import Episode
class EpisodeStore(ABC):
    @abstractmethod
    def add(self, episode: Episode) -> None:
        raise NotImplementedError
    @abstractmethod
    def get(self, episode_id: str) -> Episode | None:
        raise NotImplementedError
    @abstractmethod
    def delete(self, episode_id: str) -> bool:
        raise NotImplementedError
    @abstractmethod
    def candidates(self, feature_indices: Iterable[int], limit: int = 500) -> list[Episode]:
        raise NotImplementedError
    @abstractmethod
    def all(self) -> Iterator[Episode]:
        raise NotImplementedError
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
    def close(self) -> None:
        return None
class InMemoryEpisodeStore(EpisodeStore):
    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._postings: dict[int, set[str]] = defaultdict(set)
    def add(self, episode: Episode) -> None:
        old = self._episodes.get(episode.episode_id)
        if old is not None:
            for feature in old.code.indices:
                self._postings[feature].discard(old.episode_id)
        self._episodes[episode.episode_id] = episode
        for feature in episode.code.indices:
            self._postings[feature].add(episode.episode_id)
    def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)
    def delete(self, episode_id: str) -> bool:
        episode = self._episodes.pop(episode_id, None)
        if episode is None:
            return False
        for feature in episode.code.indices:
            self._postings[feature].discard(episode_id)
            if not self._postings[feature]:
                del self._postings[feature]
        return True
    def candidates(self, feature_indices: Iterable[int], limit: int = 500) -> list[Episode]:
        votes: dict[str, int] = defaultdict(int)
        for feature in feature_indices:
            for episode_id in self._postings.get(int(feature), ()):
                votes[episode_id] += 1
        ordered = sorted(votes, key=lambda eid: (-votes[eid], eid))[:limit]
        return [self._episodes[eid] for eid in ordered]
    def all(self) -> Iterator[Episode]:
        yield from self._episodes.values()
    def __len__(self) -> int:
        return len(self._episodes)
    def clear(self) -> None:
        self._episodes.clear()
        self._postings.clear()
    def stats(self) -> dict[str, Any]:
        return {
            "episodes": len(self._episodes),
            "posting_features": len(self._postings),
            "posting_entries": sum(len(v) for v in self._postings.values()),
        }
