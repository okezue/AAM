from __future__ import annotations
import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.memory.store import EpisodeStore
from aamemory.schema import Episode
Verifier = Callable[[Episode], bool]
def sourcechecksumverifier(episode: Episode) -> bool:
    checksum = episode.source.checksum
    if checksum is None:
        return False
    return hashlib.sha256(episode.text.encode("utf-8")).hexdigest() == checksum
@dataclass(frozen=True)
class ReplayReport:
    selected: int
    verified: int
    rejected: int
    updates: int
class ReplayEngine:
    def __init__(
        self,
        store: EpisodeStore,
        graph: SparseAssociationGraph,
        *,
        verifier: Verifier = sourcechecksumverifier,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.graph = graph
        self.verifier = verifier
        self.rng = random.Random(seed)
    def select(self, budget: int, strategy: str = "salience") -> list[Episode]:
        episodes = list(self.store.all())
        if strategy == "random":
            self.rng.shuffle(episodes)
            return episodes[:budget]
        if strategy == "recency":
            return sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:budget]
        if strategy == "salience":
            return sorted(
                episodes,
                key=lambda e: (e.salience * max(e.confidence, 0.0), e.timestamp),
                reverse=True,
            )[:budget]
        if strategy == "uncertainty":
            return sorted(episodes, key=lambda e: abs(e.confidence - 0.5))[:budget]
        raise ValueError(f"unknown replay selection strategy: {strategy}")
    def run(
        self,
        *,
        budget: int,
        strategy: str = "salience",
        learningscale: float = 0.25,
        temporal: bool = False,
    ) -> ReplayReport:
        selected = self.select(budget, strategy)
        verified = rejected = updates = 0
        previous = None
        for episode in selected:
            if not self.verifier(episode):
                rejected += 1
                previous = None
                continue
            verified += 1
            self.graph.write(
                episode.code,
                salience=learningscale * episode.salience,
                previous=previous if temporal else None,
            )
            previous = episode.code
            updates += 1
        return ReplayReport(len(selected), verified, rejected, updates)
