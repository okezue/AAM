from __future__ import annotations
from dataclasses import dataclass
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.memory.completion import EpisodeIndexGraph
from aamemory.memory.store import EpisodeStore
from aamemory.schema import Episode
@dataclass(frozen=True)
class VerifiedReplayReport:
    candidates: int
    verified: int
    quarantined: int
    potentiated: int
    depressed: int
class SourceVerifiedReplayEngine:
    def __init__(self, store: EpisodeStore, graph: SparseAssociationGraph, episodeindex: EpisodeIndexGraph) -> None:
        self.store = store
        self.graph = graph
        self.episodeindex = episodeindex
        self.quarantined_ids: set[str] = set()
        self.cycles = 0
    def select(self, budget: int, strategy: str = "salience") -> list[Episode]:
        episodes = list(self.store.all())
        if strategy == "low_confidence":
            episodes.sort(key=lambda e: (e.confidence, -e.salience, e.episode_id))
        else:
            episodes.sort(key=lambda e: (-e.salience, e.confidence, e.episode_id))
        return episodes[: max(0, int(budget))]
    def run(self, *, budget: int = 32, learningscale: float = 0.15, strategy: str = "salience") -> VerifiedReplayReport:
        self.cycles += 1
        candidates = self.select(budget, strategy=strategy)
        verified = quarantined = potentiated = depressed = 0
        previous = None
        for episode in candidates:
            status = str(episode.metadata.get("engram_status", "observed"))
            generated = bool(episode.metadata.get("generated") or episode.metadata.get("hypothetical"))
            if generated or status in {"hypothetical", "rejected"} or not episode.source.checksum:
                self.quarantined_ids.add(episode.episode_id)
                self.episodeindex.depress(episode.episode_id, learningscale)
                quarantined += 1
                depressed += 1
                continue
            verified += 1
            self.graph.write(
                episode.code,
                salience=max(0.01, episode.salience * learningscale),
                previous=previous.code if previous is not None else None,
            )
            self.episodeindex.potentiate(episode.episode_id, 0.02)
            potentiated += 1
            previous = episode
        return VerifiedReplayReport(len(candidates), verified, quarantined, potentiated, depressed)
    def hallucinatedamplificationrate(self, before_false_edge_mass: float, after_false_edge_mass: float) -> float:
        if before_false_edge_mass <= 0:
            return float(after_false_edge_mass > 0)
        return max(0.0, (after_false_edge_mass - before_false_edge_mass) / before_false_edge_mass)
