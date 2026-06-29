from __future__ import annotations
import math
from datetime import datetime, timezone
from aamemory.config import RetrievalConfig
from aamemory.memory.associations import RecallTrace, SparseAssociationGraph
from aamemory.memory.store import EpisodeStore
from aamemory.schema import QueryResult, SparseCode
def recencyscore(timestamp: str, half_life_days: float = 30.0) -> float:
    try:
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - value).total_seconds() / 86400.0)
        return 2.0 ** (-age_days / max(half_life_days, 1e-6))
    except (TypeError, ValueError):
        return 0.0
class AssociativeRetriever:
    def __init__(
        self,
        store: EpisodeStore,
        graph: SparseAssociationGraph,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.config = config or RetrievalConfig()
    def retrieve(self, query: SparseCode) -> tuple[list[QueryResult], RecallTrace]:
        trace = self.graph.recall(query, self.config)
        candidate_features = set(query.indices) | set(trace.final.indices) | set(trace.temporal.indices)
        candidates = self.store.candidates(
            candidate_features, limit=self.config.candidatelimit
        )
        if not candidates and len(self.store) <= self.config.candidatelimit:
            candidates = list(self.store.all())
        results: list[QueryResult] = []
        for episode in candidates:
            exact = episode.code.dot(query)
            associative = episode.code.dot(trace.final)
            temporal = episode.code.dot(trace.temporal)
            recency = recencyscore(episode.timestamp)
            score = (
                self.config.exactweight * exact
                + self.config.associativeweight * associative
                + self.config.temporalweight * temporal
                + self.config.recencyweight * recency
                + self.config.confidenceweight * episode.confidence
            )
            if not math.isfinite(score):
                continue
            results.append(
                QueryResult(
                    episode_id=episode.episode_id,
                    score=float(score),
                    exact_score=float(exact),
                    associative_score=float(associative),
                    temporal_score=float(temporal),
                    recency_score=float(recency),
                    episode=episode,
                    trace={
                        "query_overlap": episode.code.overlap(query),
                        "completed_overlap": episode.code.overlap(trace.final),
                    },
                )
            )
        results.sort(key=lambda result: (-result.score, result.episode_id))
        return results[: self.config.topk], trace
