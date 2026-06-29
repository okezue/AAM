from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from aamemory.config import RetrievalConfig
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.memory.contextgraph import ContextAssociationGraph
from aamemory.schema import SparseCode
@dataclass(frozen=True)
class CompletionTrace:
    final: SparseCode
    temporal: SparseCode
    episode_scores: Mapping[str, float]
    steps: tuple[dict[int, float], ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
class EpisodeIndexGraph:
    def __init__(self, *, maxfeaturesperepisode: int = 192, decay: float = 0.0) -> None:
        self.maxfeaturesperepisode = int(maxfeaturesperepisode)
        self.decay_rate = float(decay)
        self.episode_to_features: dict[str, dict[int, float]] = {}
        self.feature_to_episodes: dict[int, dict[str, float]] = defaultdict(dict)
        self.authority: dict[str, float] = defaultdict(lambda: 1.0)
        self.write_count = 0
    @classmethod
    def fromconfig(cls, config: Mapping[str, Any] | None = None) -> EpisodeIndexGraph:
        config = dict(config or {})
        return cls(
            maxfeaturesperepisode=int(config.get("maxfeaturesperepisode", 192)),
            decay=float(config.get("decay", 0.0)),
        )
    def clear(self) -> None:
        self.episode_to_features.clear()
        self.feature_to_episodes.clear()
        self.authority.clear()
        self.write_count = 0
    def write(
        self,
        episode_id: str,
        code: SparseCode,
        *,
        salience: float = 1.0,
        authority: float = 1.0,
    ) -> None:
        self.write_count += 1
        old = self.episode_to_features.pop(episode_id, None)
        if old is not None:
            for feature in old:
                self.feature_to_episodes[feature].pop(episode_id, None)
        pairs = sorted(
            zip(code.indices, code.values, strict=True), key=lambda item: abs(item[1]), reverse=True
        )[: self.maxfeaturesperepisode]
        row = {int(i): float(v) * float(salience) for i, v in pairs if v != 0.0}
        self.episode_to_features[episode_id] = row
        self.authority[episode_id] = float(authority)
        for feature, weight in row.items():
            self.feature_to_episodes[feature][episode_id] = weight
    def activate(self, state: Mapping[int, float], *, top_m: int = 64) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for feature, value in state.items():
            for episode_id, weight in self.feature_to_episodes.get(int(feature), {}).items():
                scores[episode_id] += float(value) * weight * self.authority[episode_id]
        return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_m])
    def reactivatefeatures(self, episode_scores: Mapping[str, float], *, topk: int = 192) -> dict[int, float]:
        features: dict[int, float] = defaultdict(float)
        for episode_id, score in episode_scores.items():
            for feature, weight in self.episode_to_features.get(episode_id, {}).items():
                features[feature] += float(score) * weight
        return dict(sorted(features.items(), key=lambda item: abs(item[1]), reverse=True)[:topk])
    def depress(self, episode_id: str, amount: float) -> None:
        self.authority[episode_id] = max(0.0, self.authority[episode_id] - float(amount))
    def potentiate(self, episode_id: str, amount: float) -> None:
        self.authority[episode_id] = min(2.0, self.authority[episode_id] + float(amount))
    def stats(self) -> dict[str, int]:
        return {
            "episode_nodes": len(self.episode_to_features),
            "episode_feature_edges": sum(len(x) for x in self.episode_to_features.values()),
        }
    def statedict(self) -> dict[str, Any]:
        return {
            "episode_to_features": {
                eid: {str(i): v for i, v in row.items()} for eid, row in self.episode_to_features.items()
            },
            "authority": dict(self.authority),
            "write_count": self.write_count,
        }
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        self.episode_to_features = {
            str(eid): {int(i): float(v) for i, v in row.items()}
            for eid, row in state.get("episode_to_features", {}).items()
        }
        self.feature_to_episodes = defaultdict(dict)
        for episode_id, row in self.episode_to_features.items():
            for feature, weight in row.items():
                self.feature_to_episodes[feature][episode_id] = weight
        self.authority = defaultdict(lambda: 1.0, {str(k): float(v) for k, v in state.get("authority", {}).items()})
        self.write_count = int(state.get("write_count", 0))
def degree(graph: Mapping[int, Mapping[int, float]], node: int) -> float:
    return sum(abs(v) for v in graph.get(node, {}).values())
def msgs(
    state: Mapping[int, float],
    graph: Mapping[int, Mapping[int, float]],
    *,
    reverse: bool = False,
    normalize: bool = True,
) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    if reverse:
        for source, neighbors in graph.items():
            for target, weight in neighbors.items():
                if target not in state:
                    continue
                denom = 1.0
                if normalize:
                    denom = math.sqrt(max(degree(graph, source), 1e-12) * max(degree(graph, target), 1e-12))
                out[source] += state[target] * weight / denom
        return dict(out)
    for source, source_value in state.items():
        for target, weight in graph.get(source, {}).items():
            denom = 1.0
            if normalize:
                denom = math.sqrt(max(degree(graph, source), 1e-12) * max(degree(graph, target), 1e-12))
            out[target] += source_value * weight / denom
    return dict(out)
def topknormalize(values: Mapping[int, float], *, dimension: int, k: int, threshold: float, signed: bool) -> SparseCode:
    if signed:
        activated = {i: math.tanh(v) for i, v in values.items() if abs(v) >= threshold}
    else:
        activated = {i: max(0.0, v) for i, v in values.items() if v >= threshold}
    items = sorted(activated.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
    if not items:
        return SparseCode.empty(dimension)
    return SparseCode.frommapping(dimension, dict(items)).normalized()
class HippocampalCompletionNetwork:
    def __init__(
        self,
        *,
        episodeindex: EpisodeIndexGraph,
        context_graph: ContextAssociationGraph,
        config: RetrievalConfig | None = None,
        episodefeedbackstrength: float = 0.35,
        contextstrength: float = 0.25,
        topepisodenodes: int = 64,
    ) -> None:
        self.episodeindex = episodeindex
        self.context_graph = context_graph
        self.config = config or RetrievalConfig()
        self.episodefeedbackstrength = float(episodefeedbackstrength)
        self.contextstrength = float(contextstrength)
        self.topepisodenodes = int(topepisodenodes)
    @classmethod
    def fromconfig(
        cls,
        *,
        episodeindex: EpisodeIndexGraph,
        context_graph: ContextAssociationGraph,
        retrieval_config: RetrievalConfig,
        config: Mapping[str, Any] | None = None,
    ) -> HippocampalCompletionNetwork:
        config = dict(config or {})
        return cls(
            episodeindex=episodeindex,
            context_graph=context_graph,
            config=retrieval_config,
            episodefeedbackstrength=float(config.get("episodefeedbackstrength", 0.35)),
            contextstrength=float(config.get("contextstrength", 0.25)),
            topepisodenodes=int(config.get("topepisodenodes", 64)),
        )
    def complete(self, query: SparseCode, context: SparseCode, graph: SparseAssociationGraph) -> CompletionTrace:
        cfg = self.config
        anchor = query.asdict()
        state = dict(anchor)
        steps: list[dict[int, float]] = [dict(state)]
        temporal_total: dict[int, float] = defaultdict(float)
        all_episode_scores: dict[str, float] = {}
        message_edge_visits = 0
        assoc_graph = graph.conditionedgraph(graph.association, graph_name="association")
        temporal_graph = graph.conditionedgraph(graph.temporal, graph_name="temporal")
        context_messages = self.context_graph.messages(context, normalize=cfg.normalizemessages)
        for _ in range(max(0, cfg.recurrencesteps)):
            message_edge_visits += sum(len(assoc_graph.get(i, {})) for i in state)
            message_edge_visits += sum(len(temporal_graph.get(i, {})) for i in state)
            assoc = msgs(state, assoc_graph, normalize=cfg.normalizemessages)
            temporal_forward = msgs(state, temporal_graph, normalize=cfg.normalizemessages)
            temporal_backward = msgs(state, temporal_graph, reverse=True, normalize=cfg.normalizemessages)
            episode_scores = self.episodeindex.activate(state, top_m=self.topepisodenodes)
            for episode_id, score in episode_scores.items():
                all_episode_scores[episode_id] = max(all_episode_scores.get(episode_id, 0.0), score)
            episode_feedback = self.episodeindex.reactivatefeatures(
                episode_scores, topk=cfg.featuretopk
            )
            combined: dict[int, float] = defaultdict(float)
            for i, value in anchor.items():
                combined[i] += cfg.queryanchor * value
            for i, value in assoc.items():
                penalty = (1.0 + degree(assoc_graph, i)) ** graph.config.hubpenalty
                combined[i] += cfg.associationstrength * value / penalty
            for messages in (temporal_forward, temporal_backward):
                for i, value in messages.items():
                    temporal_total[i] += value
                    combined[i] += cfg.temporalstrength * value
            for i, value in episode_feedback.items():
                combined[i] += self.episodefeedbackstrength * value
            for i, value in context_messages.items():
                combined[i] += self.contextstrength * value
            code = topknormalize(
                combined,
                dimension=query.dimension,
                k=cfg.featuretopk,
                threshold=cfg.threshold,
                signed=cfg.use_signed_messages,
            )
            state = code.asdict()
            steps.append(dict(state))
            if not state:
                break
        final = SparseCode.frommapping(query.dimension, state).normalized()
        temporal = SparseCode.frommapping(query.dimension, temporal_total).topk(cfg.featuretopk).normalized()
        return CompletionTrace(
            final=final,
            temporal=temporal,
            episode_scores=dict(sorted(all_episode_scores.items(), key=lambda item: (-item[1], item[0]))),
            steps=tuple(steps),
            diagnostics={
                "association_edges": sum(len(x) for x in graph.association.values()),
                "temporal_edges": sum(len(x) for x in graph.temporal.values()),
                "episode_nodes": len(self.episodeindex.episode_to_features),
                "context_edges": sum(len(x) for x in self.context_graph.links.values()),
                "message_edge_visits": message_edge_visits,
                "queryanchor": cfg.queryanchor,
                "text_used_for_scoring": False,
            },
        )
