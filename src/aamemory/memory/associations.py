from __future__ import annotations
import math
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
from aamemory.config import GraphConfig, RetrievalConfig
from aamemory.schema import SparseCode
@dataclass(frozen=True)
class RecallTrace:
    final: SparseCode
    temporal: SparseCode
    steps: tuple[dict[int, float], ...]
    diagnostics: Mapping[str, Any]
class SparseAssociationGraph:
    def __init__(self, dimension: int, config: GraphConfig | None = None) -> None:
        self.dimension = int(dimension)
        self.config = config or GraphConfig()
        self.association: dict[int, dict[int, float]] = defaultdict(dict)
        self.temporal: dict[int, dict[int, float]] = defaultdict(dict)
        self.activity_mean: dict[int, float] = defaultdict(float)
        self.activity_second_moment: dict[int, float] = defaultdict(float)
        self.write_count = 0
        self.association_update_count = 0
        self.temporal_update_count = 0
        self._condition_cache: dict[
            tuple[str, int, int, str], dict[int, dict[int, float]]
        ] = {}
    def checked(self, code: SparseCode) -> None:
        if code.dimension != self.dimension:
            raise ValueError(
                f"code dimension {code.dimension} does not match graph {self.dimension}"
            )
    def clear(self) -> None:
        self.association.clear()
        self.temporal.clear()
        self.activity_mean.clear()
        self.activity_second_moment.clear()
        self.write_count = 0
        self.association_update_count = 0
        self.temporal_update_count = 0
        self._condition_cache.clear()
    def updateedge(
        self,
        graph: dict[int, dict[int, float]],
        source: int,
        target: int,
        delta: float,
    ) -> None:
        if source == target or delta == 0.0 or not math.isfinite(delta):
            return
        value = graph[source].get(target, 0.0) + float(delta)
        if not self.config.allownegativeedges:
            value = max(0.0, value)
        if abs(value) < 1e-12:
            graph[source].pop(target, None)
        else:
            graph[source][target] = value
    def pairdelta(self, i: int, vi: float, j: int, vj: float, salience: float) -> float:
        eta = self.config.learningrate * salience
        rule = self.config.rule.lower()
        old = self.association.get(i, {}).get(j, 0.0)
        if rule in {"none", "off", "zero"}:
            return 0.0
        if rule == "hebb":
            return eta * vi * vj
        if rule == "covariance":
            return eta * (vi - self.activity_mean[i]) * (vj - self.activity_mean[j])
        if rule == "oja":
            norm_term = 0.5 * (vi * vi + vj * vj) * old
            return eta * (vi * vj - norm_term)
        if rule == "bcm":
            theta_i = self.activity_second_moment[i]
            theta_j = self.activity_second_moment[j]
            return 0.5 * eta * (vi * vj * (vi - theta_i) + vj * vi * (vj - theta_j))
        raise ValueError(f"unknown plasticity rule: {self.config.rule}")
    def write(
        self,
        code: SparseCode,
        *,
        salience: float = 1.0,
        previous: SparseCode | None = None,
    ) -> None:
        self.checked(code)
        if previous is not None:
            self.checked(previous)
        self.write_count += 1
        if self.config.decayinterval > 0 and self.write_count % self.config.decayinterval == 0:
            self.decay(self.config.decay)
        pairs = sorted(
            zip(code.indices, code.values, strict=True), key=lambda item: abs(item[1]), reverse=True
        )[: self.config.maxpairfeatures]
        for offset, (i, vi) in enumerate(pairs):
            for j, vj in pairs[offset + 1 :]:
                delta = self.pairdelta(i, vi, j, vj, salience)
                self.updateedge(self.association, i, j, delta)
                self.updateedge(self.association, j, i, delta)
                if delta != 0.0:
                    self.association_update_count += 2
        if previous is not None and self.config.rule.lower() not in {"none", "off", "zero"}:
            prev_pairs = sorted(
                zip(previous.indices, previous.values, strict=True),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[: self.config.maxpairfeatures]
            for i, vi in prev_pairs:
                for j, vj in pairs:
                    delta = self.config.temporallearningrate * salience * vi * vj
                    self.updateedge(self.temporal, i, j, delta)
                    if delta != 0.0:
                        self.temporal_update_count += 1
        mean_rate = min(0.1, max(1e-4, self.config.learningrate * 0.1))
        for i, value in pairs:
            magnitude = abs(value)
            self.activity_mean[i] = (1 - mean_rate) * self.activity_mean[i] + mean_rate * magnitude
            self.activity_second_moment[i] = (
                (1 - mean_rate) * self.activity_second_moment[i]
                + mean_rate * magnitude * magnitude
            )
        self.prune()
        if self.config.normalizeafterwrite:
            self.capnodenorms()
        self._condition_cache.clear()
    def decay(self, rate: float | None = None) -> None:
        rate = self.config.decay if rate is None else float(rate)
        multiplier = max(0.0, 1.0 - rate)
        for graph in (self.association, self.temporal):
            for source in list(graph):
                for target in list(graph[source]):
                    graph[source][target] *= multiplier
                    if abs(graph[source][target]) < 1e-9:
                        del graph[source][target]
                if not graph[source]:
                    del graph[source]
        self._condition_cache.clear()
    def prune(self) -> None:
        maxdegree = max(1, self.config.maxdegree)
        for graph in (self.association, self.temporal):
            for source in list(graph):
                neighbors = graph[source]
                if len(neighbors) > maxdegree:
                    keep = sorted(neighbors, key=lambda j: abs(neighbors[j]), reverse=True)[:maxdegree]
                    graph[source] = {j: neighbors[j] for j in keep}
                if not graph[source]:
                    del graph[source]
        self._condition_cache.clear()
    def conditionedgraph(
        self,
        graph: Mapping[int, Mapping[int, float]],
        *,
        graph_name: str,
    ) -> Mapping[int, Mapping[int, float]]:
        condition = self.config.edgecondition.lower().replace("-", "_")
        if condition in {"learned", "identity", "original"}:
            return graph
        if condition in {"zero", "none", "off"}:
            return {}
        cache_key = (condition, int(self.config.edgeseed), self.write_count, graph_name)
        if cache_key in self._condition_cache:
            return self._condition_cache[cache_key]
        nodes = sorted(
            set(graph)
            | {target for neighbors in graph.values() for target in neighbors}
        )
        rng = random.Random(
            int(self.config.edgeseed)
            + (0 if graph_name == "association" else 1_000_003)
            + self.write_count * 97
        )
        conditioned: dict[int, dict[int, float]] = defaultdict(dict)
        if condition in {"shuffled", "shuffled_degree_preserving", "permuted"}:
            permuted = list(nodes)
            rng.shuffle(permuted)
            mapping = dict(zip(nodes, permuted, strict=True))
            for source, neighbors in graph.items():
                new_source = mapping[source]
                for target, weight in neighbors.items():
                    new_target = mapping[target]
                    if new_source != new_target:
                        conditioned[new_source][new_target] = float(weight)
        elif condition in {"random", "random_degree_matched"}:
            for source, neighbors in graph.items():
                targets = [node for node in nodes if node != source]
                if not targets:
                    continue
                weights = list(neighbors.values())
                if len(targets) >= len(weights):
                    sampled = rng.sample(targets, len(weights))
                else:
                    sampled = [rng.choice(targets) for _ in weights]
                for target, weight in zip(sampled, weights, strict=True):
                    conditioned[source][target] = float(weight)
        else:
            raise ValueError(f"unknown edge condition: {self.config.edgecondition}")
        out = {source: dict(neighbors) for source, neighbors in conditioned.items()}
        self._condition_cache[cache_key] = out
        return out
    def capnodenorms(self, maximum: float = 1.0) -> None:
        for graph in (self.association, self.temporal):
            for source, neighbors in graph.items():
                norm = math.sqrt(sum(value * value for value in neighbors.values()))
                if norm > maximum:
                    scale = maximum / norm
                    graph[source] = {target: value * scale for target, value in neighbors.items()}
    @staticmethod
    def degree(graph: Mapping[int, Mapping[int, float]], node: int) -> float:
        return sum(abs(value) for value in graph.get(node, {}).values())
    def messages(
        self,
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
                        denom = math.sqrt(
                            max(self.degree(graph, source), 1e-12)
                            * max(self.degree(graph, target), 1e-12)
                        )
                    out[source] += state[target] * weight / denom
            return dict(out)
        for source, source_value in state.items():
            for target, weight in graph.get(source, {}).items():
                denom = 1.0
                if normalize:
                    denom = math.sqrt(
                        max(self.degree(graph, source), 1e-12)
                        * max(self.degree(graph, target), 1e-12)
                    )
                out[target] += source_value * weight / denom
        return dict(out)
    @staticmethod
    def topknormalize(
        values: Mapping[int, float], *, k: int, threshold: float, signed: bool
    ) -> dict[int, float]:
        if signed:
            activated = {i: math.tanh(v) for i, v in values.items() if abs(v) >= threshold}
        else:
            activated = {i: max(0.0, v) for i, v in values.items() if v >= threshold}
        items = sorted(activated.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
        norm = math.sqrt(sum(v * v for _, v in items))
        if norm <= 1e-12:
            return {}
        return {i: v / norm for i, v in items}
    def recall(self, query: SparseCode, config: RetrievalConfig | None = None) -> RecallTrace:
        self.checked(query)
        cfg = config or RetrievalConfig()
        anchor = query.asdict()
        state = dict(anchor)
        trace: list[dict[int, float]] = [dict(state)]
        temporal_total: dict[int, float] = defaultdict(float)
        association_graph = self.conditionedgraph(
            self.association, graph_name="association"
        )
        temporal_graph = self.conditionedgraph(self.temporal, graph_name="temporal")
        message_edge_visits = 0
        for _ in range(max(0, cfg.recurrencesteps)):
            message_edge_visits += sum(len(association_graph.get(i, {})) for i in state)
            message_edge_visits += sum(len(temporal_graph.get(i, {})) for i in state)
            message_edge_visits += sum(
                1
                for neighbors in temporal_graph.values()
                for target in neighbors
                if target in state
            )
            assoc = self.messages(
                state, association_graph, normalize=cfg.normalizemessages
            )
            temporal_forward = self.messages(
                state, temporal_graph, normalize=cfg.normalizemessages
            )
            temporal_backward = self.messages(
                state, temporal_graph, reverse=True, normalize=cfg.normalizemessages
            )
            combined: dict[int, float] = defaultdict(float)
            for i, value in anchor.items():
                combined[i] += cfg.queryanchor * value
            for i, value in assoc.items():
                penalty = (1.0 + self.degree(association_graph, i)) ** self.config.hubpenalty
                combined[i] += cfg.associationstrength * value / penalty
            for messages in (temporal_forward, temporal_backward):
                for i, value in messages.items():
                    temporal_total[i] += value
                    combined[i] += cfg.temporalstrength * value
            state = self.topknormalize(
                combined,
                k=cfg.featuretopk,
                threshold=cfg.threshold,
                signed=cfg.use_signed_messages,
            )
            trace.append(dict(state))
            if not state:
                break
        final = SparseCode.frommapping(self.dimension, state).normalized()
        temporal = SparseCode.frommapping(self.dimension, temporal_total).topk(
            cfg.featuretopk
        ).normalized()
        return RecallTrace(
            final=final,
            temporal=temporal,
            steps=tuple(trace),
            diagnostics={
                "association_nodes": len(self.association),
                "association_edges": sum(len(x) for x in self.association.values()),
                "temporal_nodes": len(self.temporal),
                "temporal_edges": sum(len(x) for x in self.temporal.values()),
                "config": asdict(cfg),
                "edgecondition": self.config.edgecondition,
                "edgeseed": self.config.edgeseed,
                "message_edge_visits": message_edge_visits,
            },
        )
    def statedict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "association": {str(i): {str(j): w for j, w in n.items()} for i, n in self.association.items()},
            "temporal": {str(i): {str(j): w for j, w in n.items()} for i, n in self.temporal.items()},
            "activity_mean": {str(i): v for i, v in self.activity_mean.items()},
            "activity_second_moment": {
                str(i): v for i, v in self.activity_second_moment.items()
            },
            "write_count": self.write_count,
            "association_update_count": self.association_update_count,
            "temporal_update_count": self.temporal_update_count,
        }
    def loadstatedict(self, state: Mapping[str, Any]) -> None:
        if int(state["dimension"]) != self.dimension:
            raise ValueError("state graph dimension mismatch")
        self.association = defaultdict(
            dict,
            {
                int(i): {int(j): float(w) for j, w in neighbors.items()}
                for i, neighbors in state.get("association", {}).items()
            },
        )
        self.temporal = defaultdict(
            dict,
            {
                int(i): {int(j): float(w) for j, w in neighbors.items()}
                for i, neighbors in state.get("temporal", {}).items()
            },
        )
        self.activity_mean = defaultdict(
            float, {int(i): float(v) for i, v in state.get("activity_mean", {}).items()}
        )
        self.activity_second_moment = defaultdict(
            float,
            {
                int(i): float(v)
                for i, v in state.get("activity_second_moment", {}).items()
            },
        )
        self.write_count = int(state.get("write_count", 0))
        self.association_update_count = int(state.get("association_update_count", 0))
        self.temporal_update_count = int(state.get("temporal_update_count", 0))
        self._condition_cache.clear()
