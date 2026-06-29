from __future__ import annotations
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from aamemory.config import MemoryConfig
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.encoding.binding import BoundAddress, composeboundaddress
from aamemory.encoding.context import ContextEncoder
from aamemory.encoding.factory import buildencoder
from aamemory.memory.accounting import estimatememoryfootprint
from aamemory.memory.allocation import EligibilityAllocator
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.memory.capacity import CapacityManager
from aamemory.memory.completion import CompletionTrace, EpisodeIndexGraph, HippocampalCompletionNetwork
from aamemory.memory.contextgraph import ContextAssociationGraph
from aamemory.memory.dopamine import DopamineGate, DopamineMetrics
from aamemory.memory.neurogenesis import NeurogenesisController
from aamemory.memory.prospection import FuturePromptSimulator
from aamemory.memory.reconsolidation import ReconsolidationLedger
from aamemory.memory.replayv2 import SourceVerifiedReplayEngine
from aamemory.memory.sqlitestore import SQLiteEpisodeStore
from aamemory.memory.store import EpisodeStore, InMemoryEpisodeStore
from aamemory.schema import Episode, MemoryEvent, QueryResult, SourceRef, SparseCode, utcnowiso
class HippocampalActivationMemory:
    primary_memory_substrate = "activation_engram"
    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        encoder: FeatureEncoder | None = None,
        store: EpisodeStore | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.hidden_encoder = encoder or buildencoder(self.config.encoder)
        context_cfg = dict(self.config.context or {})
        self.context_encoder = ContextEncoder(**context_cfg.get("encoder", context_cfg))
        binding_cfg = dict(self.config.context.get("binding", {})) if self.config.context else {}
        self.bindingdimension = int(binding_cfg.get("bindingdimension", binding_cfg.get("dimension", 8192)))
        self.neuro_reserved = int((self.config.neurogenesis or {}).get("reservedfeatures", 0))
        self.address_top_k = int(binding_cfg.get("topk", self.config.retrieval.featuretopk))
        dummy = composeboundaddress(
            SparseCode.empty(self.hidden_encoder.dimension),
            SparseCode.empty(self.context_encoder.dimension),
            bindingdimension=self.bindingdimension,
            neurogenesis_reserved=self.neuro_reserved,
            topk=self.address_top_k,
        )
        self.dimension = dummy.code.dimension
        self.encoder = self
        self.store = store or self.buildstore(self.config.store)
        self.graph = SparseAssociationGraph(self.dimension, self.config.graph)
        self.episodeindex = EpisodeIndexGraph.fromconfig(self.config.episodeindex)
        self.context_graph = ContextAssociationGraph.fromconfig(self.config.context.get("graph", {}) if self.config.context else {})
        self.completion = HippocampalCompletionNetwork.fromconfig(
            episodeindex=self.episodeindex,
            context_graph=self.context_graph,
            retrieval_config=self.config.retrieval,
            config=self.config.completion,
        )
        self.dopamine_gate = DopamineGate.fromconfig(self.config.dopamine)
        self.allocator = EligibilityAllocator.fromconfig(self.config.allocation)
        self.neurogenesis = NeurogenesisController.fromconfig(
            reserved_start=dummy.neurogenesis_offset,
            config=self.config.neurogenesis,
        )
        self.reconsolidation = ReconsolidationLedger()
        self.prospection = FuturePromptSimulator.fromconfig(self.config.prospection)
        self.replay_engine = SourceVerifiedReplayEngine(self.store, self.graph, self.episodeindex)
        self.capacity_manager = CapacityManager(self.store)
        self._previous_by_stream: dict[str, Episode] = {}
        self._maintenance_log: list[dict[str, Any]] = []
        self._event_write_count = 0
        self._last_binding: BoundAddress | None = None
    @property
    def featuredimension(self) -> int:
        return self.dimension
    @staticmethod
    def buildstore(config: Mapping[str, Any]) -> EpisodeStore:
        kind = str(config.get("type", "memory")).lower()
        if kind in {"memory", "in_memory", "ram"}:
            return InMemoryEpisodeStore()
        if kind == "sqlite":
            return SQLiteEpisodeStore(config.get("path", "runs/aamemory-v2.sqlite3"))
        raise ValueError(f"unknown episode store: {kind}")
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        hidden = self.hidden_encoder.encode(text, metadata=metadata)
        context = self.context_encoder.encode(metadata)
        bound = self.compose(hidden.code, context.code)
        return EncodingResult(
            code=bound.code,
            payload={"hidden_payload": dict(hidden.payload), "context_features": list(context.features)},
            diagnostics={**dict(hidden.diagnostics), **dict(context.diagnostics), **dict(bound.diagnostics)},
        )
    def compose(self, hidden: SparseCode, context: SparseCode) -> BoundAddress:
        binding_cfg = dict(self.config.context.get("binding", {})) if self.config.context else {}
        bound = composeboundaddress(
            hidden,
            context,
            bindingdimension=self.bindingdimension,
            neurogenesis_reserved=self.neuro_reserved,
            topk=self.address_top_k,
            hidden_weight=float(binding_cfg.get("hidden_weight", 1.0)),
            contextweight=float(binding_cfg.get("contextweight", 0.35)),
            bindingweight=float(binding_cfg.get("bindingweight", 0.45)),
            seed=int(binding_cfg.get("seed", 17)),
        )
        self._last_binding = bound
        return bound
    def novelty(self, code: SparseCode) -> tuple[float, float]:
        candidates = self.store.candidates(code.indices, limit=30)
        if not candidates:
            return 1.0, 0.0
        sims = [candidate.code.dot(code) for candidate in candidates]
        maximum = max(sims, default=0.0)
        close = sum(sim > 0.45 for sim in sims)
        return max(0.0, min(1.0, 1.0 - maximum)), min(1.0, close / 5.0)
    def sourceforevent(self, event: MemoryEvent, metadata: Mapping[str, Any]) -> SourceRef:
        source = event.source
        generated = bool(metadata.get("generated") or metadata.get("hypothetical"))
        if source.checksum is None and not generated:
            return SourceRef.fortext(
                event.text,
                uri=source.uri,
                document_id=source.document_id,
                start=source.start,
                end=source.end,
                metadata=source.metadata,
            )
        return source
    def write(
        self,
        event: MemoryEvent,
        *,
        salience: float | None = None,
        confidence: float = 1.0,
        stream_id: str | None = None,
    ) -> Episode:
        metadata = dict(event.metadata)
        metadata.setdefault("text_is_provenance_only", True)
        metadata.setdefault("primary_memory_substrate", self.primary_memory_substrate)
        hidden = self.hidden_encoder.encode(event.text, metadata=metadata)
        context = self.context_encoder.encode({**metadata, "timestamp": event.timestamp})
        bound = self.compose(hidden.code, context.code)
        novelty, interference = self.novelty(bound.code)
        dopamine = DopamineMetrics.frommetadata(
            {**metadata, "text_length": len(event.text)}, novelty=novelty, redundancy=1.0 - novelty
        )
        delta_pos, delta_neg = self.dopamine_gate.score(dopamine)
        if salience is not None:
            delta_pos = float(salience)
        allocation = self.allocator.allocate(
            bound.code,
            graph=self.graph,
            novelty=dopamine.novelty,
            utility=dopamine.downstreamutility,
        )
        address = self.allocator.apply(bound.code, allocation)
        address, birth_record, birthscore = self.neurogenesis.maybeaugment(
            address,
            novelty=novelty,
            interference=interference,
            retrieval_error=float(metadata.get("retrieval_error", 0.0)),
            budget_pressure=float(metadata.get("budget_pressure", 0.0)),
        )
        source = self.sourceforevent(event, metadata)
        status = "hypothetical" if bool(metadata.get("hypothetical") or metadata.get("generated")) else "observed"
        if metadata.get("rejected"):
            status = "rejected"
        payload: dict[str, Any] = {
            "payload_kind": "activation_engram_v2",
            "hidden_code": hidden.code.tojsonable(),
            "context_code": context.code.tojsonable(),
            "hidden_payload": dict(hidden.payload),
            "context_features": list(context.features),
            "dopamine_metrics": dopamine.todict(),
            "delta_ltp": delta_pos,
            "delta_ltd": delta_neg,
            "allocation": dict(allocation.diagnostics),
            "binding": dict(bound.diagnostics),
            "neurogenesis_birth_score": birthscore,
            "latent_authority": 0.0 if status == "hypothetical" else max(0.0, 1.0 - dopamine.poisonrisk),
        }
        if birth_record is not None:
            payload["new_feature"] = birth_record.todict()
        episode = Episode(
            episode_id=event.event_id,
            text=event.text,
            code=address,
            timestamp=str(event.timestamp or metadata.get("timestamp") or utcnowiso()),
            salience=float(delta_pos),
            confidence=float(confidence),
            source=source,
            payload=payload,
            metadata={
                **metadata,
                "engram_status": status,
                "dopamine": dopamine.todict(),
                "delta_ltd": delta_neg,
                "novelty": novelty,
                "interference": interference,
                "context_diagnostics": dict(context.diagnostics),
                "encoding_diagnostics": dict(hidden.diagnostics),
            },
        )
        stream = stream_id or str(metadata.get("stream_id", "default"))
        previous = self._previous_by_stream.get(stream)
        if delta_neg > 0.5:
            self.graph.decay(min(0.05, 0.01 * delta_neg))
            self.context_graph.decay(min(0.05, 0.01 * delta_neg))
        if status != "rejected":
            self.graph.write(address, salience=episode.salience, previous=previous.code if previous else None)
            authority = float(payload["latent_authority"])
            self.episodeindex.write(episode.episode_id, address, salience=episode.salience, authority=authority)
            self.context_graph.write(context.code, address, salience=episode.salience)
        self.store.add(episode)
        self.reconsolidation.register(
            episode.episode_id,
            status=status,
            confidence=episode.confidence,
            authority=float(payload["latent_authority"]),
            source_checksum=episode.source.checksum,
            metadata={"text_is_provenance_only": True},
        )
        corrects = metadata.get("corrects") or metadata.get("supersedes")
        if corrects:
            if isinstance(corrects, str):
                correct_ids = [corrects]
            else:
                correct_ids = [str(x) for x in corrects]
            for old_id in correct_ids:
                report = self.reconsolidation.supersede(old_id, episode.episode_id)
                old = self.store.get(old_id)
                if old is not None:
                    old.metadata["engram_status"] = "superseded"
                    old.metadata["superseded_by"] = episode.episode_id
                    old.confidence = min(old.confidence, 0.25)
                    self.store.add(old)
                    self.episodeindex.depress(old_id, 0.35)
                self._maintenance_log.append({"kind": "reconsolidation", **asdict(report)})
        self._previous_by_stream[stream] = episode
        self._event_write_count += 1
        self.runmaintenance()
        return episode
    def completequery(self, text: str, metadata: Mapping[str, Any] | None = None) -> tuple[SparseCode, SparseCode, CompletionTrace]:
        hidden = self.hidden_encoder.encode(text, metadata=metadata)
        context = self.context_encoder.encode(metadata)
        query = self.compose(hidden.code, context.code).code
        trace = self.completion.complete(query, context.code, self.graph)
        return query, context.code, trace
    def contextsimilarity(self, episode: Episode, query_context: SparseCode) -> float:
        try:
            context_payload = episode.payload.get("context_code")
            if context_payload:
                return SparseCode.fromjsonable(context_payload).dot(query_context)
        except (AttributeError, KeyError, TypeError, ValueError):
            return 0.0
        return 0.0
    def candidateepisodes(self, query: SparseCode, trace: CompletionTrace) -> list[Episode]:
        features = set(query.indices) | set(trace.final.indices) | set(trace.temporal.indices)
        candidates = self.store.candidates(features, limit=self.config.retrieval.candidatelimit)
        for episode_id in list(trace.episode_scores)[: self.config.retrieval.candidatelimit]:
            episode = self.store.get(episode_id)
            if episode is not None and episode not in candidates:
                candidates.append(episode)
        if not candidates and len(self.store) <= self.config.retrieval.candidatelimit:
            candidates = list(self.store.all())
        return candidates
    def recencyscore(self, episode: Episode) -> float:
        return float(episode.metadata.get("recency", 0.0))
    def retrievefromtrace(
        self,
        query: SparseCode,
        query_context: SparseCode,
        trace: CompletionTrace,
    ) -> list[QueryResult]:
        cfg = self.config.retrieval
        completion_cfg = dict(self.config.completion or {})
        episode_weight = float(completion_cfg.get("episodenodeweight", 0.20))
        contextweight = float(completion_cfg.get("contextscoreweight", 0.10))
        risk_weight = float(completion_cfg.get("risk_weight", 0.20))
        results: list[QueryResult] = []
        for episode in self.candidateepisodes(query, trace):
            status = str(episode.metadata.get("engram_status", "observed"))
            if status in {"rejected"}:
                continue
            exact = episode.code.dot(query)
            associative = episode.code.dot(trace.final)
            temporal = episode.code.dot(trace.temporal)
            episode_node = float(trace.episode_scores.get(episode.episode_id, 0.0))
            context_score = self.contextsimilarity(episode, query_context)
            risk = float(episode.metadata.get("dopamine", {}).get("poisonrisk", episode.metadata.get("poisonrisk", 0.0)))
            authority = float(episode.payload.get("latent_authority", 1.0))
            confidence = episode.confidence
            if status == "hypothetical":
                authority *= 0.1
            if status == "superseded":
                authority *= 0.2
                confidence *= 0.2
            score = (
                cfg.exactweight * exact
                + cfg.associativeweight * associative
                + cfg.temporalweight * temporal
                + cfg.recencyweight * self.recencyscore(episode)
                + cfg.confidenceweight * confidence
                + episode_weight * episode_node
                + contextweight * context_score
                - risk_weight * risk
            ) * max(0.0, authority)
            if not math.isfinite(score):
                continue
            results.append(
                QueryResult(
                    episode_id=episode.episode_id,
                    score=float(score),
                    exact_score=float(exact),
                    associative_score=float(associative + episode_weight * episode_node + contextweight * context_score),
                    temporal_score=float(temporal),
                    recency_score=float(self.recencyscore(episode)),
                    episode=episode,
                    trace={
                        "episode_node_score": episode_node,
                        "context_score": context_score,
                        "risk": risk,
                        "authority": authority,
                        "engram_status": status,
                        "text_used_for_scoring": False,
                        "score_components": {
                            "direct": exact,
                            "completed": associative,
                            "temporal": temporal,
                            "episode_node": episode_node,
                            "context": context_score,
                            "risk": risk,
                            "authority": authority,
                        },
                    },
                )
            )
        results.sort(key=lambda result: (-result.score, result.episode_id))
        return results[: cfg.topk]
    def query(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> list[QueryResult]:
        results, _ = self.querywithtrace(text, metadata=metadata)
        return results
    def querywithtrace(
        self,
        text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[list[QueryResult], CompletionTrace]:
        query, query_context, trace = self.completequery(text, metadata)
        results = self.retrievefromtrace(query, query_context, trace)
        verified = None if results else False
        report = self.reconsolidation.afteruse(
            results,
            store=self.store,
            episodeindex=self.episodeindex,
            verified=verified,
        )
        if report.updated or report.depressed:
            self._maintenance_log.append({"kind": "reconsolidation_after_use", **asdict(report)})
        return results, trace
    def simulatefutureprompts(
        self,
        *,
        recent_query: str | None = None,
        task_state: Mapping[str, Any] | None = None,
    ) -> list[Any]:
        prompts = self.prospection.proposeprompts(recent_query=recent_query, task_state=task_state)
        def encodequery(prompt: str, meta: Mapping[str, Any] | None) -> SparseCode:
            return self.encode(prompt, metadata=meta).code
        def complete(q: SparseCode, meta: Mapping[str, Any] | None) -> SparseCode:
            context = self.context_encoder.encode(meta).code
            return self.completion.complete(q, context, self.graph).final
        return self.prospection.simulate(
            prompts,
            encodequery=encodequery,
            complete=complete,
            metadata={**dict(task_state or {}), "hypothetical": True},
        )
    def replaycycle(self, *, budget: int | None = None) -> dict[str, Any]:
        cfg = dict(self.config.replay or {})
        report = self.replay_engine.run(
            budget=int(budget if budget is not None else cfg.get("budgetpercycle", 32)),
            learningscale=float(cfg.get("learningscale", 0.15)),
            strategy=str(cfg.get("strategy", "salience")),
        )
        data = {"kind": "verified_replay", **asdict(report)}
        self._maintenance_log.append(data)
        return data
    def runmaintenance(self) -> None:
        replay_cfg = dict(self.config.replay or {})
        if bool(replay_cfg.get("enabled", False)):
            interval = max(1, int(replay_cfg.get("intervalwrites", 500)))
            if self._event_write_count % interval == 0:
                self.replaycycle()
        consolidation_cfg = dict(self.config.consolidation or {})
        maximum = int(consolidation_cfg.get("fixedcapacityepisodes", consolidation_cfg.get("max_episodes", 0)) or 0)
        if maximum > 0 and len(self.store) > maximum:
            report = self.capacity_manager.enforce(maximum, policy=str(consolidation_cfg.get("evictionpolicy", "salience")))
            self._maintenance_log.append({"kind": "eviction", **asdict(report)})
            if report.evicted_ids:
                self.rebuildgraphfromstore()
    def clear(self) -> None:
        self.store.clear()
        self.graph.clear()
        self.episodeindex.clear()
        self.context_graph.links.clear()
        self.allocator.eligibility.clear()
        self.reconsolidation.versions.clear()
        self.reconsolidation.children.clear()
        self.prospection.traces.clear()
        self._previous_by_stream.clear()
        self._maintenance_log.clear()
        self._event_write_count = 0
    def rebuildgraphfromstore(self) -> dict[str, int]:
        episodes = sorted(self.store.all(), key=lambda e: (e.timestamp, e.episode_id))
        self.graph.clear()
        self.episodeindex.clear()
        self.context_graph.links.clear()
        previous_by_stream: dict[str, Episode] = {}
        for episode in episodes:
            status = str(episode.metadata.get("engram_status", "observed"))
            if status == "rejected":
                continue
            stream = str(episode.metadata.get("stream_id", "default"))
            previous = previous_by_stream.get(stream)
            self.graph.write(episode.code, salience=episode.salience, previous=previous.code if previous else None)
            self.episodeindex.write(
                episode.episode_id,
                episode.code,
                salience=episode.salience,
                authority=float(episode.payload.get("latent_authority", 1.0)),
            )
            try:
                context = SparseCode.fromjsonable(episode.payload["context_code"])
                self.context_graph.write(context, episode.code, salience=episode.salience)
            except (KeyError, TypeError, ValueError):
                pass
            previous_by_stream[stream] = episode
        self._previous_by_stream = previous_by_stream
        return {
            "episodes_replayed": len(episodes),
            "association_edges": sum(len(x) for x in self.graph.association.values()),
            "temporal_edges": sum(len(x) for x in self.graph.temporal.values()),
            "episode_index_edges": sum(len(x) for x in self.episodeindex.episode_to_features.values()),
            "context_edges": sum(len(x) for x in self.context_graph.links.values()),
        }
    def stats(self) -> dict[str, Any]:
        store_stats = self.store.stats() if hasattr(self.store, "stats") else {"episodes": len(self.store)}
        footprint = estimatememoryfootprint(self.store.all(), self.graph).todict()
        context_stats = self.context_graph.stats()
        return {
            "variant": "aam_v2_hippocampal_activation_memory",
            "primary_memory_substrate": self.primary_memory_substrate,
            "raw_text_role": "audit_provenance_exact_fallback_only",
            "episodes": len(self.store),
            "dimension": self.dimension,
            "hidden_dimension": self.hidden_encoder.dimension,
            "context_dimension": self.context_encoder.dimension,
            "association_edges": sum(len(x) for x in self.graph.association.values()),
            "temporal_edges": sum(len(x) for x in self.graph.temporal.values()),
            "episodeindex": self.episodeindex.stats(),
            "context_graph": asdict(context_stats),
            "neurogenesis": self.neurogenesis.stats(),
            "prospection": self.prospection.stats(),
            "replay_quarantined": len(self.replay_engine.quarantined_ids),
            "event_writes": self._event_write_count,
            "maintenance_events": len(self._maintenance_log),
            "last_maintenance": self._maintenance_log[-1] if self._maintenance_log else None,
            "logical_footprint": footprint,
            "store": store_stats,
        }
    def statedict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.statedict(),
            "episodeindex": self.episodeindex.statedict(),
            "context_graph": self.context_graph.statedict(),
            "allocator": self.allocator.statedict(),
            "neurogenesis": self.neurogenesis.statedict(),
            "reconsolidation": self.reconsolidation.statedict(),
            "prospection": self.prospection.statedict(),
            "previous_by_stream": {k: v.episode_id for k, v in self._previous_by_stream.items()},
        }
    def savegraph(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.statedict(), separators=(",", ":")), encoding="utf-8")
    def loadgraph(self, path: str | Path) -> None:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        self.graph.loadstatedict(state["graph"])
        self.episodeindex.loadstatedict(state.get("episodeindex", {}))
        self.context_graph.loadstatedict(state.get("context_graph", {}))
        self.allocator.loadstatedict(state.get("allocator", {}))
        self.neurogenesis.loadstatedict(state.get("neurogenesis", {}))
        self.reconsolidation.loadstatedict(state.get("reconsolidation", {}))
    def close(self) -> None:
        self.store.close()
