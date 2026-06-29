from __future__ import annotations
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from aamemory.config import MemoryConfig
from aamemory.encoding.base import FeatureEncoder
from aamemory.encoding.factory import buildencoder
from aamemory.memory.accounting import estimateepisodelogicalbytes, estimatememoryfootprint
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.memory.capacity import CapacityManager
from aamemory.memory.consolidation import PrototypeConsolidator
from aamemory.memory.replay import ReplayEngine
from aamemory.memory.retrieval import AssociativeRetriever
from aamemory.memory.salience import SalienceGate, SalienceSignals
from aamemory.memory.sqlitestore import SQLiteEpisodeStore
from aamemory.memory.store import EpisodeStore, InMemoryEpisodeStore
from aamemory.schema import Episode, MemoryEvent, QueryResult, SourceRef, utcnowiso
class ActivationAssociativeMemory:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        *,
        encoder: FeatureEncoder | None = None,
        store: EpisodeStore | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.encoder = encoder or buildencoder(self.config.encoder)
        self.store = store or self.buildstore(self.config.store)
        self.graph = SparseAssociationGraph(self.encoder.dimension, self.config.graph)
        self.retriever = AssociativeRetriever(self.store, self.graph, self.config.retrieval)
        self.salience_gate = SalienceGate(self.config.salience)
        self._previous_by_stream: dict[str, Episode] = {}
        self.replay_engine = ReplayEngine(self.store, self.graph)
        self.consolidator = PrototypeConsolidator(self.store)
        self.capacity_manager = CapacityManager(self.store)
        self._maintenance_log: list[dict[str, Any]] = []
        self._event_write_count = 0
    @staticmethod
    def buildstore(config: Mapping[str, Any]) -> EpisodeStore:
        kind = str(config.get("type", "memory")).lower()
        if kind in {"memory", "in_memory", "ram"}:
            return InMemoryEpisodeStore()
        if kind == "sqlite":
            return SQLiteEpisodeStore(config.get("path", "runs/aamemory.sqlite3"))
        raise ValueError(f"unknown episode store: {kind}")
    def novelty(self, code: Any) -> float:
        candidates = self.store.candidates(code.indices, limit=20)
        if not candidates:
            return 1.0
        maximum = max((candidate.code.dot(code) for candidate in candidates), default=0.0)
        return max(0.0, min(1.0, 1.0 - maximum))
    def write(
        self,
        event: MemoryEvent,
        *,
        salience: float | None = None,
        confidence: float = 1.0,
        stream_id: str | None = None,
    ) -> Episode:
        encoded = self.encoder.encode(event.text, metadata=event.metadata)
        metadata = dict(event.metadata)
        signals = SalienceSignals.frommetadata(metadata)
        signals = SalienceSignals(
            surprise=signals.surprise,
            task_relevance=signals.task_relevance,
            user_importance=signals.user_importance,
            novelty=self.novelty(encoded.code) if "novelty" not in metadata else signals.novelty,
            redundancy=signals.redundancy,
        )
        salience_value = self.salience_gate.score(signals) if salience is None else float(salience)
        source = event.source
        if source.checksum is None and not bool(metadata.get("generated", False)):
            source = SourceRef.fortext(
                event.text,
                uri=source.uri,
                document_id=source.document_id,
                start=source.start,
                end=source.end,
                metadata=source.metadata,
            )
        episode = Episode(
            episode_id=event.event_id,
            text=event.text,
            code=encoded.code,
            timestamp=str(event.timestamp or metadata.get("timestamp") or utcnowiso()),
            salience=salience_value,
            confidence=float(confidence),
            source=source,
            payload=dict(encoded.payload),
            metadata={**metadata, "encoding_diagnostics": dict(encoded.diagnostics)},
        )
        stream = stream_id or str(metadata.get("stream_id", "default"))
        previous = self._previous_by_stream.get(stream)
        self.graph.write(
            episode.code,
            salience=episode.salience,
            previous=previous.code if previous is not None else None,
        )
        self.store.add(episode)
        self._previous_by_stream[stream] = episode
        self._event_write_count += 1
        self.runmaintenance()
        return episode
    def runmaintenance(self) -> None:
        consolidation_cfg = self.config.consolidation
        maximum = int(
            consolidation_cfg.get(
                "fixedcapacityepisodes",
                consolidation_cfg.get("max_episodes", 0),
            )
            or 0
        )
        graph_needs_rebuild = False
        if bool(consolidation_cfg.get("enabled", False)) and maximum > 0 and len(self.store) > maximum:
            report = self.consolidator.run(
                similaritythreshold=float(consolidation_cfg.get("similaritythreshold", 0.92)),
                minimumgroupsize=int(consolidation_cfg.get("minimumgroupsize", 2)),
                topk=int(consolidation_cfg.get("topk", self.config.retrieval.featuretopk)),
                delete_members=True,
            )
            self._maintenance_log.append({"kind": "consolidation", **asdict(report)})
            graph_needs_rebuild = graph_needs_rebuild or bool(
                report.removed_episodes or report.created_prototypes
            )
        if maximum > 0 and len(self.store) > maximum:
            report = self.capacity_manager.enforce(
                maximum,
                policy=str(consolidation_cfg.get("evictionpolicy", "salience")),
            )
            self._maintenance_log.append({"kind": "eviction", **asdict(report)})
            graph_needs_rebuild = graph_needs_rebuild or bool(report.evicted_ids)
        if graph_needs_rebuild and bool(
            consolidation_cfg.get("rebuildgraphaftermaintenance", True)
        ):
            rebuild = self.rebuildgraphfromstore()
            self._maintenance_log.append({"kind": "graph_rebuild", **rebuild})
        maximum_bytes = int(consolidation_cfg.get("fixedcapacitybytes", 0) or 0)
        if maximum_bytes > 0:
            for _ in range(3):
                footprint = estimatememoryfootprint(self.store.all(), self.graph)
                if footprint.totalbytes <= maximum_bytes or len(self.store) == 0:
                    break
                episode_budget = max(0, maximum_bytes - footprint.graphbytes)
                report = self.capacity_manager.enforcebytes(
                    episode_budget,
                    bytesize=estimateepisodelogicalbytes,
                    policy=str(consolidation_cfg.get("evictionpolicy", "salience")),
                )
                self._maintenance_log.append({"kind": "byte_eviction", **asdict(report)})
                if not report.evicted_ids:
                    break
                rebuild = self.rebuildgraphfromstore()
                self._maintenance_log.append({"kind": "graph_rebuild", **rebuild})
        replay_cfg = self.config.replay
        if bool(replay_cfg.get("enabled", False)):
            interval = max(1, int(replay_cfg.get("intervalwrites", 500)))
            if self._event_write_count % interval == 0:
                report = self.replay_engine.run(
                    budget=int(replay_cfg.get("budgetpercycle", 64)),
                    strategy=str(replay_cfg.get("strategy", "salience")),
                    learningscale=float(replay_cfg.get("learningscale", 0.25)),
                    temporal=bool(replay_cfg.get("temporal", False)),
                )
                self._maintenance_log.append({"kind": "replay", **asdict(report)})
    def clear(self) -> None:
        self.store.clear()
        self.graph.clear()
        self._previous_by_stream.clear()
        self._maintenance_log.clear()
        self._event_write_count = 0
    def rebuildgraphfromstore(self) -> dict[str, int]:
        episodes = sorted(self.store.all(), key=lambda e: (e.timestamp, e.episode_id))
        self.graph.clear()
        previous_by_stream: dict[str, Episode] = {}
        for episode in episodes:
            stream = str(episode.metadata.get("stream_id", "default"))
            previous = previous_by_stream.get(stream)
            self.graph.write(
                episode.code,
                salience=episode.salience,
                previous=previous.code if previous is not None else None,
            )
            previous_by_stream[stream] = episode
        self._previous_by_stream = previous_by_stream
        return {
            "episodes_replayed": len(episodes),
            "association_edges": sum(len(x) for x in self.graph.association.values()),
            "temporal_edges": sum(len(x) for x in self.graph.temporal.values()),
        }
    def query(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> list[QueryResult]:
        encoded = self.encoder.encode(text, metadata=metadata)
        results, _ = self.retriever.retrieve(encoded.code)
        self.recordaccess(results)
        return results
    def querywithtrace(
        self, text: str, *, metadata: Mapping[str, Any] | None = None
    ) -> tuple[list[QueryResult], Any]:
        encoded = self.encoder.encode(text, metadata=metadata)
        results, trace = self.retriever.retrieve(encoded.code)
        self.recordaccess(results)
        return results, trace
    def recordaccess(self, results: list[QueryResult]) -> None:
        now = utcnowiso()
        for result in results:
            episode = result.episode
            episode.metadata["access_count"] = int(episode.metadata.get("access_count", 0)) + 1
            episode.metadata["last_access"] = now
            self.store.add(episode)
    def stats(self) -> dict[str, Any]:
        store_stats = self.store.stats() if hasattr(self.store, "stats") else {"episodes": len(self.store)}
        return {
            "episodes": len(self.store),
            "dimension": self.encoder.dimension,
            "association_nodes": len(self.graph.association),
            "association_edges": sum(len(x) for x in self.graph.association.values()),
            "temporal_nodes": len(self.graph.temporal),
            "temporal_edges": sum(len(x) for x in self.graph.temporal.values()),
            "writes": self.graph.write_count,
            "event_writes": self._event_write_count,
            "association_updates": self.graph.association_update_count,
            "temporal_updates": self.graph.temporal_update_count,
            "store": store_stats,
            "maintenance_events": len(self._maintenance_log),
            "last_maintenance": self._maintenance_log[-1] if self._maintenance_log else None,
        }
    def savegraph(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.graph.statedict(), separators=(",", ":")))
    def loadgraph(self, path: str | Path) -> None:
        self.graph.loadstatedict(json.loads(Path(path).read_text()))
    def close(self) -> None:
        self.store.close()
