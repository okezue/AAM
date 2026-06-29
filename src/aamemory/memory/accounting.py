from __future__ import annotations
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.schema import Episode
def jsonbytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
@dataclass(frozen=True)
class MemoryFootprint:
    text_bytes: int
    sparse_index_bytes: int
    sparse_value_bytes: int
    payload_bytes: int
    source_bytes: int
    metadata_bytes: int
    posting_index_bytes: int
    graph_edge_bytes: int
    graph_node_state_bytes: int
    episode_count: int
    posting_entries: int
    graph_edges: int
    graph_nodes: int
    @property
    def episodebytes(self) -> int:
        return (
            self.text_bytes
            + self.sparse_index_bytes
            + self.sparse_value_bytes
            + self.payload_bytes
            + self.source_bytes
            + self.metadata_bytes
        )
    @property
    def graphbytes(self) -> int:
        return self.graph_edge_bytes + self.graph_node_state_bytes
    @property
    def totalbytes(self) -> int:
        return self.episodebytes + self.posting_index_bytes + self.graphbytes
    def todict(self) -> dict[str, int]:
        raw = {key: int(value) for key, value in asdict(self).items()}
        raw["episodebytes"] = int(self.episodebytes)
        raw["graphbytes"] = int(self.graphbytes)
        raw["totalbytes"] = int(self.totalbytes)
        return raw
def estimatememoryfootprint(
    episodes: Iterable[Episode],
    graph: SparseAssociationGraph,
    *,
    feature_index_bytes: int = 4,
    feature_value_bytes: int = 4,
    episode_pointer_bytes: int = 8,
    graph_weight_bytes: int = 4,
) -> MemoryFootprint:
    episode_list = list(episodes)
    text_bytes = sparse_index_bytes = sparse_value_bytes = 0
    payload_bytes = source_bytes = metadata_bytes = posting_entries = 0
    for episode in episode_list:
        text_bytes += len(episode.text.encode("utf-8"))
        active = len(episode.code.indices)
        sparse_index_bytes += active * feature_index_bytes
        sparse_value_bytes += active * feature_value_bytes
        posting_entries += active
        payload_bytes += jsonbytes(dict(episode.payload))
        source_bytes += jsonbytes(
            {
                "uri": episode.source.uri,
                "document_id": episode.source.document_id,
                "start": episode.source.start,
                "end": episode.source.end,
                "checksum": episode.source.checksum,
                "metadata": dict(episode.source.metadata),
            }
        )
        metadata_bytes += jsonbytes(episode.metadata)
        metadata_bytes += len(episode.episode_id.encode("utf-8"))
        metadata_bytes += len(episode.timestamp.encode("utf-8")) + 8
    posting_index_bytes = posting_entries * (feature_index_bytes + episode_pointer_bytes)
    association_edges = sum(len(neighbors) for neighbors in graph.association.values())
    temporal_edges = sum(len(neighbors) for neighbors in graph.temporal.values())
    graph_edges = association_edges + temporal_edges
    graph_edge_bytes = graph_edges * (2 * feature_index_bytes + graph_weight_bytes)
    graph_nodes_set = (
        set(graph.association)
        | set(graph.temporal)
        | set(graph.activity_mean)
        | set(graph.activity_second_moment)
    )
    graph_nodes = len(graph_nodes_set)
    graph_node_state_bytes = graph_nodes * (feature_index_bytes + 2 * graph_weight_bytes)
    return MemoryFootprint(
        text_bytes=text_bytes,
        sparse_index_bytes=sparse_index_bytes,
        sparse_value_bytes=sparse_value_bytes,
        payload_bytes=payload_bytes,
        source_bytes=source_bytes,
        metadata_bytes=metadata_bytes,
        posting_index_bytes=posting_index_bytes,
        graph_edge_bytes=graph_edge_bytes,
        graph_node_state_bytes=graph_node_state_bytes,
        episode_count=len(episode_list),
        posting_entries=posting_entries,
        graph_edges=graph_edges,
        graph_nodes=graph_nodes,
    )
def estimateepisodelogicalbytes(
    episode: Episode,
    *,
    feature_index_bytes: int = 4,
    feature_value_bytes: int = 4,
    episode_pointer_bytes: int = 8,
) -> int:
    active = len(episode.code.indices)
    return int(
        len(episode.text.encode("utf-8"))
        + active * (feature_index_bytes + feature_value_bytes)
        + active * (feature_index_bytes + episode_pointer_bytes)
        + jsonbytes(dict(episode.payload))
        + jsonbytes(
            {
                "uri": episode.source.uri,
                "document_id": episode.source.document_id,
                "start": episode.source.start,
                "end": episode.source.end,
                "checksum": episode.source.checksum,
                "metadata": dict(episode.source.metadata),
            }
        )
        + jsonbytes(episode.metadata)
        + len(episode.episode_id.encode("utf-8"))
        + len(episode.timestamp.encode("utf-8"))
        + 8
    )
