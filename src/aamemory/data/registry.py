from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from aamemory.data.base import BenchmarkDataset
def builddataset(config: Mapping[str, Any]) -> BenchmarkDataset:
    kind = str(config.get("type", "synthetic")).lower().replace("-", "_")
    params = dict(config.get("params", {}))
    if kind == "synthetic":
        from aamemory.data.synthetic import SyntheticMemoryDataset
        return SyntheticMemoryDataset(**params)
    if kind in {"chunked", "segmented"}:
        from aamemory.data.chunked import ChunkedDataset
        return ChunkedDataset(**params)
    if kind in {"continual", "continual_retention"}:
        from aamemory.data.continual import ContinualRetentionDataset
        return ContinualRetentionDataset(**params)
    if kind == "longmemeval":
        from aamemory.data.longmemeval import LongMemEvalDataset
        return LongMemEvalDataset(**params)
    if kind in {"longmemeval_v2", "longmemeval2"}:
        from aamemory.data.longmemevalv2 import LongMemEvalV2Dataset
        return LongMemEvalV2Dataset(**params)
    if kind == "locomo":
        from aamemory.data.locomo import LoCoMoDataset
        return LoCoMoDataset(**params)
    if kind == "longbench":
        from aamemory.data.longbench import LongBenchDataset
        return LongBenchDataset(**params)
    if kind in {"longbench_v2", "longbench2"}:
        from aamemory.data.longbenchv2 import LongBenchV2Dataset
        return LongBenchV2Dataset(**params)
    if kind == "ruler":
        from aamemory.data.ruler import RulerDataset
        return RulerDataset(**params)
    if kind in {"contextual", "contextual_memory", "context_association"}:
        from aamemory.data.contextualmemory import ContextualMemoryDataset
        return ContextualMemoryDataset(**params)
    if kind in {"overlap_identity", "synapse_identity", "overlapping_memory"}:
        from aamemory.data.overlapidentity import OverlapIdentityDataset
        return OverlapIdentityDataset(**params)
    if kind in {"prospection", "future_simulation", "future_prompt"}:
        from aamemory.data.prospection import ProspectionDataset
        return ProspectionDataset(**params)
    if kind in {"neurogenesis", "feature_birth"}:
        from aamemory.data.neurogenesis import NeurogenesisDataset
        return NeurogenesisDataset(**params)
    if kind in {"state_tensor", "pseudo_multimodal_state"}:
        from aamemory.data.statetensor import StateTensorDataset
        return StateTensorDataset(**params)
    raise ValueError(f"unknown dataset type: {kind}")
