from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class BeaconPayload:
    model_id: str
    compression_ratio: int
    chunksize: int
    layerwise_kv: Mapping[str, Any]
    positions: Mapping[str, Any]
class ActivationBeaconPayloadAdapter:
    def capture(self, model: Any, text: str) -> BeaconPayload:
        raise NotImplementedError(
            "Wire this method to the selected FlagEmbedding Activation Beacon checkpoint."
        )
