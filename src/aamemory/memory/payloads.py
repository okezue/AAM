from __future__ import annotations
import base64
import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import numpy as np
@dataclass
class QuantizedArrayCodec:
    compression_level: int = 6
    def encode(self, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        encoded: dict[str, Any] = {"codec": "symmetric_int8_zlib", "arrays": {}}
        for name, raw in arrays.items():
            value = np.asarray(raw, dtype=np.float32)
            maximum = float(np.max(np.abs(value))) if value.size else 0.0
            scale = maximum / 127.0 if maximum > 0 else 1.0
            quantized = np.clip(np.round(value / scale), -127, 127).astype(np.int8)
            compressed = zlib.compress(quantized.tobytes(), level=self.compression_level)
            encoded["arrays"][name] = {
                "shape": list(value.shape),
                "scale": scale,
                "data": base64.b64encode(compressed).decode("ascii"),
            }
        return encoded
    def decode(self, payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if payload.get("codec") != "symmetric_int8_zlib":
            raise ValueError("unsupported activation payload codec")
        out: dict[str, np.ndarray] = {}
        for name, item in payload.get("arrays", {}).items():
            compressed = base64.b64decode(item["data"])
            raw = zlib.decompress(compressed)
            quantized = np.frombuffer(raw, dtype=np.int8).reshape(item["shape"])
            out[name] = quantized.astype(np.float32) * float(item["scale"])
        return out
    @staticmethod
    def bytesize(payload: Mapping[str, Any]) -> int:
        return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
