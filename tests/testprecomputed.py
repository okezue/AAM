from __future__ import annotations
import json
import pytest
from aamemory.encoding.precomputed import PrecomputedFeatureEncoder, textsha256
def testprecomputedencoderroundtripandexacttextkey(tmp_path) -> None:
    text = "A source-backed episode."
    path = tmp_path / "features.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": text,
                "textsha256": textsha256(text),
                "dimension": 32,
                "indices": [2, 9],
                "values": [0.5, 1.0],
                "payload": {"kind": "test"},
                "diagnostics": {"model_revision": "abc123"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    encoder = PrecomputedFeatureEncoder(path=path, dimension=32, verifytext=True)
    result = encoder.encode(text)
    assert result.code.indices == (2, 9)
    assert result.payload["kind"] == "test"
    assert result.diagnostics["model_revision"] == "abc123"
    with pytest.raises(KeyError):
        encoder.encode(text + " changed")
def testprecomputedencoderrejectsdimensionmismatch(tmp_path) -> None:
    path = tmp_path / "features.jsonl"
    path.write_text(
        json.dumps(
            {
                "textsha256": textsha256("x"),
                "dimension": 16,
                "indices": [1],
                "values": [1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PrecomputedFeatureEncoder(path=path, dimension=32)
