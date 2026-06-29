from __future__ import annotations
from aamemory.encoding.hashing import HashingFeatureEncoder
def testhashingencoderisdeterministicandaliasaware() -> None:
    encoder = HashingFeatureEncoder(dimension=4096, topk=64, seed=42)
    first = encoder.encode("Mira's favourite colour is blue.").code
    second = encoder.encode("Mira's favorite color is blue.").code
    repeated = encoder.encode("Mira's favourite colour is blue.").code
    assert first == repeated
    assert first.dot(second) > 0.2
    assert len(first.indices) <= 64
