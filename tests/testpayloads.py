from __future__ import annotations
import numpy as np
from aamemory.memory.payloads import QuantizedArrayCodec
def testquantizedarraycodecroundtrip() -> None:
    rng = np.random.default_rng(0)
    arrays = {"layer": rng.standard_normal((3, 5), dtype=np.float32)}
    codec = QuantizedArrayCodec()
    payload = codec.encode(arrays)
    decoded = codec.decode(payload)
    assert decoded["layer"].shape == arrays["layer"].shape
    assert np.max(np.abs(decoded["layer"] - arrays["layer"])) < 0.03
