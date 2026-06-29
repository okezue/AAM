from __future__ import annotations
from aamemory.memory.accounting import estimateepisodelogicalbytes
from aamemory.memory.capacity import CapacityManager
from aamemory.memory.store import InMemoryEpisodeStore
from aamemory.schema import Episode, SparseCode
def testbytecapacityevictsuntilbudget() -> None:
    store = InMemoryEpisodeStore()
    for index in range(4):
        store.add(
            Episode(
                f"e{index}",
                "x" * (100 + index),
                SparseCode.frommapping(32, {index: 1.0}),
                timestamp=f"2026-01-0{index + 1}T00:00:00+00:00",
            )
        )
    manager = CapacityManager(store)
    target = sum(estimateepisodelogicalbytes(e) for e in list(store.all())[2:])
    report = manager.enforcebytes(target, bytesize=estimateepisodelogicalbytes, policy="fifo")
    assert report.after_bytes <= target
    assert report.evicted_ids[:2] == ("e0", "e1")
