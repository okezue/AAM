from __future__ import annotations
from aamemory.config import EncoderConfig, GraphConfig, MemoryConfig
from aamemory.memory.replay import ReplayEngine
from aamemory.memory.system import ActivationAssociativeMemory
from aamemory.schema import MemoryEvent, SourceRef
def testreplayrejectsgeneratedsourcefreememory() -> None:
    memory = ActivationAssociativeMemory(
        MemoryConfig(
            encoder=EncoderConfig(
                type="hashing", params={"dimension": 2048, "topk": 32, "seed": 1}
            ),
            graph=GraphConfig(rule="hebb", learningrate=0.1),
        )
    )
    verified_text = "Verified key ALPHA maps to blue."
    memory.write(
        MemoryEvent(
            "verified",
            verified_text,
            source=SourceRef.fortext(verified_text, document_id="verified"),
        )
    )
    memory.write(
        MemoryEvent(
            "generated",
            "Generated speculation says ALPHA maps to red.",
            metadata={"generated": True},
        )
    )
    report = ReplayEngine(memory.store, memory.graph).run(budget=10, strategy="salience")
    assert report.verified == 1
    assert report.rejected == 1
    memory.close()
