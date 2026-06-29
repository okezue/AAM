from aamemory.config import EncoderConfig, MemoryConfig, RetrievalConfig
from aamemory.memory.dopamine import DopamineGate, DopamineMetrics
from aamemory.memory.hippocampal import HippocampalActivationMemory
from aamemory.schema import MemoryEvent, SourceRef
def testdopaminegateseparatessalientfrompoisonedwrites() -> None:
    gate = DopamineGate()
    good = DopamineMetrics(novelty=1.0, useremphasis=1.0, source_trust=1.0, poisonrisk=0.0)
    bad = DopamineMetrics(novelty=1.0, useremphasis=1.0, source_trust=0.0, poisonrisk=1.0)
    assert gate.positive(good) > gate.positive(bad)
    assert gate.negative(bad) > gate.negative(good)
def testneurogenesisallocatesimmaturefeatureundernovelinterference() -> None:
    cfg = MemoryConfig(
        variant="aam_v2",
        encoder=EncoderConfig(type="hashing", params={"dimension": 1024, "topk": 48}),
        retrieval=RetrievalConfig(topk=1, recurrencesteps=1, featuretopk=64),
        context={"encoder": {"dimension": 256, "topk": 16}, "binding": {"bindingdimension": 256, "topk": 96}},
        neurogenesis={"enabled": True, "reservedfeatures": 8, "birththreshold": 0.2, "maturationhits": 2},
    )
    memory = HippocampalActivationMemory(cfg)
    text = "A high novelty latent feature QUASAR-1199 binds to phase-blue under interference."
    memory.write(
        MemoryEvent(
            "novel",
            text,
            source=SourceRef.fortext(text),
            metadata={"novelty": 1.0, "interference": 0.9, "retrieval_error": 1.0},
        )
    )
    stats = memory.stats()["neurogenesis"]
    assert stats["births"] == 1
    assert stats["immature_features"] == 1
