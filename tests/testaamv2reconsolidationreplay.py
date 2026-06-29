from aamemory.config import EncoderConfig, MemoryConfig, RetrievalConfig
from aamemory.memory.hippocampal import HippocampalActivationMemory
from aamemory.schema import MemoryEvent, SourceRef
def cfg() -> MemoryConfig:
    return MemoryConfig(
        variant="aam_v2",
        encoder=EncoderConfig(type="hashing", params={"dimension": 2048, "topk": 64}),
        retrieval=RetrievalConfig(topk=3, recurrencesteps=1, featuretopk=96, confidenceweight=0.3),
        context={"encoder": {"dimension": 256, "topk": 16}, "binding": {"bindingdimension": 256, "topk": 112}},
        replay={"enabled": False, "budgetpercycle": 8},
    )
def testcorrectionsupersedesoldmemoryanddepressesauthority() -> None:
    memory = HippocampalActivationMemory(cfg())
    old = "Verified source: key K maps to blue."
    new = "Correction: key K now maps to green."
    memory.write(MemoryEvent("old", old, source=SourceRef.fortext(old), metadata={"entity": "K"}))
    memory.write(
        MemoryEvent(
            "new",
            new,
            source=SourceRef.fortext(new),
            metadata={"entity": "K", "corrects": "old", "correction": 1.0},
        )
    )
    assert memory.store.get("old").metadata["engram_status"] == "superseded"
    results, _ = memory.querywithtrace("What does key K map to now?", metadata={"entity": "K"})
    assert results[0].episode_id == "new"
def testverifiedreplayquarantinesgeneratedhypotheticaltrace() -> None:
    memory = HippocampalActivationMemory(cfg())
    true = "Verified source: key Q maps to amber."
    false = "Generated speculation: key Q maps to violet."
    memory.write(MemoryEvent("true", true, source=SourceRef.fortext(true), metadata={"entity": "Q"}))
    memory.write(MemoryEvent("false", false, metadata={"generated": True, "hypothetical": True, "entity": "Q"}))
    report = memory.replaycycle(budget=8)
    assert report["verified"] >= 1
    assert report["quarantined"] >= 1
    assert "false" in memory.replay_engine.quarantined_ids
