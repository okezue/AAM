from aamemory.config import EncoderConfig, MemoryConfig, RetrievalConfig
from aamemory.memory.hippocampal import HippocampalActivationMemory
from aamemory.schema import MemoryEvent, SourceRef
def testprospectivetraceshavenofactualauthorityandnotextprimary() -> None:
    cfg = MemoryConfig(
        variant="aam_v2",
        encoder=EncoderConfig(type="hashing", params={"dimension": 1024, "topk": 48}),
        retrieval=RetrievalConfig(topk=2, recurrencesteps=1, featuretopk=64),
        context={"encoder": {"dimension": 256, "topk": 16}, "binding": {"bindingdimension": 256, "topk": 96}},
        prospection={"enabled": True, "maxprompts": 3, "authority": 0.05},
    )
    memory = HippocampalActivationMemory(cfg)
    text = "Project atlas will need the notebook during beta review."
    memory.write(MemoryEvent("support", text, source=SourceRef.fortext(text), metadata={"topic": "atlas"}))
    traces = memory.simulatefutureprompts(recent_query="What will project atlas need?", task_state={"topic": "atlas"})
    assert traces
    assert all(trace.factual_authority == 0.0 for trace in traces)
    assert memory.stats()["primary_memory_substrate"] == "activation_engram"
    results, _ = memory.querywithtrace("What will project atlas need?", metadata={"topic": "atlas"})
    assert all(result.trace["text_used_for_scoring"] is False for result in results)
