from aamemory.config import EncoderConfig, MemoryConfig, RetrievalConfig
from aamemory.memory.hippocampal import HippocampalActivationMemory
from aamemory.schema import MemoryEvent, SourceRef
def cfg() -> MemoryConfig:
    return MemoryConfig(
        variant="aam_v2",
        encoder=EncoderConfig(type="hashing", params={"dimension": 2048, "topk": 64}),
        retrieval=RetrievalConfig(topk=2, recurrencesteps=2, featuretopk=96, confidenceweight=0.1),
        context={
            "encoder": {"dimension": 512, "topk": 32},
            "binding": {"bindingdimension": 512, "topk": 128, "contextweight": 0.65, "bindingweight": 0.55},
            "graph": {"learningrate": 0.12},
        },
        completion={"contextscoreweight": 0.25, "episodenodeweight": 0.2},
    )
def testcontextfeaturesparticipateinretrievaladdress() -> None:
    memory = HippocampalActivationMemory(cfg())
    text_a = "The shared record says artifact zephyr has visible state color blue."
    text_b = "The shared record says artifact zephyr has visible state color red."
    memory.write(
        MemoryEvent(
            "alice",
            text_a,
            source=SourceRef.fortext(text_a),
            metadata={"speaker": "alice", "tool_state": {"tool": "browser"}},
        )
    )
    memory.write(
        MemoryEvent(
            "bob",
            text_b,
            source=SourceRef.fortext(text_b),
            metadata={"speaker": "bob", "tool_state": {"tool": "terminal"}},
        )
    )
    results, _ = memory.querywithtrace(
        "What color is in the shared record for artifact zephyr?",
        metadata={"speaker": "alice", "tool_state": {"tool": "browser"}},
    )
    assert results[0].episode_id == "alice"
    assert results[0].trace["context_score"] > 0
    assert results[0].trace["text_used_for_scoring"] is False
