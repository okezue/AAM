from aamemory.config import EncoderConfig, MemoryConfig, RetrievalConfig
from aamemory.data.overlapidentity import OverlapIdentityDataset
from aamemory.memory.hippocampal import HippocampalActivationMemory
def testepisodeindexpreservesoverlappingmemoryidentity() -> None:
    cfg = MemoryConfig(
        variant="aam_v2",
        encoder=EncoderConfig(type="hashing", params={"dimension": 4096, "topk": 96}),
        retrieval=RetrievalConfig(topk=3, recurrencesteps=2, featuretopk=128),
        context={
            "encoder": {"dimension": 512, "topk": 32},
            "binding": {"bindingdimension": 512, "topk": 160},
        },
        completion={"episodenodeweight": 0.35, "contextscoreweight": 0.25},
    )
    memory = HippocampalActivationMemory(cfg)
    example = next(iter(OverlapIdentityDataset(examplespertask=1, seed=2, overlapcount=5)))
    for event in example.events:
        memory.write(event)
    results, trace = memory.querywithtrace(example.query, metadata=example.metadata)
    assert results[0].episode_id in example.evidence_ids
    assert trace.episode_scores
    assert memory.episodeindex.stats()["episode_nodes"] == 5
