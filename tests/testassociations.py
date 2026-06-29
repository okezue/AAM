from __future__ import annotations
from aamemory.config import GraphConfig, RetrievalConfig
from aamemory.memory.associations import SparseAssociationGraph
from aamemory.schema import SparseCode
def code(*indices: int, dimension: int = 16) -> SparseCode:
    return SparseCode.frommapping(dimension, {index: 1.0 for index in indices}).normalized()
def testtwosteppatterncompletionreachesindirectfeature() -> None:
    graph = SparseAssociationGraph(
        16,
        GraphConfig(
            learningrate=1.0,
            temporallearningrate=0.0,
            rule="hebb",
            normalizeafterwrite=False,
            maxdegree=16,
            maxpairfeatures=16,
            hubpenalty=0.0,
        ),
    )
    graph.write(code(1, 2))
    graph.write(code(2, 3))
    trace = graph.recall(
        code(1),
        RetrievalConfig(
            recurrencesteps=2,
            featuretopk=8,
            queryanchor=1.0,
            associationstrength=1.0,
            temporalstrength=0.0,
            threshold=0.0,
        ),
    )
    assert 2 in trace.final.indices
    assert 3 in trace.final.indices
def testtemporallinksupportsforwardrecall() -> None:
    graph = SparseAssociationGraph(
        16,
        GraphConfig(
            learningrate=0.0,
            temporallearningrate=1.0,
            rule="hebb",
            normalizeafterwrite=False,
            maxdegree=16,
            maxpairfeatures=16,
            hubpenalty=0.0,
        ),
    )
    graph.write(code(4), previous=code(1))
    trace = graph.recall(
        code(1),
        RetrievalConfig(
            recurrencesteps=1,
            featuretopk=8,
            queryanchor=0.1,
            associationstrength=0.0,
            temporalstrength=1.0,
        ),
    )
    assert 4 in trace.final.indices
def testzeroandshufflededgecontrolsaredeterministic() -> None:
    learned = SparseAssociationGraph(
        16,
        GraphConfig(
            learningrate=1.0,
            temporallearningrate=0.0,
            rule="hebb",
            normalizeafterwrite=False,
            maxdegree=16,
            maxpairfeatures=16,
            edgecondition="learned",
        ),
    )
    learned.write(code(1, 2))
    learned.write(code(2, 3))
    cfg = RetrievalConfig(
        recurrencesteps=2,
        featuretopk=8,
        queryanchor=1.0,
        associationstrength=1.0,
        temporalstrength=0.0,
    )
    assert 3 in learned.recall(code(1), cfg).final.indices
    learned.config.edgecondition = "zero"
    zero = learned.recall(code(1), cfg).final
    assert zero.indices == (1,)
    learned.config.edgecondition = "shuffled_degree_preserving"
    learned.config.edgeseed = 9
    first = learned.recall(code(1), cfg).final
    second = learned.recall(code(1), cfg).final
    assert first == second
