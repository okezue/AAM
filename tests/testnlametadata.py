from __future__ import annotations
import pytest
from aamemory.models.nla import KNOWN_PUBLIC_NLA_CHECKPOINTS, getnlacheckpoint
def testknownnlacheckpointpairsarecomplete() -> None:
    assert len(KNOWN_PUBLIC_NLA_CHECKPOINTS) == 4
    for checkpoint in KNOWN_PUBLIC_NLA_CHECKPOINTS:
        assert checkpoint.layer > 0
        assert checkpoint.d_model > 0
        assert checkpoint.avcheckpoint.endswith("-av")
        assert checkpoint.archeckpoint.endswith("-ar")
        assert getnlacheckpoint(checkpoint.name) == checkpoint
def testunknownnlacheckpointfailsloudly() -> None:
    with pytest.raises(KeyError):
        getnlacheckpoint("not-a-released-family")
