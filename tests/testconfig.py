from __future__ import annotations
from aamemory.config import loadconfig
def testconfigenvironmentexpansion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODEL", "model-x")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
name: test
dataset: {type: synthetic, params: {examplespertask: 1}}
memory: {}
generator:
  type: openai
  params:
    model: ${TEST_MODEL:-fallback}
outputdir: runs/test
"""
    )
    config = loadconfig(path)
    assert config.generator["params"]["model"] == "model-x"
