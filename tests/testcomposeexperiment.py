from __future__ import annotations
import subprocess
import sys
from aamemory.config import loadconfig
def testcomposeexperimentcreatesrunnableconfig(tmp_path) -> None:
    destination = tmp_path / "composed.yaml"
    subprocess.run(
        [
            sys.executable,
            "scripts/composeexperiment.py",
            "configs/experiments/l0_cpu_smoke.yaml",
            "--model",
            "configs/models/hashing.yaml",
            "--dataset",
            "configs/datasets/synthetic.yaml",
            "--name",
            "composed-test",
            "--output",
            str(destination),
        ],
        check=True,
    )
    config = loadconfig(destination)
    assert config.name == "composed-test"
    assert config.memory.encoder.type == "hashing"
    assert config.dataset["type"] == "synthetic"
