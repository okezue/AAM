from __future__ import annotations
import json
import pytest
from aamemory.config import loadconfig
from aamemory.eval.runner import ExperimentRunner
from aamemory.memory.sqlitestore import SQLiteEpisodeStore
from aamemory.schema import Episode, SparseCode
def testcpusmokerunnerwritesartifacts(tmp_path) -> None:
    config = loadconfig("configs/experiments/l0_cpu_smoke.yaml")
    config.outputdir = str(tmp_path / "run")
    config.evaluation["limit"] = 4
    config.evaluation["bootstrapsamples"] = 50
    report = ExperimentRunner(config).run()
    assert report["aggregate"]["examples"] == 4
    assert (tmp_path / "run" / "per_example.jsonl").exists()
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "resolved_config.yaml").exists()
    assert (tmp_path / "run" / "environment.json").exists()
    first = json.loads((tmp_path / "run" / "per_example.jsonl").read_text().splitlines()[0])
    assert first["memory_bytes_total"] > 0
    assert first["memory_footprint"]["totalbytes"] == first["memory_bytes_total"]
def testsharedsqliterunstartsfreshunlessresumerequested(tmp_path) -> None:
    database = tmp_path / "shared.sqlite3"
    store = SQLiteEpisodeStore(database)
    store.add(Episode("stale", "stale", SparseCode.frommapping(4096, {1: 1.0})))
    store.close()
    config = loadconfig("configs/experiments/l0_cpu_smoke.yaml")
    config.outputdir = str(tmp_path / "run")
    config.memory.store = {"type": "sqlite", "path": str(database)}
    config.evaluation["resetbetweenexamples"] = False
    config.evaluation["limit"] = 1
    config.evaluation["bootstrapsamples"] = 10
    ExperimentRunner(config).run()
    store = SQLiteEpisodeStore(database)
    assert store.get("stale") is None
    assert len(store) > 0
    store.close()
def testresumestorerequiresgraph(tmp_path) -> None:
    config = loadconfig("configs/experiments/l0_cpu_smoke.yaml")
    config.outputdir = str(tmp_path / "run")
    config.evaluation["resetbetweenexamples"] = False
    config.evaluation["resume_store"] = True
    with pytest.raises(ValueError, match="resume_graph_path"):
        ExperimentRunner(config).run()
