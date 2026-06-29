from __future__ import annotations
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from aamemory.config import ExperimentConfig
from aamemory.data.registry import builddataset
from aamemory.eval.metrics import (
    answerinretrieved,
    exactmatch,
    falseassociationrate,
    hiddenstateretrievalaccuracy,
    hubcapturerate,
    retrievalmetrics,
    sourceattributionaccuracy,
    substringmatch,
    tokenf1,
)
from aamemory.eval.statistics import bootstrapmeanci, clusterbootstrapmeanci
from aamemory.memory.accounting import estimatememoryfootprint
from aamemory.memory.hippocampal import HippocampalActivationMemory
from aamemory.memory.system import ActivationAssociativeMemory
from aamemory.models.factory import buildgenerator
from aamemory.models.injection import TextMemoryInjector
from aamemory.schema import BenchmarkExample
def jsondefault(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
def gitcommit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
def finitemean(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(np.mean(clean)) if clean else float("nan")
class ExperimentRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.outputdir = Path(config.outputdir)
        self.outputdir.mkdir(parents=True, exist_ok=True)
        self.generator = buildgenerator(config.generator)
    def newmemory(self, *, clear_store: bool = False) -> ActivationAssociativeMemory | HippocampalActivationMemory:
        variant = str(getattr(self.config.memory, "variant", "aam_v1")).lower()
        if variant in {"aam_v2", "hippocampal", "hippocampal_activation", "memory_as_activations"}:
            memory = HippocampalActivationMemory(self.config.memory)
        else:
            memory = ActivationAssociativeMemory(self.config.memory)
        if clear_store:
            memory.clear()
        return memory
    def writerunmetadata(self) -> dict[str, Any]:
        resolved = self.config.todict()
        resolved_yaml = yaml.safe_dump(resolved, sort_keys=False)
        config_hash = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
        (self.outputdir / "resolved_config.yaml").write_text(resolved_yaml, encoding="utf-8")
        environment = {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "git_commit": gitcommit(),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "config_sha256": config_hash,
        }
        (self.outputdir / "environment.json").write_text(
            json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return environment
    def runexample(
        self,
        example: BenchmarkExample,
        memory: ActivationAssociativeMemory | HippocampalActivationMemory,
    ) -> dict[str, Any]:
        write_started = time.perf_counter()
        for event in example.events:
            confidence = float(event.metadata.get("confidence", 1.0))
            memory.write(event, confidence=confidence)
        write_seconds = time.perf_counter() - write_started
        read_started = time.perf_counter()
        results, trace = memory.querywithtrace(example.query, metadata=example.metadata)
        read_seconds = time.perf_counter() - read_started
        footprint = estimatememoryfootprint(memory.store.all(), memory.graph).todict()
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "task": example.task,
            "query": example.query,
            "answers": list(example.answers),
            "evidence_ids": list(example.evidence_ids),
            "negative_evidence_ids": list(example.negative_evidence_ids),
            "retrieved_ids": [result.episode_id for result in results],
            "retrieved_scores": [result.score for result in results],
            "retrieved_exact_scores": [result.exact_score for result in results],
            "retrieved_associative_scores": [result.associative_score for result in results],
            "retrieved_temporal_scores": [result.temporal_score for result in results],
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "memory_stats": memory.stats(),
            "memory_footprint": footprint,
            "memory_bytes_total": footprint["totalbytes"],
            "memory_episode_bytes": footprint["episodebytes"],
            "memory_graph_bytes": footprint["graphbytes"],
            "memory_posting_index_bytes": footprint["posting_index_bytes"],
            "recall_steps": len(trace.steps) - 1,
            "recall_active_features": len(trace.final.indices),
            "recall_message_edge_visits": int(trace.diagnostics.get("message_edge_visits", 0)),
            "graph_association_updates": memory.graph.association_update_count,
            "graph_temporal_updates": memory.graph.temporal_update_count,
            "metadata": dict(example.metadata),
        }
        row.update(retrievalmetrics(results, example.evidence_ids, example.negative_evidence_ids))
        row["sourceattributionaccuracy"] = sourceattributionaccuracy(results, example.evidence_ids)
        row["hiddenstateretrievalaccuracy"] = hiddenstateretrievalaccuracy(results, example.evidence_ids)
        row["falseassociationrate"] = falseassociationrate(results, example.evidence_ids, example.negative_evidence_ids)
        row["hubcapturerate"] = hubcapturerate(results, example.metadata.get("hub_ids", ()))
        row["answerinretrieved"] = answerinretrieved(results, example.answers)
        row["primary_memory_substrate"] = getattr(memory, "primary_memory_substrate", "activation_or_text_control")
        row["text_used_for_scoring"] = any(bool(result.trace.get("text_used_for_scoring", False)) for result in results)
        if self.generator is not None:
            injector_config = dict(self.config.evaluation.get("textinjector", {}))
            injection = TextMemoryInjector(**injector_config).renderwithtrace(
                example.query, results
            )
            generation_started = time.perf_counter()
            generation = self.generator.generate(
                injection.prompt,
                system=self.config.evaluation.get("system_prompt"),
                maxtokens=int(self.config.evaluation.get("maxtokens", 256)),
                temperature=float(self.config.evaluation.get("temperature", 0.0)),
            )
            row["generation_seconds"] = time.perf_counter() - generation_started
            row["prediction"] = generation.text
            row["generation_model"] = generation.model
            row["generation_usage"] = dict(generation.usage)
            row["generation_metadata"] = dict(generation.metadata)
            row["injected_memory_ids"] = list(injection.episode_ids)
            row["injected_memory_characters"] = injection.memory_characters
            row["injection_truncated"] = injection.truncated
            row["exactmatch"] = exactmatch(generation.text, example.answers)
            row["substringmatch"] = substringmatch(generation.text, example.answers)
            row["tokenf1"] = tokenf1(generation.text, example.answers)
        return row
    def run(self) -> dict[str, Any]:
        environment = self.writerunmetadata()
        dataset = builddataset(self.config.dataset)
        limit = self.config.evaluation.get("limit")
        reset = bool(self.config.evaluation.get("resetbetweenexamples", True))
        resume_store = bool(self.config.evaluation.get("resume_store", False))
        resume_graph_path = self.config.evaluation.get("resume_graph_path")
        if resume_store and reset:
            raise ValueError("resume_store=true is incompatible with resetbetweenexamples=true")
        if resume_store and not resume_graph_path:
            raise ValueError(
                "resume_store=true requires evaluation.resume_graph_path so store and graph stay consistent"
            )
        records_path = self.outputdir / "per_example.jsonl"
        records: list[dict[str, Any]] = []
        shared_memory = None
        if not reset:
            shared_memory = self.newmemory(clear_store=not resume_store)
            if resume_store:
                shared_memory.loadgraph(str(resume_graph_path))
        with records_path.open("w", encoding="utf-8") as sink:
            for index, example in enumerate(dataset):
                if limit is not None and index >= int(limit):
                    break
                memory = self.newmemory(clear_store=True) if reset else shared_memory
                assert memory is not None
                try:
                    row = self.runexample(example, memory)
                finally:
                    if reset:
                        memory.close()
                records.append(row)
                sink.write(json.dumps(row, ensure_ascii=False, default=jsondefault) + "\n")
        shared_memory_stats = None
        if shared_memory is not None:
            shared_memory_stats = shared_memory.stats()
            shared_memory.savegraph(self.outputdir / "graph.json")
            shared_memory.close()
        metric_names = sorted(
            {
                key
                for row in records
                for key, value in row.items()
                if isinstance(value, (int, float)) and key not in {"recall_steps"}
            }
        )
        aggregate: dict[str, Any] = {"examples": len(records)}
        clusterkey = self.config.evaluation.get("clusterkey")
        for metric in metric_names:
            eligible = [row for row in records if metric in row]
            values = [float(row[metric]) for row in eligible]
            bootstrapsamples = int(self.config.evaluation.get("bootstrapsamples", 2000))
            if clusterkey and all(clusterkey in row.get("metadata", {}) for row in eligible):
                ci = clusterbootstrapmeanci(
                    values,
                    [row["metadata"][clusterkey] for row in eligible],
                    samples=bootstrapsamples,
                    seed=self.config.seed,
                )
            else:
                ci = bootstrapmeanci(
                    values,
                    samples=bootstrapsamples,
                    seed=self.config.seed,
                )
            aggregate[metric] = asdict(ci)
        aggregate["bootstrap_unit"] = f"cluster:{clusterkey}" if clusterkey else "example"
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            by_task[row["task"]].append(row)
        aggregate["by_task"] = {
            task: {
                metric: finitemean(float(row[metric]) for row in rows if metric in row)
                for metric in metric_names
            }
            for task, rows in by_task.items()
        }
        manifest = {
            "name": self.config.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.todict(),
            "environment": environment,
            "shared_memory_stats": shared_memory_stats,
            "aggregate": aggregate,
        }
        (self.outputdir / "summary.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=jsondefault)
        )
        return manifest
