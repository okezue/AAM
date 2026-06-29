from __future__ import annotations
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
def expandenv(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [expandenv(v) for v in value]
    if isinstance(value, dict):
        return {k: expandenv(v) for k, v in value.items()}
    return value
@dataclass
class EncoderConfig:
    type: str = "hashing"
    params: dict[str, Any] = field(default_factory=dict)
@dataclass
class GraphConfig:
    learningrate: float = 0.12
    temporallearningrate: float = 0.08
    decay: float = 0.001
    decayinterval: int = 100
    rule: str = "covariance"
    maxdegree: int = 128
    normalizeafterwrite: bool = True
    allownegativeedges: bool = False
    hubpenalty: float = 0.5
    maxpairfeatures: int = 128
    edgecondition: str = "learned"
    edgeseed: int = 0
@dataclass
class RetrievalConfig:
    topk: int = 5
    candidatelimit: int = 500
    recurrencesteps: int = 2
    featuretopk: int = 128
    queryanchor: float = 1.0
    associationstrength: float = 0.65
    temporalstrength: float = 0.15
    exactweight: float = 0.65
    associativeweight: float = 0.3
    temporalweight: float = 0.05
    recencyweight: float = 0.0
    confidenceweight: float = 0.0
    threshold: float = 0.0
    normalizemessages: bool = True
    use_signed_messages: bool = False
@dataclass
class SalienceConfig:
    base: float = 0.5
    surpriseweight: float = 0.25
    taskweight: float = 0.15
    userweight: float = 0.2
    noveltyweight: float = 0.25
    redundancyweight: float = 0.15
    minimum: float = 0.05
    maximum: float = 2.0
@dataclass
class MemoryConfig:
    variant: str = "aam_v1"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    store: dict[str, Any] = field(default_factory=lambda: {"type": "memory"})
    replay: dict[str, Any] = field(default_factory=dict)
    consolidation: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    dopamine: dict[str, Any] = field(default_factory=dict)
    allocation: dict[str, Any] = field(default_factory=dict)
    episodeindex: dict[str, Any] = field(default_factory=dict)
    completion: dict[str, Any] = field(default_factory=dict)
    neurogenesis: dict[str, Any] = field(default_factory=dict)
    prospection: dict[str, Any] = field(default_factory=dict)
    reconsolidation: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    seed: int = 0
    dataset: dict[str, Any] = field(default_factory=dict)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generator: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    evaluation: dict[str, Any] = field(default_factory=dict)
    outputdir: str = "runs/unnamed"
    notes: str = ""
    def todict(self) -> dict[str, Any]:
        return asdict(self)
def coercedataclass(data: Mapping[str, Any]) -> ExperimentConfig:
    memory_raw = dict(data.get("memory", {}))
    encoder_raw = dict(memory_raw.get("encoder", {}))
    graph_raw = dict(memory_raw.get("graph", {}))
    retrieval_raw = dict(memory_raw.get("retrieval", {}))
    salience_raw = dict(memory_raw.get("salience", {}))
    memory = MemoryConfig(
        variant=str(memory_raw.get("variant", "aam_v1")),
        encoder=EncoderConfig(**encoder_raw),
        graph=GraphConfig(**graph_raw),
        retrieval=RetrievalConfig(**retrieval_raw),
        salience=SalienceConfig(**salience_raw),
        store=dict(memory_raw.get("store", {"type": "memory"})),
        replay=dict(memory_raw.get("replay", {})),
        consolidation=dict(memory_raw.get("consolidation", {})),
        context=dict(memory_raw.get("context", {})),
        dopamine=dict(memory_raw.get("dopamine", {})),
        allocation=dict(memory_raw.get("allocation", {})),
        episodeindex=dict(memory_raw.get("episodeindex", {})),
        completion=dict(memory_raw.get("completion", {})),
        neurogenesis=dict(memory_raw.get("neurogenesis", {})),
        prospection=dict(memory_raw.get("prospection", {})),
        reconsolidation=dict(memory_raw.get("reconsolidation", {})),
        safety=dict(memory_raw.get("safety", {})),
    )
    return ExperimentConfig(
        name=str(data.get("name", "unnamed")),
        seed=int(data.get("seed", 0)),
        dataset=dict(data.get("dataset", {})),
        memory=memory,
        generator=dict(data.get("generator", {"type": "none"})),
        evaluation=dict(data.get("evaluation", {})),
        outputdir=str(data.get("outputdir", f"runs/{data.get('name', 'unnamed')}")),
        notes=str(data.get("notes", "")),
    )
def loadconfig(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text()) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError(f"unsupported config format: {path.suffix}")
    raw = expandenv(raw)
    return coercedataclass(raw)
