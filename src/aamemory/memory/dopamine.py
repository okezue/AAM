from __future__ import annotations
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any
def num(metadata: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in metadata:
            try:
                value = float(metadata[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return default
def bool(metadata: Mapping[str, Any], *names: str) -> float:
    return float(any(bool(metadata.get(name, False)) for name in names))
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
@dataclass(frozen=True)
class DopamineMetrics:
    prediction_loss: float = 0.0
    entropy: float = 0.0
    evidence_kl: float = 0.0
    novelty: float = 0.0
    redundancy: float = 0.0
    contradiction: float = 0.0
    correction: float = 0.0
    useremphasis: float = 0.0
    downstreamutility: float = 0.0
    source_trust: float = 1.0
    poisonrisk: float = 0.0
    prompt_length: float = 0.0
    recency: float = 0.0
    @classmethod
    def frommetadata(
        cls,
        metadata: Mapping[str, Any] | None,
        *,
        novelty: float = 0.0,
        redundancy: float = 0.0,
    ) -> DopamineMetrics:
        m = dict(metadata or {})
        prompt_length = num(m, "prompt_length", "tokens", "length", default=0.0)
        if not prompt_length and "text_length" in m:
            prompt_length = num(m, "text_length")
        generated = bool(m, "generated", "self_generated", "hypothetical")
        untrusted = bool(m, "untrusted", "prompt_injection", "poison")
        return cls(
            prediction_loss=num(m, "prediction_loss", "loss", "loss_spike"),
            entropy=num(m, "entropy", "predictive_entropy"),
            evidence_kl=num(m, "evidence_kl", "kl", "evidence_impact"),
            novelty=num(m, "novelty", default=novelty),
            redundancy=num(m, "redundancy", default=redundancy),
            contradiction=num(m, "contradiction", "conflict") + bool(m, "contradicts"),
            correction=num(m, "correction", "user_correction") + bool(m, "corrects", "supersedes"),
            useremphasis=num(m, "useremphasis", "user_importance", "remember"),
            downstreamutility=num(m, "utility", "downstreamutility", "reward"),
            source_trust=num(m, "source_trust", "trust", default=1.0),
            poisonrisk=max(num(m, "poisonrisk", "risk"), generated * 0.65, untrusted),
            prompt_length=math.log1p(max(0.0, prompt_length)) / 10.0,
            recency=num(m, "recency", default=0.0),
        )
    def todict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}
class DopamineGate:
    DEFAULT_POSITIVE_WEIGHTS: dict[str, float] = {
        "bias": -0.20,
        "prediction_loss": 0.40,
        "entropy": 0.20,
        "evidence_kl": 0.45,
        "novelty": 0.80,
        "redundancy": -0.45,
        "correction": 0.70,
        "useremphasis": 0.65,
        "downstreamutility": 0.75,
        "source_trust": 0.25,
        "poisonrisk": -1.25,
        "prompt_length": 0.05,
    }
    DEFAULT_NEGATIVE_WEIGHTS: dict[str, float] = {
        "bias": -1.0,
        "contradiction": 1.15,
        "correction": 0.55,
        "poisonrisk": 0.85,
        "source_trust": -0.40,
        "redundancy": 0.35,
    }
    def __init__(
        self,
        *,
        positiveweights: Mapping[str, float] | None = None,
        negativeweights: Mapping[str, float] | None = None,
        minimum: float = 0.02,
        maximum: float = 2.0,
    ) -> None:
        self.positiveweights = {**self.DEFAULT_POSITIVE_WEIGHTS, **dict(positiveweights or {})}
        self.negativeweights = {**self.DEFAULT_NEGATIVE_WEIGHTS, **dict(negativeweights or {})}
        self.minimum = float(minimum)
        self.maximum = float(maximum)
    @classmethod
    def fromconfig(cls, config: Mapping[str, Any] | None = None) -> DopamineGate:
        config = dict(config or {})
        return cls(
            positiveweights=config.get("positiveweights"),
            negativeweights=config.get("negativeweights"),
            minimum=float(config.get("minimum", 0.02)),
            maximum=float(config.get("maximum", 2.0)),
        )
    def positive(self, metrics: DopamineMetrics) -> float:
        raw = self.positiveweights.get("bias", 0.0)
        values = metrics.todict()
        for key, value in values.items():
            raw += self.positiveweights.get(key, 0.0) * value
        return max(self.minimum, min(self.maximum, self.maximum * sigmoid(raw)))
    def negative(self, metrics: DopamineMetrics) -> float:
        raw = self.negativeweights.get("bias", 0.0)
        values = metrics.todict()
        for key, value in values.items():
            raw += self.negativeweights.get(key, 0.0) * value
        return max(0.0, min(self.maximum, self.maximum * sigmoid(raw)))
    def score(self, metrics: DopamineMetrics) -> tuple[float, float]:
        return self.positive(metrics), self.negative(metrics)
