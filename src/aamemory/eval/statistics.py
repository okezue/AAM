from __future__ import annotations
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    level: float
    samples: int
def bootstrapmeanci(
    values: Sequence[float],
    *,
    samples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if samples <= 0:
        raise ValueError("samples must be positive")
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if clean.size == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), level, samples)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = max(1, min(512, samples))
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, clean.size, size=(stop - start, clean.size))
        means[start:stop] = clean[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        estimate=float(clean.mean()),
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        level=level,
        samples=samples,
    )
def clusterbootstrapmeanci(
    values: Sequence[float],
    cluster_ids: Sequence[Hashable],
    *,
    samples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must have equal length")
    if samples <= 0:
        raise ValueError("samples must be positive")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    clean_values: list[float] = []
    for value, cluster in zip(values, cluster_ids, strict=True):
        if math.isfinite(value):
            grouped[cluster].append(float(value))
            clean_values.append(float(value))
    if not grouped:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), level, samples)
    keys = list(grouped)
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=np.float64)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch_size = max(1, min(512, samples))
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        draws = rng.integers(0, len(keys), size=(stop - start, len(keys)))
        means[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return ConfidenceInterval(
        estimate=float(np.mean(clean_values)),
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1.0 - alpha)),
        level=level,
        samples=samples,
    )
def pairedbootstrapdifference(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    if len(baseline) != len(treatment):
        raise ValueError("paired samples must have equal length")
    differences = [t - b for b, t in zip(baseline, treatment, strict=True)]
    return bootstrapmeanci(differences, samples=samples, level=level, seed=seed)
def pairedrandomizationtest(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = 100_000,
    seed: int = 0,
) -> float:
    if len(baseline) != len(treatment):
        raise ValueError("paired samples must have equal length")
    differences = np.asarray(treatment, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    batch_size = max(1, min(1024, samples))
    choices = np.asarray([-1.0, 1.0])
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        signs = rng.choice(choices, size=(size, differences.size))
        permuted = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed))
    return (exceed + 1) / (samples + 1)
def holmbonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    valid = sorted(
        ((name, p) for name, p in p_values.items() if math.isfinite(p)), key=lambda item: item[1]
    )
    m = len(valid)
    output: dict[str, dict[str, float | bool]] = {}
    still_rejecting = True
    adjusted_running = 0.0
    for rank, (name, p_value) in enumerate(valid, start=1):
        threshold = alpha / (m - rank + 1)
        reject = still_rejecting and p_value <= threshold
        if not reject:
            still_rejecting = False
        adjusted_running = max(adjusted_running, min(1.0, (m - rank + 1) * p_value))
        output[name] = {
            "p_value": p_value,
            "adjusted_p_value": adjusted_running,
            "threshold": threshold,
            "reject": reject,
        }
    return output
