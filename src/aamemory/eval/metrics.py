from __future__ import annotations
import math
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from aamemory.schema import QueryResult
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
def normalizeanswer(text: object) -> str:
    value = str(text).lower()
    value = "".join(ch for ch in value if ch not in string.punctuation)
    value = _ARTICLES.sub(" ", value)
    return " ".join(value.split())
def exactmatch(prediction: str, answers: Sequence[str]) -> float:
    normalized = normalizeanswer(prediction)
    return float(any(normalized == normalizeanswer(answer) for answer in answers))
def substringmatch(prediction: str, answers: Sequence[str]) -> float:
    normalized = normalizeanswer(prediction)
    return float(any(normalizeanswer(answer) in normalized for answer in answers if answer))
def tokenf1(prediction: str, answers: Sequence[str]) -> float:
    prediction_tokens = normalizeanswer(prediction).split()
    if not answers:
        return 0.0
    scores: list[float] = []
    for answer in answers:
        answer_tokens = normalizeanswer(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if not prediction_tokens or not answer_tokens:
            scores.append(float(prediction_tokens == answer_tokens))
            continue
        if overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores)
def retrievalmetrics(
    results: Sequence[QueryResult],
    evidence_ids: Sequence[str],
    negative_evidence_ids: Sequence[str] = (),
) -> dict[str, float]:
    retrieved = [result.episode_id for result in results]
    relevant = set(evidence_ids)
    negative = set(negative_evidence_ids)
    if relevant:
        hits = [int(eid in relevant) for eid in retrieved]
        recall = len(relevant.intersection(retrieved)) / len(relevant)
        precision = sum(hits) / len(retrieved) if retrieved else 0.0
        reciprocal_rank = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
        dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), len(retrieved))))
        ndcg = dcg / ideal if ideal else 0.0
    else:
        recall = precision = reciprocal_rank = ndcg = float("nan")
    return {
        "retrieval_recall": recall,
        "retrieval_precision": precision,
        "retrieval_mrr": reciprocal_rank,
        "retrieval_ndcg": ndcg,
        "negative_hit_rate": (
            sum(eid in negative for eid in retrieved) / len(retrieved) if retrieved and negative else 0.0
        ),
    }
def answerinretrieved(results: Sequence[QueryResult], answers: Sequence[str]) -> float:
    combined = "\n".join(result.episode.text for result in results)
    normalized = normalizeanswer(combined)
    return float(any(normalizeanswer(answer) in normalized for answer in answers if answer))
def hubcapturerate(results: Sequence[QueryResult], hub_ids: Iterable[str]) -> float:
    hub_set = set(hub_ids)
    if not results:
        return 0.0
    return sum(result.episode_id in hub_set for result in results) / len(results)
def sourceattributionaccuracy(results: Sequence[QueryResult], evidence_ids: Sequence[str]) -> float:
    if not evidence_ids:
        return float("nan")
    return float(bool(results and results[0].episode_id in set(evidence_ids)))
def hiddenstateretrievalaccuracy(results: Sequence[QueryResult], evidence_ids: Sequence[str]) -> float:
    if not evidence_ids:
        return float("nan")
    gold = set(evidence_ids)
    for result in results:
        hidden_id = str(
            result.episode.metadata.get(
                "hidden_state_id",
                result.episode.payload.get("hidden_state_id", result.episode_id),
            )
        )
        if hidden_id in gold or result.episode_id in gold:
            return 1.0
    return 0.0
def falseassociationrate(
    results: Sequence[QueryResult],
    evidence_ids: Sequence[str],
    negative_evidence_ids: Sequence[str] = (),
) -> float:
    if not results:
        return 0.0
    positive = set(evidence_ids)
    negative = set(negative_evidence_ids)
    false = 0
    for result in results:
        if result.episode_id in negative:
            false += 1
        elif positive and result.episode_id not in positive:
            false += 1
    return false / len(results)
def bytesperretainedfact(memory_bytes: float, retained_correct: float) -> float:
    if retained_correct <= 0:
        return float("inf")
    return float(memory_bytes) / float(retained_correct)
def hallucinatedreplayamplificationrate(before_false_mass: float, after_false_mass: float) -> float:
    if before_false_mass <= 0:
        return float(after_false_mass > 0)
    return max(0.0, (after_false_mass - before_false_mass) / before_false_mass)
def forgettingcurve(rows: Sequence[Mapping[str, object]], *, distance_key: str = "stream_distance", score_key: str = "answerinretrieved") -> dict[int, float]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), Mapping) else {}
        distance = int(metadata.get(distance_key, row.get(distance_key, 0)))
        score = float(row.get(score_key, 0.0))
        buckets.setdefault(distance, []).append(score)
    return {distance: sum(values) / len(values) for distance, values in sorted(buckets.items()) if values}
