from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from aamemory.eval.statistics import pairedbootstrapdifference, pairedrandomizationtest
def loadmetric(path: Path, metric: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "example_id" not in row:
            raise ValueError(f"{path}:{line_number} has no example_id")
        example_id = str(row["example_id"])
        if example_id in out:
            raise ValueError(f"duplicate example_id {example_id!r} in {path}")
        if metric in row and row[metric] is not None:
            out[example_id] = float(row[metric])
    return out
def compare(
    baseline_path: Path,
    treatment_path: Path,
    *,
    metric: str,
    bootstrapsamples: int,
    randomization_samples: int,
    seed: int,
    missing_id_sample: int = 20,
) -> dict[str, Any]:
    baseline = loadmetric(baseline_path, metric)
    treatment = loadmetric(treatment_path, metric)
    baseline_ids = set(baseline)
    treatment_ids = set(treatment)
    paired_ids = sorted(baseline_ids & treatment_ids)
    baseline_only = sorted(baseline_ids - treatment_ids)
    treatment_only = sorted(treatment_ids - baseline_ids)
    if not paired_ids:
        raise ValueError(
            f"no paired examples for metric {metric!r}; "
            f"baseline rows={len(baseline)}, treatment rows={len(treatment)}"
        )
    baseline_values = [baseline[example_id] for example_id in paired_ids]
    treatment_values = [treatment[example_id] for example_id in paired_ids]
    ci = pairedbootstrapdifference(
        baseline_values,
        treatment_values,
        samples=bootstrapsamples,
        seed=seed,
    )
    p_value = pairedrandomizationtest(
        baseline_values,
        treatment_values,
        samples=randomization_samples,
        seed=seed,
    )
    return {
        "metric": metric,
        "baseline_file": str(baseline_path),
        "treatment_file": str(treatment_path),
        "baseline_examples_with_metric": len(baseline),
        "treatment_examples_with_metric": len(treatment),
        "paired_examples": len(paired_ids),
        "baseline_only_count": len(baseline_only),
        "treatment_only_count": len(treatment_only),
        "baseline_only_ids_sample": baseline_only[:missing_id_sample],
        "treatment_only_ids_sample": treatment_only[:missing_id_sample],
        "treatment_minus_baseline": ci.__dict__,
        "paired_randomization_p": p_value,
    }
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("treatment")
    parser.add_argument("--metric", default="retrieval_recall")
    parser.add_argument("--bootstrapsamples", type=int, default=10000)
    parser.add_argument("--randomization-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-id-sample", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = compare(
        Path(args.baseline),
        Path(args.treatment),
        metric=args.metric,
        bootstrapsamples=args.bootstrapsamples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
        missing_id_sample=args.missing_id_sample,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
