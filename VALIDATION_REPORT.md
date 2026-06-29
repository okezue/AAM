# Validation report

Generated after the AAM-v2 Hippocampal Activation Memory update.

## What was validated locally

- Python compilation of all `src`, `tests`, and `scripts` files.
- Full unit test suite: **36 passed**.
- YAML parsing of all configs under `configs/`: **68 parsed successfully**.
- AAM-v2 CPU smoke run:
  - `configs/experiments/aam_v2_cpu_smoke.yaml`
  - executed with `--limit 6`
  - wrote `runs/aam_v2_cpu_smoke/summary.json` and `per_example.jsonl`.
- No-text-primary invariant check:
  - `scripts/checknotextprimary.py runs/aam_v2_cpu_smoke`
  - result: `ok: true`, zero violations.
- AAM-v2 checkpoint-suite dry run:
  - `scripts/runcheckpointsuite.py --dry-run`
  - validated C0 through C13 config paths.
- Ablation materialization smoke check:
  - `configs/ablations/aam_v2_core_matrix.yaml`
  - materialized three configs successfully.
- Dataset prepare/index checks for new diagnostic datasets:
  - `contextualmemory`
  - `overlapidentity`
  - `prospection`
  - `neurogenesis`
  - `statetensor`

## Commands used

```bash
python -m compileall -q src tests scripts
PYTHONPATH=src pytest -q
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from aamemory.config import loadconfig
bad=[]
for p in sorted(Path('configs').rglob('*.yaml')):
    try: loadconfig(p)
    except Exception as e: bad.append((str(p), repr(e)))
assert not bad, bad
PY
PYTHONPATH=src python -m aamemory.cli run configs/experiments/aam_v2_cpu_smoke.yaml --limit 6
PYTHONPATH=src python scripts/checknotextprimary.py runs/aam_v2_cpu_smoke
PYTHONPATH=src python scripts/runcheckpointsuite.py --dry-run
PYTHONPATH=src python scripts/materializeablationmatrix.py configs/ablations/aam_v2_core_matrix.yaml --outputdir /tmp/materialized_test --max-runs 3
for d in contextualmemory overlapidentity prospection neurogenesis statetensor; do
  PYTHONPATH=src python scripts/preparedatasets.py "$d" --limit 2 --output-root /tmp/prepared
done
```

## Not validated in this environment

The following require external compute, credentials, downloads, or model licenses and are intentionally left behind explicit interfaces/configs rather than silently mocked as scientific evidence:

- GPU hidden-state export from large Hugging Face models.
- Qwen-Scope / Gemma Scope / SAE-Lens full forwards.
- Memory-token or side-cross-attention payload training at benchmark scale.
- Architecture-specific direct-KV injection and RoPE/head-layout validation.
- Paid closed-model reader API calls.
- Activation Beacon subprocess baseline execution.
- Full LongBench, RULER, LoCoMo, LongMemEval, and long-context benchmark downloads/runs.

## Ruff/static style

`ruff` was not installed in this execution environment, so Ruff was not run after the AAM-v2 update.  The code compiles and the full local test suite passes.
