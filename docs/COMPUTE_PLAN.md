# Compute plan

No account-specific cloud resources are created by this repository. Model downloads, access-token
acceptance, API credentials, IAM, networking, quotas, regions, and cost policy remain host-side.
Scheduler templates are in `infrastructure/`.

## Tier 0 — CPU validation

Run:

```bash
pip install -e .[dev]
aam smoke
pytest
```

Covers graph/store/replay/capacity/accounting, synthetic and continual data, config composition, and
external JSON integration. This validates software invariants only.

## Tier 1 — representation pilot

Recommended first GPU condition:

- Qwen3.5-2B-Base;
- official 32K-width Qwen-Scope TopK SAE;
- one mid and one late layer;
- 1,000–10,000 diagnostic examples;
- retrieval only.

Extract and cache sparse episode/query codes once. Reuse them across kNN, learned, zero, shuffled,
random-degree-matched, rule, and recurrence conditions.

Approximate resource class: one accelerator that fits the writer and one SAE layer. The SAE encoder
matrix itself can be held on CPU or a second device if needed; benchmark this choice and log transfer
time.

## Tier 2 — cross-family replication

Replicate the winning address/graph condition on:

- Qwen3.5-9B;
- Gemma 2 2B or Gemma 3 4B;
- Llama 3.1 8B.

Use five graph/training seeds. Shard activation/feature artifacts with checksums. Do not rerun model
forwards for graph-only ablations.

## Tier 3 — payload training

- frozen 1.5B–8B reader initially;
- full-context teacher;
- train pooled codec/reconstruction first;
- then memory tokens or gated cross-attention;
- NLA as a fixed public payload condition;
- direct K/V only for one architecture after shape/position tests.

Use mixed precision and gradient accumulation, but evaluate reconstruction in a declared precision.
Log writer, codec, payload transfer, prefill, and decode separately.

## Tier 4 — continual and long context

- streams at least 10× capacity;
- RULER/LongBench/LongMemEval/LoCoMo;
- 32K–1M full/compression baselines where feasible;
- persistent SQLite/graph checkpoints;
- exact logical and physical byte accounting;
- synchronized accelerator timing after warmup.

Long-context baseline jobs may need multiple GPUs or tensor parallelism. Record topology, model
sharding, quantization, driver, CUDA/PyTorch, attention kernel, power mode, and cache implementation.

## Tier 5 — closed-reader subset

Run source retrieval locally first, then issue closed-reader requests only for the frozen confirmatory
subset. Use separate queues/secrets and durable request caching where terms permit. Oracle and
no-memory conditions bound reader capability.

## Launch templates

### Local/materialized matrix

```bash
PYTHONPATH=src python scripts/materializeablationmatrix.py \
  configs/ablations/minimal_confirmatory.yaml \
  --outputdir runs/materialized_configs --max-runs 100
```

### Slurm

`infrastructure/slurm/aam_array.sbatch` reads one YAML path per array index. Edit account, partition,
module/container, walltime, and GPU count for the target cluster.

### AWS Batch

`infrastructure/aws_batch/job_definition.template.json` and `submit_array.sh` define the job command,
resources, S3 config/output boundary, and required environment variables. They intentionally omit
IAM roles, VPC/subnets, AMI/compute environment, ECR repository, buckets, secrets, and prices.

## Artifact layout

```text
runs/<experiment>/<seed>/
  resolved_config.yaml
  environment.json
  revision_lock.json
  model_and_sae_manifest.json
  dataset_manifest.json
  per_example.jsonl
  summary.json
  comparison.json
  graph.json
  memory.sqlite3
  logs/
```

## Efficiency accounting

Measure:

- tokenization/segmentation;
- writer forward and SAE/projection time;
- graph write and maintenance/rebuild time;
- recurrence edge visits and latency;
- candidate search/rank time;
- payload decode/transfer;
- reader prefill/decode;
- peak host/GPU memory and KV cache;
- logical bytes by source/code/payload/metadata/postings/graph;
- physical database/object-store bytes;
- FLOPs and energy where measured consistently.

Amortized claims must state the number of queries sharing the query-independent write cost.
