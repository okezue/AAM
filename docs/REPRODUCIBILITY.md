# Reproducibility and result governance

## Required immutable artifacts

Every serious run preserves:

- fully resolved YAML and SHA-256;
- AAM source commit and dirty-state patch if applicable;
- external repository commit lock;
- base model and tokenizer commits;
- SAE repository/release/ID/hook/site and commit;
- dataset revision, local hashes, split, and processed/chunk manifest;
- software environment and container digest;
- hardware/topology/driver/attention kernel;
- all data, graph, edge-control, training, and generation seeds;
- raw per-example outputs and comparison reports;
- graph/store checkpoint pair for resumed streams.

## Determinism

The CPU core uses deterministic hashing and seeded Python/NumPy RNG. GPU runs should set PyTorch
seeds and deterministic algorithms where feasible, record nondeterministic kernels, and quantify
repeat variance. Never discard an unfavorable seed unless a pre-registered infrastructure failure
rule applies.

Graph-null and shuffle controls are deterministic functions of the learned graph, edge condition,
seed, and write count. Feature artifacts should be shared across these conditions.

## Stage caches

Cache immutable stages independently:

1. source segmentation;
2. raw activation capture;
3. SAE feature extraction;
4. sparse episode/query codes;
5. graph condition;
6. retrieval outputs;
7. payload reconstruction;
8. reader requests/responses.

Every cache key includes upstream text hash, preprocessing, tokenizer/model/SAE revisions, layer/site,
pooling, dtype, top-k, and code version. Matching dimensions do not establish compatibility; never
reuse activations across model revisions.

## External revisions

`third_party/baselines.yaml` lists upstream repositories. Run:

```bash
python scripts/resolveexternalrevisions.py
```

Archive `REVISION_LOCK.json`, then clone exact commits into isolated environments. For a baseline
with model-specific CLI code, use `JSONSubprocessBaseline` so request/response files and command are
captured without vendoring or modifying upstream behavior invisibly.

## Fresh versus resumed memory

- Independent runs clear both store and graph.
- A shared-stream run starts from an empty store unless `resume_store: true`.
- Resume requires `resume_graph_path`; store-only resume is rejected.
- Eviction/consolidation rebuilds the reference graph from retained episodes by default.

These rules prevent hidden state leakage across trials and stale graph associations after deletion.

## Result comparison

`ExperimentRunner` writes `per_example.jsonl`, `summary.json`, `resolved_config.yaml`, and
`environment.json`. `scripts/compareruns.py` joins on `example_id`, reports unpaired rows, and
refuses duplicate IDs or an empty intersection.

For repeated questions within a conversation/item, use `evaluation.clusterkey` so bootstrap
resampling respects dependence.

## Confirmatory versus exploratory work

Keep frozen manifests in separate directories, for example:

```text
configs/confirmatory/
configs/exploratory/
```

Materialize the confirmatory matrix, resolve all placeholder IDs/revisions, hash the generated YAMLs,
and publish/register them before test outcomes are inspected. Exploratory sweeps, failed runs, and
post-hoc exclusions must be labelled.

## Closed readers

Save complete prompts, exact model string, timestamp, provider response ID, usage, finish/refusal
metadata, retry history, and request hash. A provider alias update is a new condition. Never mix
outputs collected from materially different model versions under one condition without a planned
stability analysis.
