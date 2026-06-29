# Result files and accounting

## `resolved_config.yaml`

The fully expanded experiment configuration after environment-variable substitution. Its SHA-256 is
stored in `environment.json` and `summary.json`.

## `environment.json`

Records Python version/executable, platform, source git commit when available,
`CUDA_VISIBLE_DEVICES`, and config hash. Accelerator harnesses should extend it with GPU, driver,
CUDA, PyTorch, attention kernel, model/SAE revisions, dataset hashes, and external revision lock.

## `per_example.jsonl`

Each row includes:

- example/task/query/answers, positive and negative evidence IDs;
- retrieved IDs and total/exact/associative/temporal scores;
- recall, precision, MRR, NDCG, negative-hit rate, and answer-in-source indicators;
- write/read/generation latency;
- recall depth, active features, and message-passing edge visits;
- cumulative association and temporal update counts;
- memory/store/graph diagnostics;
- exact logical memory footprint by component;
- prediction, answer metrics, reader usage, provider metadata, and injection trace when enabled;
- source dataset metadata.

Logical footprint fields include:

- `memory_bytes_total`;
- `memory_episode_bytes`;
- `memory_graph_bytes`;
- `memory_posting_index_bytes`;
- nested source, source-metadata, episode-metadata, sparse-index, sparse-value, payload, graph-edge,
  and graph-node-state components.

SQLite store diagnostics also expose physical database/WAL/SHM size when available. Logical bytes
are the primary cross-method comparison because allocator/page overhead varies by backend; physical
bytes remain an operational metric.

## `summary.json`

Contains:

- run name/time;
- resolved config;
- environment/config hash;
- final shared-memory stats for stream experiments;
- mean and bootstrap interval for each numeric field;
- per-task means;
- bootstrap unit (`example` or configured cluster key).

## Persistent-stream artifacts

When `resetbetweenexamples: false`, the runner writes `graph.json` and keeps the configured store.
A resume run requires both the store and explicit `evaluation.resume_graph_path`; this prevents a
persistent episode store from being paired with a fresh graph.

## Paired comparison report

`scripts/compareruns.py` joins by `example_id`, reports baseline-only/treatment-only IDs, computes a
paired bootstrap difference and paired randomization p-value, and can write a JSON report. It errors
when no paired rows exist or IDs are duplicated.

## Publication extensions

Model-specific harnesses should add:

- official benchmark scorer outputs;
- synchronized GPU timing and warmup policy;
- peak allocated/reserved memory and KV bytes;
- writer/SAE/payload/reader FLOPs;
- energy/power trace when stable;
- full request hash/cache key for closed APIs;
- exact model/SAE/tokenizer/dataset/external commits;
- failure/refusal/retry categories;
- Holm-adjusted secondary p-values and pre-registered non-inferiority decisions.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
