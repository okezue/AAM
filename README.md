# Activation-Associative Memory (AAM)

A research repository for testing whether **sparse model-internal feature patterns** can serve as
persistent, content-addressable memory addresses for language models.

AAM separates four questions that are often conflated:

1. Which representation is the best memory address: text, dense activations, projected activations,
   sparse autoencoder features, or learned bottlenecks?
2. Does online coactivation/temporal plasticity add value beyond nearest-neighbor retrieval?
3. Which payload should a retrieved address point to: immutable source text, a compressed activation,
   natural-language activation code, memory tokens, cross-attention values, or model-specific K/V?
4. Can replay and consolidation preserve useful memories under a fixed byte budget without amplifying
   unsupported traces?

The reference system stores source provenance by default. Latent states are treated as addresses and
optional payloads—not as magically lossless evidence.

## Implemented now

The CPU/reference path is executable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
aam smoke
pytest
```

It includes:

- deterministic sparse hashing and precomputed-feature encoders;
- Hugging Face hidden-state, TransformerLens hook, SAE-Lens, sentence-transformer, and exact
  Qwen-Scope adapters;
- online Hebb, sparse centered-covariance, Oja-like, and BCM-like writes;
- directed temporal associations and query-anchored recurrent pattern completion;
- learned, zero, random-degree-matched, and degree-preserving shuffled graph controls;
- RAM/SQLite stores, source checksums, replay, consolidation, episode and byte capacity limits;
- graph rebuild after eviction/consolidation so removed episodes do not leave ghost associations;
- logical byte accounting for episode codes, payloads, source text, postings, and graph state;
- synthetic, continual-retention, LongMemEval, LongMemEval-V2, LoCoMo, LongBench,
  LongBench-v2, and RULER loaders;
- reusable token/word/character chunking with shared segmentation across conditions;
- Hugging Face, OpenAI, Anthropic, and Gemini readers with source-labelled text injection;
- activation-codec training/export utilities and public Natural Language Autoencoder adapters;
- paired bootstrap/randomization statistics, cluster bootstrap, per-example provenance, configs,
  tests, Slurm templates, and an AWS Batch boundary.

Optional extras:

```bash
pip install -e .[hf]          # open-model activation extraction and generation
pip install -e .[sae]         # SAE-Lens / TransformerLens experiments
pip install -e .[embeddings]  # dense text controls
pip install -e .[closed]      # closed-reader hybrid experiments
pip install -e .[nla]         # Natural Language Autoencoder payload evaluation
pip install -e .[all]
```

## Architecture

```text
observation
  ├─ writer activation site ─ SAE/projection ─ sparse address z_e
  │                                           ├─ symmetric association graph A
  │                                           └─ directed temporal graph T
  ├─ immutable source pointer/text
  └─ optional activation payload c_e

query ─ same address encoder ─ q ─ recurrent completion over A,T
                                     └─ ranked source-backed episodes
                                          ├─ source-text reader injection
                                          ├─ NLA / activation-codec analysis
                                          └─ model-specific latent injection hook
```

The default recall update is

\[
a^{(r+1)}=\operatorname{TopK}\!\left[\phi\!\left(
\alpha q+\beta\widetilde A a^{(r)}+\gamma_fTa^{(r)}+
\gamma_bT^\top a^{(r)}-\theta
\right)\right].
\]

The persistent query term prevents drift toward generic attractors. Degree normalization, row caps,
hub penalties, edge controls, and bounded feature cardinality are separately ablated.

## Fastest serious experiment

The most direct open-model pilot uses the official Qwen-Scope SAE for Qwen3.5-2B:

```bash
# Resolve and record immutable model/SAE revisions before a confirmatory run.
PYTHONPATH=src python scripts/materializeablationmatrix.py \
  configs/ablations/minimal_confirmatory.yaml \
  --outputdir runs/materialized_configs --max-runs 10

# Then run each generated YAML on a GPU host.
aam run runs/materialized_configs/00002_aam_full.yaml
```

The five-condition confirmatory subset is:

1. dense text retrieval;
2. sparse Qwen-Scope activation kNN;
3. full AAM;
4. recurrence with a zero graph;
5. recurrence with a degree-preserving shuffled graph.

This isolates representation quality, learned association, and extra recurrent compute.

## Experiment levels

| Level | Main question | Status |
|---|---|---|
| L0 | Are graph, store, replay, accounting, and recurrence correct? | CPU executable |
| L1 | Do sparse internal addresses beat dense text and activation kNN? | Exact Qwen-Scope path plus SAE-Lens/model-hook alternatives |
| L2 | Can activation payloads replace source text at matched bytes? | Codec/export/NLA harness executable; reader surgery remains model-specific |
| L3 | Do verified replay and consolidation improve fixed-capacity retention? | Continual stream, replay, eviction, rebuild, and byte accounting executable |
| L4 | Does AAM improve quality per byte/FLOP on long context? | Dataset/chunking/baseline contracts wired; accelerator runs not executed here |
| L5 | Does retrieval transfer to a closed reader? | Open writer + closed text reader executable with credentials |

See [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md),
[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md), and
[`docs/MATHEMATICS.md`](docs/MATHEMATICS.md).
The checks actually executed in the build environment are recorded in
[`VALIDATION_REPORT.md`](VALIDATION_REPORT.md).

## Model families

- **Exact public Qwen-Scope:** Qwen3.5 2B/9B/27B model fragments are provided.
- **Gemma Scope / Gemma Scope 2:** SAE-Lens-compatible paths; revision-sensitive IDs must be
  resolved before execution.
- **Llama Scope:** SAE-Lens path for Llama 3.1 8B, with an explicit unresolved SAE ID field.
- **No-SAE controls:** pooled hidden states plus seeded projection/top-k, TransformerLens hooks,
  sentence embeddings, lexical hashing, and precomputed sparse artifacts.
- **Natural Language Autoencoders:** public Qwen2.5-7B, Gemma-3-12B/27B, and Llama-3.3-70B
  checkpoint metadata is represented as an optional payload condition.
- **Closed readers:** an open surrogate writes/retrieves; source text is injected into the closed
  reader. This is hybrid associative retrieval, not proprietary-model activation memory.

A public SAE alone is insufficient for a proprietary production model unless the exact model
revision, tokenizer, activation site, normalization convention, and runtime activation interface are
also available.

## Datasets

```bash
# Dependency-free synthetic inspection
aam prepare configs/experiments/l0_cpu_smoke.yaml --limit 3

# Public datasets require the Hugging Face/data extras.
python scripts/preparedatasets.py \
  synthetic continual_retention longmemeval locomo longbench longbenchv2 ruler --limit 3
```

Long-input configs use the `chunked` wrapper so all methods receive identical segments. The
repository does not redistribute benchmark data; revisions and file hashes must be locked for a
publication run.

## Reproducible runs

```bash
# Run one experiment
aam run configs/experiments/l3b_continual_retention.yaml

# Compose a model and dataset fragment
PYTHONPATH=src python scripts/composeexperiment.py \
  configs/experiments/l1_activation_retrieval.yaml \
  --model configs/models/qwen35_2b_qwen_scope.yaml \
  --dataset configs/datasets/synthetic.yaml \
  --name pilot --output runs/pilot.yaml

# Compare paired outputs; missing IDs are reported.
PYTHONPATH=src python scripts/compareruns.py \
  runs/baseline/per_example.jsonl runs/aam/per_example.jsonl \
  --metric retrieval_recall --output runs/comparison.json
```

Each run writes a resolved config, environment/config hash, per-example JSONL, summary statistics,
component byte accounting, and—when persistent—graph/store artifacts.

## External baselines and compute

`third_party/baselines.yaml` records upstream repositories. `JSONSubprocessBaseline` provides an
auditable JSON input/output boundary for pinned model-specific implementations such as Activation
Beacon. No external code is silently vendored.

- Slurm array template: `infrastructure/slurm/aam_array.sbatch`
- AWS Batch boundary: `infrastructure/aws_batch/`
- Compute protocol: `docs/COMPUTE_PLAN.md`

## Repository map

```text
src/aamemory/encoding/       sparse address encoders
src/aamemory/memory/         graph, store, retrieval, replay, capacity, accounting
src/aamemory/models/         open/closed readers and injection contracts
src/aamemory/data/           public benchmark and diagnostic loaders
src/aamemory/eval/           metrics, statistics, experiment runner
src/aamemory/training/       activation payload codecs
src/aamemory/integrations/   external-baseline and benchmark boundaries
configs/                     models, datasets, experiments, ablations
scripts/                     preparation, export, training, comparison, revision locking
docs/                        full specification and protocol
infrastructure/              scheduler templates without credentials
```

## Non-claims

- Residual coordinates are not assumed to be literal biological neurons.
- “SAE feature” does not imply perfectly monosemantic or causally privileged.
- A low-bit engram cannot losslessly recover arbitrary high-entropy data without a retained channel.
- Recurrence earns its complexity only if it beats kNN and graph-null/shuffle controls at matched
  bytes, candidates, and compute.
- Same-model activation capture/injection is not available through ordinary closed-model APIs.
- The novelty review is a structured research assessment, not a patent opinion.


## AAM-v2: Hippocampal Activation Memory update

This repo now includes a complete AAM-v2 implementation path for **Memory as Activations**.  The AAM-v2 condition keeps raw text as provenance/exact-recall fallback only; ranking and recurrence use sparse activation engrams, sensory/context bindings, episode-index connectivity, Hebbian/temporal graphs, dopamine-gated writes, verified replay, reconsolidation, future-query prospection, and sidecar neurogenesis.

Fast validation:

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m aamemory.cli run configs/experiments/aam_v2_cpu_smoke.yaml --limit 6
PYTHONPATH=src python scripts/checknotextprimary.py runs/aam_v2_cpu_smoke
```

Checkpoint suite:

```bash
PYTHONPATH=src python scripts/runcheckpointsuite.py --dry-run
PYTHONPATH=src python scripts/runcheckpointsuite.py --limit 10
```

Key documents:

- `docs/AAM_V2_SPECIFICATION.md`: complete biological-to-computational design.
- `docs/AAM_V2_MATHEMATICS.md`: equations for sparse addresses, dopamine gates, LTP/LTD/STDP, completion, scoring, replay, and neurogenesis.
- `docs/AAM_V2_EXPERIMENTS.md`: checkpoints, standalone ablations, datasets, and metrics.
- `docs/AAM_V2_IMPLEMENTATION.md`: open-model/closed-model feasibility and GPU placeholders.
- `docs/AAM_V2_FILE_MAP.md`: exact files added/changed.

Representative configs added under `configs/experiments/` include C0 through C13: sparse engrams, context binding, episode-index identity, Hebbian graph, temporal/STDP graph, recurrent completion, dopamine-gated writes, latent payload codec, memory injection, verified replay, reconsolidation, neurogenesis, future simulation, and pseudo-multimodal state tensors.  Ablation matrices live under `configs/ablations/`.
