# Implementation status

Reviewed: 2026-06-21.

## Executable reference implementation

### Address encoders

- deterministic lexical feature hashing;
- sentence-transformer embedding followed by seeded sparse projection;
- generic Hugging Face hidden-state capture and projection;
- TransformerLens arbitrary hook-point capture;
- SAE-Lens feature extraction;
- exact official Qwen-Scope TopK checkpoint loader for Qwen3.5 2B/9B/27B;
- precomputed sparse-feature artifacts for one-time GPU extraction and cheap graph sweeps.

### Memory algorithm

- source-backed episode schema and checksums;
- sparse Hebb, active-pair centered covariance, Oja-like, and BCM-like writes;
- directed temporal links;
- bounded rows, decay, normalization, and hub penalty;
- query-anchored recurrent completion;
- learned, zero, degree-preserving shuffled, and random-degree-matched edge conditions;
- inverted candidate index and separately logged exact/associative/temporal scores;
- RAM and SQLite stores;
- int8 array payloads and pooled activation payload plumbing;
- source-gated replay;
- prototype consolidation, episode eviction, and logical byte-budget eviction;
- graph reconstruction after deletion/consolidation to remove ghost associations;
- component accounting for source, metadata, sparse code, payload, postings, node state, and edges.

### Evaluation and data

- synthetic paired-associate, paraphrase, multi-hop, temporal, interference, hub, random-string, and
  poison diagnostics;
- continual retention stream with configurable probe lags;
- LongMemEval, schema-tolerant LongMemEval-V2 exports/backend, LoCoMo, LongBench,
  LongBench-v2, and RULER loaders;
- reusable character/word/token chunking with overlap and source-offset provenance;
- retrieval and answer metrics;
- paired bootstrap differences, paired randomization tests, ordinary and cluster bootstrap intervals;
- per-example JSONL, resolved config, environment/config hash, memory/compute proxies, and summaries;
- missing-ID-aware paired comparison utility.

### Readers, payloads, and integration

- local Hugging Face generation;
- OpenAI Responses, Anthropic Messages, and Gemini GenerateContent readers;
- source-labelled text injection with a hard character budget and injection trace;
- public NLA checkpoint metadata, download, SGLang launch, and reconstruction-scoring scripts;
- MLP activation-codec training and hidden-activation export;
- JSON subprocess contract for pinned external baselines;
- Activation Beacon payload interface;
- LongMemEval-V2 backend-shaped adapter;
- Slurm and AWS Batch launch boundaries without credentials.

## Configured but not executed in this environment

- downloading and forwarding multi-billion-parameter writer/reader models;
- Qwen-Scope, Gemma Scope, Llama Scope, or NLA checkpoint evaluation on accelerators;
- official benchmark-scale runs;
- paid closed-provider API calls;
- upstream Activation Beacon, Neurocache, MemoryLLM, Larimar, and other external baselines;
- distributed/multi-node timing and power measurement.

These require the user's compute, model acceptance tokens where applicable, provider credentials, and
immutable revision choices. The repository contains interfaces and run manifests, but no result is
claimed for an experiment that was not run.

## Deliberate model-specific placeholders

The following cannot be made honestly architecture-neutral:

- learned memory-token decoder integrated into a particular reader;
- gated side cross-attention surgery;
- direct pre-RoPE K/V reconstruction/injection;
- Activation Beacon K/V capture from a selected upstream checkpoint/revision;
- contradiction-trained LTD and full-context teacher distillation;
- slow LoRA/adapter consolidation;
- production distributed graph/index backend;
- account-specific IAM, networking, object storage, AMIs, quotas, and credentials.

The generic latent-injection adapter raises an explicit error rather than pretending that hidden-state
or K/V formats are portable across architectures.

## Important distinctions

- The **text/source retrieval path** is end-to-end executable.
- Pooled/int8/MLP/NLA payload extraction or reconstruction can be evaluated, but the generic reader
  does not yet consume those payloads as internal memory tokens.
- Qwen-Scope IDs in the supplied fragments are concrete public repositories; model and SAE commit
  hashes still must be pinned at run time.
- Gemma Scope 2 and Llama Scope IDs marked `PLACEHOLDER...` require registry resolution because
  release naming changes independently of this repository.
- A passing CPU test suite validates implementation invariants, not the scientific hypothesis.
