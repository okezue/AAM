# Dataset wiring and evaluation roles

The generic runner consumes `BenchmarkExample` objects containing source-backed `MemoryEvent`s,
query, answers, positive evidence IDs, negative evidence IDs, task, and metadata. Public loaders are
schema checked and do not silently invent evidence IDs when the source format is unknown.

## Synthetic mechanism diagnostics

`SyntheticMemoryDataset` generates:

- paired associates;
- paraphrased cues;
- multi-hop chains;
- temporal updates/supersession;
- near-neighbor interference;
- common-hub distractors;
- exact random strings;
- replay-poison cases.

Each row carries explicit evidence and negative-evidence IDs. Use it to diagnose the algorithm, not
to claim broad language competence.

Config: `configs/datasets/synthetic.yaml`.

## Continual retention

`ContinualRetentionDataset` creates an accumulating stream and emits probes at configurable lags.
Each target memory receives distractors and a high-entropy payload so retention can be plotted against
intervening writes. It is designed for `resetbetweenexamples: false`; IDs and cluster metadata
support item-level cluster bootstrap.

Config: `configs/datasets/continual_retention.yaml` and
`configs/experiments/l3b_continual_retention.yaml`.

## Shared chunking wrapper

`ChunkedDataset` wraps any configured base dataset and partitions every source episode by:

- characters;
- whitespace-delimited words; or
- tokens from a pinned Hugging Face tokenizer.

It preserves parent episode ID, chunk index/count, source URI/document ID, and offset metadata, and
remaps evidence IDs to all child chunks. Overlap is configurable. Use the same materialized chunks
for every long-context condition so a method is not advantaged by different segmentation.

L2 and L4 configs use token chunking with the same Qwen tokenizer as their writer.

## LongMemEval

Default source: `xiaowu0162/LongMemEval`.

Supported variants/files:

- `longmemeval_oracle.json`;
- `longmemeval_s_cleaned.json`;
- `longmemeval_m_cleaned.json`.

The loader maps each session to one episode, namespaces episode IDs by question, and preserves
`answer_session_ids` as evidence. It records question type/date and haystack session metadata.

```bash
aam prepare configs/experiments/l3_online_replay.yaml --limit 3
```

## LongMemEval-V2

Because the active upstream interface/export schema may evolve, AAM provides two explicit paths:

1. `LongMemEvalV2Dataset`, a schema-tolerant offline JSON/JSONL export loader requiring a path;
2. `AAMLongMemEvalV2Backend`, exposing insert/query/reset for a thin upstream adapter.

Question/trajectory IDs are namespaced. Unknown evidence schemas raise or remain empty rather than
being guessed. Pin the upstream commit and archive the export schema used by a run.

## LoCoMo

The loader obtains `data/locomo10.json` from `snap-research/locomo`, creates one event per dialogue
turn, sorts sessions robustly, and maps annotated evidence dialog IDs to event IDs. Report
conversation-cluster bootstrap intervals because the number of conversations is much smaller than
the number of QA rows.

Config: `configs/datasets/locomo.yaml`.

## LongBench

Default dataset: `THUDM/LongBench`. The loader supports selected official subsets and maps one
context to one episode in its base form. For long-context comparisons, wrap it with `chunked` and
freeze the resulting tokenizer-based segments.

Config: `configs/datasets/longbench.yaml`; L2 provides a chunked example.

## LongBench-v2

Default dataset: `THUDM/LongBench-v2`. The loader renders available multiple-choice options without
shifting labels and preserves category/length metadata. Use the official evaluation/judge protocol
for publication scores; the generic runner supplies evidence retrieval and simple answer metrics.

Config: `configs/datasets/longbench_v2.yaml`.

## RULER

Default pre-generated source: `self-long/RULER-llama3-1M`, with context-length configurations.
RULER covers retrieval, multi-hop tracing, aggregation, and QA-style synthetic tasks. For formal
model-specific comparisons, prefer the official NVIDIA/RULER generator pinned to a commit, because
construction can depend on tokenizer and target context length.

Config: `configs/datasets/ruler.yaml`; L4 wraps it in fixed token chunks.

## Needle and multi-needle protocols

The repository's synthetic random-string and temporal tasks cover mechanism-level needles. For
publication-quality NIAH/Multi-NIAH, clone and pin the official chosen generator through
`third_party/baselines.yaml`, generate source/evidence IDs into the AAM schema, and archive the exact
needle templates, insertion depths, lengths, tokenizer, and judge. Do not rely only on an LLM judge
score when exact source IDs/strings are available.

## Preparation

```bash
pip install -e .[hf]
python scripts/preparedatasets.py \
  synthetic continual longmemeval locomo longbench longbenchv2 ruler --limit 3
python scripts/resolveexternalrevisions.py
```

The preparation script performs a small load/schema pass. A full publication snapshot should also
materialize the processed examples or chunk index and record hashes.

## Required dataset manifest

Archive for every run:

- source repository/dataset ID and immutable revision;
- local file SHA-256 and size;
- loader source commit;
- official split/config name;
- preprocessing/chunking config and tokenizer revision;
- examples excluded and reason;
- query/evidence ID namespace policy;
- synthetic seed and generator version;
- license/terms review;
- official scorer/judge version.

The repository wires loaders to public sources but does not redistribute data or assert that every
possible downstream use is permitted by the source license or terms.
