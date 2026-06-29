# Experimental design

## 1. Scientific questions

**RQ1 — Representation.** At equal total memory bytes, candidate count, and writer forwards, do
sparse model-internal addresses retrieve supporting episodes better than lexical or dense text
representations?

**RQ2 — Association.** Given the same sparse addresses, does learned coactivation/temporal structure
plus recurrent completion outperform activation kNN and graph-null/shuffle controls?

**RQ3 — Payload.** Holding retrieved episode IDs fixed, can a compressed activation payload preserve
reader behavior more efficiently than source text?

**RQ4 — Continual retention.** Under an episode or byte cap, do salience, replay, consolidation, and
decay improve retention without strengthening false or superseded memories?

**RQ5 — Long context.** Does recent raw context + AAM + an immutable archive improve quality per
byte/FLOP/latency over full attention, compression, and ordinary RAG?

**RQ6 — Reader transfer.** Does an AAM built in an open writer space improve a different open or
closed reader when only source text is transferred?

## 2. Confirmatory claims

Pre-register the final materialized YAML files—not only the matrix template—before looking at test
outcomes. The supplied `configs/ablations/minimal_confirmatory.yaml` materializes five runnable
conditions around the official Qwen3.5-2B Qwen-Scope SAE.

### H1: representation

Sparse Qwen-Scope activation kNN will exceed dense text retrieval on paraphrase and two/three-hop
evidence recall@10 at matched logical bytes and top-k. Report task-stratified effects; do not average
away a random-string failure.

### H2: learned association

Full AAM will exceed activation kNN on two/three-hop evidence recall@10, with negative-evidence hit
rate no more than one absolute percentage point worse.

### H3: structural specificity

Full AAM will exceed both:

- recurrence with a zero graph; and
- recurrence with a degree-preserving shuffled graph.

This rules out extra iterations, feature expansion, and graph degree distribution as sufficient
explanations.

### H4: replay safety

Source-verified replay will exceed unverified replay after injected false/generated traces, measured
by correct-source retention and unsupported-answer/edge-mass growth.

### H5: payload non-inferiority

At least one latent payload condition will be non-inferior to retrieved source text on answer quality
under a pre-registered margin while reducing reader input bytes. Select the margin from a separate
pilot and archive the rationale.

### Gatekeeping order

1. Test H1.
2. Test H2/H3 only if the activation address is competitive with the strongest text control.
3. Train payloads only after retrieval passes.
4. Run expensive long-context/closed-reader generation only after source retrieval passes.

This prevents expensive downstream experiments from masking a failed memory substrate.

## 3. Experimental levels

### L0 — Mechanism and software validation

Purpose: establish implementation invariants, not scientific effectiveness.

Conditions:

- hand-constructed sparse patterns;
- deterministic hashing;
- zero, learned, shuffled, and degree-matched random graphs;
- RAM and SQLite stores;
- graph save/load and store reset/resume;
- replay checksum accept/reject;
- episode- and byte-capacity maintenance;
- deletion/consolidation followed by graph rebuild;
- independent and shared-stream run semantics.

Pass criteria:

- all tests pass;
- repeated seeded runs are bit-identical where expected;
- no NaN/Inf;
- row/feature bounds hold;
- deleted features no longer contribute after rebuild;
- byte budget is met or the report states the irreducible empty-graph/node overhead;
- fresh runs cannot inherit an old SQLite store unless resume is explicit.

### L1A — Representation pilot

Writer candidates:

1. Qwen3.5-2B-Base + official Qwen-Scope SAE.
2. Same Qwen hidden state, dense.
3. Same hidden state + seeded projection + top-k.
4. Same hidden state + learned top-k bottleneck.
5. Sentence-transformer dense control.
6. Lexical hashing/BM25-style control.

Replications after the Qwen pilot:

- Qwen3.5-9B + Qwen-Scope;
- Gemma 2 2B + Gemma Scope;
- Gemma 3 4B + Gemma Scope 2;
- Llama 3.1 8B + Llama Scope.

Address-only tasks:

- arbitrary entity–attribute pairs;
- lexical and semantic paraphrase;
- alias/entity disambiguation;
- one-, two-, and three-hop chains;
- temporal update and supersession;
- near-neighbor interference;
- frequent generic hubs;
- random identifiers, dates, numbers, and strings;
- contradictory and source-poisoned memories.

Primary outcome: source recall@10. Secondary outcomes: recall@1/5, MRR, NDCG, negative-hit rate,
hub-capture rate, path/hop recall, bytes, writer time, query encoding time, and code sparsity.

Layer/site pilot:

- depth fractions 0.25, 0.50, 0.75, final;
- residual-pre/post, attention output, MLP output where available;
- select on development source retrieval only;
- freeze one selected site plus one pre-registered robustness site.

### L1B — Association and recurrence

Freeze the episode/query sparse codes from L1A. This makes graph ablations cheap and prevents model
forward nondeterminism from confounding them.

Core conditions:

1. recurrence depth 0: activation kNN;
2. depth 2 with zero graph;
3. learned coactivation graph;
4. degree-preserving shuffled learned graph;
5. random-degree-matched graph;
6. learned graph with query anchor removed;
7. learned graph without degree normalization;
8. learned graph without hub penalty;
9. association-only, temporal-only, and both;
10. Hebb, active-pair covariance, Oja-like, and BCM-like rules.

Compute matching:

- identical candidate limit and retrieved top-k;
- report message-passing edge visits;
- either cap recurrence edge visits or plot quality against them;
- do not let the recurrent condition retrieve more episodes than kNN;
- use the exact same codes and write order.

Mechanism diagnostics:

- recall by path length;
- state overlap with query by iteration;
- attractor drift and cycles;
- fraction of final features introduced by the graph;
- high-degree feature share;
- contribution of exact/association/temporal score terms;
- causal edge ablation for retrieved paths on a diagnostic subset.

### L2 — Payload and reader integration

Teacher: same open reader with full source context when feasible. Student: frozen reader with recent
context plus selected memory. Retrieval IDs are cached first and reused across payload conditions.

Payload ladder, in order:

1. source text/pointer;
2. symmetric int8 pooled hidden state;
3. learned MLP bottleneck/reconstruction;
4. Natural Language Autoencoder verbalization/reconstruction;
5. decoded memory tokens;
6. gated side cross-attention;
7. direct pre-position K/V for one supported architecture;
8. Activation Beacon K/V traces selected by AAM.

Trainable parameters initially include only codec, decoder/injection module, and gates. Freeze the
backbone. Only after a successful frozen-backbone experiment should slow LoRA consolidation be
explored.

Objectives:

- task cross-entropy;
- KL/logit agreement with full-context teacher;
- activation reconstruction;
- supporting episode/source ID prediction;
- irrelevant-memory gate loss;
- optional contrastive pattern separation.

Evaluation:

- answer and official benchmark metric;
- teacher KL/logit agreement;
- source attribution/citation precision and recall;
- random-string exactness;
- reader input bytes/tokens;
- payload and KV bytes;
- payload decode, prefill, decode latency;
- peak GPU memory and FLOPs.

A latent payload is not credited for a plausible answer unless the correct source episode was
selected. Separate retrieval error from payload reconstruction/generation error.

### L3 — Continual retention, replay, and consolidation

Stream length: at least 10× fixed capacity in the confirmatory run. Probe immediately and at
logarithmic lags such as 1, 5, 20, 50, 100, 250, 500, and 1,000 intervening writes.

Capacity conditions:

- unbounded diagnostic reference;
- fixed episode count;
- fixed total logical bytes;
- FIFO, LRU, salience/utility eviction;
- prototype consolidation;
- consolidation plus verified replay.

Replay conditions:

- none;
- random;
- recency;
- salience;
- associative centrality;
- associative + source checksum verification;
- verified replay + contradiction-driven LTD;
- verified replay + prototype consolidation;
- slow adapter consolidation, exploratory.

Controlled hazards:

- generated false trace with no immutable source;
- corrupted source checksum;
- conflicting later update;
- repeated high-salience distractor;
- cross-namespace near duplicate;
- stored prompt injection.

Primary outcomes:

- retention area under curve;
- latest-record accuracy;
- correct-source recall by lag;
- unsupported trace edge mass by replay cycle;
- replay gain per graph update/edge visit;
- bytes and capacity saturation;
- consolidation merge precision.

The reference continual config uses a shared stream (`resetbetweenexamples: false`) and cluster
bootstrap by memory item. Replacing hashing with a locked activation artifact converts it into the
confirmatory L3 run without changing stream logic.

### L4 — Long-context systems comparison

Datasets:

- RULER across multiple lengths/categories;
- LongBench and LongBench-v2;
- LongMemEval and LongMemEval-V2;
- LoCoMo;
- diagnostic Needle-in-a-Haystack/Multi-Needle generation when using an official pinned generator.

All methods must use identical precomputed segmentation. Query-independent memory writes occur
before the query; query-dependent compression/retrieval is labelled separately.

Baselines:

- full/native context where feasible;
- recent/sliding context;
- dense text RAG;
- sparse activation kNN;
- query-independent summaries;
- Activation Beacon;
- Neurocache;
- MemoryLLM/M+;
- Larimar;
- CAMELoT-like consolidated cache when reproducible;
- MRAgent/EcphoryRAG/LatentGraphMem/SAGE when code and compatible protocols are available;
- AAM without recurrence;
- full AAM.

Report a Pareto surface, not one score:

- answer quality/evidence recall;
- total logical and physical memory bytes;
- writer forward and amortization count;
- graph/read compute and reader context tokens;
- prefill/decode latency;
- peak host/GPU memory;
- energy where measured consistently.

For Activation Beacon, count all retained beacon K/V states. For AAM, count source archive bytes
separately and both include/exclude the archive only when the scientific question is explicitly
“address-store overhead” versus “complete deployed memory.”

### L5 — Different/closed reader transfer

Writer/retriever: fixed open SAE model or sentence model. Reader: a different open model or a closed
OpenAI/Anthropic/Gemini API model. Only source text crosses the reader boundary.

Conditions with the same reader and prompt:

1. no memory;
2. dense text RAG;
3. sparse activation kNN;
4. full AAM;
5. oracle evidence.

Constraints:

- fixed retrieved top-k and injected character/token ceiling;
- identical source labels and ordering template;
- temperature zero when supported;
- exact model string and provider metadata logged;
- response caching by complete request hash where terms permit;
- a dated stability subset because provider aliases may change.

This tests retriever transfer. It is not evidence that the proprietary model stores or consumes
activation engrams.

## 4. Dataset split and preprocessing governance

- Use development data only for site, hyperparameter, or threshold selection.
- Synthetic train/development/test generators use disjoint master seeds and vocabularies.
- Preserve official benchmark test labels and scorer versions.
- Freeze tokenizer-based chunks and reuse them across all methods.
- Record excluded examples and reasons.
- Namespace episode/evidence IDs by question/conversation to prevent collisions.
- For conversation datasets, bootstrap by conversation/trajectory, not only by question.
- For API-judged tasks, freeze judge prompt/model/version and blind condition labels.
- Never treat a benchmark answer appearing anywhere in a long source as proof of correct retrieval;
  source IDs are the primary evidence signal.

## 5. Budget matching

Every primary comparison records:

- source UTF-8 bytes;
- code index/value bytes;
- payload bytes;
- episode/source metadata bytes;
- posting-index bytes;
- graph edge/node-state bytes;
- physical database/WAL bytes;
- writer forwards and elapsed time;
- graph association/temporal updates;
- recall message edge visits;
- candidates scored and items returned;
- reader input tokens/characters;
- reader KV/peak memory, latency, and generation tokens.

Primary plots use total logical memory bytes. Supplementary plots may isolate address overhead,
payload overhead, or archive overhead, but exclusions must be explicit.

## 6. Metrics

### Retrieval

- source recall@1/5/10;
- precision@k, MRR, NDCG;
- path/hop recall;
- negative-evidence hit rate;
- hub capture and maximum-degree-source share;
- latest/superseded record accuracy;
- score calibration;
- source checksum validity.

### Generation

- exact match, token F1, and official task metric;
- source-supported answer rate;
- citation/source precision and recall;
- random-string exactness;
- contradiction and temporal consistency;
- full-context teacher KL/logit agreement when available.

### Continual learning

- retention curve and area under curve;
- forgetting by lag, salience, frequency, and interference class;
- forward/backward transfer;
- saturation point;
- replay benefit per update/FLOP;
- false-memory amplification slope;
- consolidation merge precision/recall.

### Systems

- writer, feature extraction, graph write, recurrence, candidate search, payload decode, prefill, and
  generation latency distributions;
- logical/physical bytes by component;
- graph nodes/edges and posting entries;
- accelerator peak memory and KV cache;
- FLOPs and energy where measured reproducibly.

## 7. Statistical plan

- Use paired examples across conditions.
- Use at least five independent seeds for learned components; deterministic graph controls may share
  a feature cache but use independent declared edge seeds.
- Report mean, absolute paired effect, relative effect, and 95% paired bootstrap CI.
- Use a paired randomization test for the primary comparison.
- Use cluster bootstrap for LoCoMo/conversation and repeated-lag continual probes.
- Correct secondary comparison families with Holm–Bonferroni.
- For non-inferiority, evaluate the lower confidence bound against the pre-registered margin.
- Archive raw per-example outputs and explicitly report unpaired/missing IDs.
- Do not infer significance from overlap of marginal confidence intervals.

A pilot estimates variance and throughput. Fix confirmatory sample size with a paired power analysis
before final labels are inspected. Stop neither early for a favorable effect nor after a null result
unless a pre-registered sequential design is used.

## 8. Ablation program

### Confirmatory minimum

- dense text RAG;
- activation kNN;
- full AAM;
- zero graph;
- degree-preserving shuffled graph.

### Representation

- raw/dense residual versus random projection versus learned bottleneck versus SAE;
- model family and scale;
- layer/site;
- single versus multi-layer;
- mean/max/positive-max/last/boundary pooling;
- feature width and episode top-k;
- signed versus positive-only codes;
- token, sentence, paragraph, message, and fixed-token episodes.

### Plasticity/recall

- none/Hebb/covariance/Oja/BCM;
- centered activity rate;
- decay and row cap;
- association/temporal direction;
- recurrence depth and threshold;
- query anchor;
- degree normalization and hub penalty;
- exact/association/temporal rank weights;
- candidate limit and feature expansion limit.

### Payload/reader

- source pointer/text versus latent-only;
- int8 activation, MLP codec, NLA, memory tokens, cross-attention, K/V, Activation Beacon;
- injection layer and gate;
- source ID loss and irrelevant-memory gate;
- frozen backbone versus slow adapter.

### Continual memory

- replay seed policy, verification, learning scale, and interval;
- episode versus byte capacity;
- eviction policy;
- consolidation threshold/group size;
- rebuild versus contribution-log unlearning;
- trusted versus generated source class.

Do not run the full Cartesian product. Use the confirmatory subset first, then fractional factorial,
successive halving, or Bayesian exploration on development data. Preserve a final untouched
replication grid.

## 9. Falsification and stopping criteria

The tested associative claim fails if learned recurrence does not beat activation kNN and both graph
controls at matched retrieval count/compute, or if any gain requires an unacceptable false-association
increase.

The activation-substrate claim fails if internal sparse addresses do not beat the strongest text or
same-model dense/projected control at matched bytes.

The payload claim fails if source text remains strictly better on the quality/byte Pareto frontier.

Stop or quarantine a run for:

- source-free replay potentiation when verification is required;
- graph edge/cardinality explosion;
- NaN/Inf;
- generic-hub collapse;
- short-context capability degradation above the registered threshold;
- namespace/privacy leakage;
- inconsistent store/graph resume state;
- inability to reproduce a seed within declared numerical tolerance;
- model/SAE dimension or revision mismatch.

Negative results are informative: the system is designed to identify exactly where activation
addresses, recurrence, or replay fail to earn their complexity.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
