# Prior art and novelty boundary

Last structured review: 2026-06-21. This is a scientific novelty assessment, not a patentability or
freedom-to-operate opinion.

## Bottom line

No exact implementation of the complete AAM combination was identified in the structured search,
but the novelty boundary is narrow. By 2026, several systems independently cover associative graph
retrieval, spreading activation, latent graph memory, active reconstruction, self-evolving memory,
provenance tiers, Hebbian matrices, activation caches, replay, or model-native compression.

A defensible contribution is therefore not “biological memory for LLMs” in general. It is the
specific integration and controlled empirical test of:

> Persistent query-independent episode addresses formed from sparse internal model features;
> salience-gated local coactivation and temporal writes; query-anchored feature-space completion;
> source pointers plus optional activation payloads; and source-verified replay under an explicit
> fixed byte budget.

## Closest 2026 work

### MRAgent — active reconstruction over an associative graph

*Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents* represents memory as a
Cue–Tag–Content graph and lets an LLM iteratively explore and prune retrieval paths. It is extremely
close to the “memory should be reconstructed through associations” motivation.

AAM's remaining distinction is substrate and update rule: sparse model-internal feature nodes,
online local coactivation/temporal plasticity, a cheap sidecar recurrence rather than LLM-guided graph
reasoning, optional model-native payloads, and byte-accounted verified replay.

- https://openreview.net/forum?id=YPoHy6lgKP
- https://arxiv.org/abs/2606.06036

### EcphoryRAG — fast-write, deep-read associative retrieval

EcphoryRAG writes lightweight entity-centric engrams and performs centroid-based spreading
activation through implicit associations. This substantially narrows any claim around ecphory,
spreading activation, fast-write/deep-read memory, or multi-hop associative RAG.

AAM must show that internal sparse feature addresses and local learned coactivation add value beyond
entity-centric/vector engrams and spreading activation.

- https://openreview.net/forum?id=YHSoIbQWR8

### LatentGraphMem — latent graph storage with explicit evidence readout

*Implicit Graph, Explicit Retrieval* learns a graph-structured latent memory, retrieves a compact
explicit subgraph under a fixed budget, and feeds it to a frozen reasoner. This narrows claims around
latent graph memory, budgeted explicit evidence, and transfer to larger frozen readers.

AAM differs in online query-independent writes, sparse model-feature nodes, local plasticity,
temporal links, attractor-style recurrence, and source/replay/capacity mechanics.

- https://arxiv.org/abs/2601.03417

### SAGE — self-evolving associative graph memory

SAGE couples a memory writer with a graph-foundation-model reader whose feedback changes the
memory. It narrows claims around dynamic graph memory, associative evidence recovery, and
reader-writer self-evolution.

AAM's narrower test is whether a local, sparse, activation-level rule can produce useful associations
without requiring a graph LLM to synthesize relations.

- https://arxiv.org/abs/2605.12061

### TierMem — provenance-aware escalation to raw evidence

*From Lossy to Verified* keeps summaries linked to immutable raw logs and escalates when compact
memory is insufficient. This overlaps strongly with AAM's insistence that latent memory remain linked
to authoritative sources.

AAM does not claim provenance-linked tiering in isolation; it treats TierMem as a source-verification
and evidence-allocation baseline.

- https://arxiv.org/abs/2602.17913

### Human-Like Lifelong Memory

This neuroscience-grounded proposal includes complementary stores, spreading activation, gating,
valence, and active encoding. It limits broad conceptual claims about a biologically inspired LLM
memory architecture. AAM's contribution must be an executable activation-level algorithm and
falsifiable matched-budget experiments, not the biological analogy itself.

- https://arxiv.org/abs/2603.29023

### Engram Neural Network

*Hebbian Memory-Augmented Recurrent Networks: Engram Neurons in Deep Learning* introduces an
explicit differentiable memory matrix with Hebbian plasticity and sparse retrieval in a recurrent
network. It is direct prior art for “engram neurons,” Hebbian matrices, and sparse recall, though it is
not a persistent transformer/RAG sidecar with source provenance and long-context evaluation.

- https://arxiv.org/abs/2507.21474

## Activation and latent-memory baselines

### Activation Beacon

Zhang et al., *Long Context Compression with Activation Beacon* (arXiv:2401.03462), progressively
compress chunks into special beacon tokens' layerwise K/V activations. The backbone is frozen while
beacon-specific projections are trained; historic beacon states accumulate and standard attention
reads them.

Overlap: query-independent model-native activation payloads and progressive long-context writing.

Difference: Activation Beacon does not add persistent sparse feature addresses, an explicit online
coactivation/temporal graph, feature-space recurrent completion, source-selected readout,
source-verified replay, or fixed-capacity graph maintenance. It is a strong compression/payload
baseline and a plausible AAM front end.

- https://arxiv.org/abs/2401.03462
- https://github.com/FlagOpen/FlagEmbedding

### Neurocache

Neurocache stores compressed intermediate states and retrieves nearest neighbors back into the
model. It is a close activation-retrieval baseline. AAM must beat it or an equivalent activation-kNN
condition before claiming that plasticity or recurrence matters.

- https://arxiv.org/abs/2402.01880
- https://github.com/alisafaya/neurocache

### CAMELoT

CAMELoT is a training-free consolidated associative memory for language models. It is mandatory
prior art for cache-like storage plus consolidation. AAM's added mechanisms must be isolated with
recurrence-null, shuffled-edge, and consolidation-matched controls.

- Search title: *CAMELoT: Towards Large Language Models with Training-Free Consolidated
  Associative Memory*

### Titans and ATLAS

Titans learns neural long-term memory at test time, with surprise-related updates. ATLAS studies
associative long-term memory at scale. They foreclose broad claims around test-time plasticity,
surprise-gated memory, or neural associative memory.

- https://arxiv.org/abs/2501.00663
- Search title: *Learning to Memorize at Test Time*

### MemoryLLM, M+, Larimar, MemoRAG, and MemGen

These cover persistent latent slots, scalable latent memory, episodic memory, global memory-assisted
retrieval, and generated latent memory. They are payload/latent-memory baselines, not evidence that
AAM's sparse online graph is useful.

- https://github.com/wangyu-ustc/MemoryLLM
- https://github.com/IBM/larimar

### DeepSeek Engram and related memory modules

DeepSeek's Engram work and other conditional-memory modules add scalable static or parametric
memory. They narrow use of “engram” as a name and any claim that memory modules themselves are
new. AAM uses “engram” descriptively for an episode's sparse address, not as a branding novelty.

## Foundational prior art

The following ideas are established and are not claimed individually:

- classical and modern Hopfield networks;
- fast weights and differentiable neural computers;
- recurrent memory transformers;
- neural caches and kNN language models;
- complementary learning systems and hippocampal indexing theories;
- Hebbian, covariance, Oja, BCM, STDP, decay, and homeostatic plasticity abstractions;
- sparse autoencoders, transcoders, superposition, and circuit tracing;
- experience replay, consolidation, and continual-learning regularization;
- RAG, graph RAG, temporal knowledge graphs, and spreading activation.

## Claim chart

| Element | Strong prior art? | What AAM must establish |
|---|---:|---|
| Activation/latent payload | Yes | Address/payload separation and matched-byte benefit |
| Sparse SAE features | Yes | Better retrieval substrate than text/dense/projected activations |
| Hebbian online matrix | Yes | Useful in a persistent transformer sidecar with sources |
| Associative graph traversal | Yes | Feature-level local writes and cheap recurrence add value |
| Temporal links | Yes | Better ordered/supersession recall without false chaining |
| Replay/consolidation | Yes | Source verification prevents self-poisoning at fixed capacity |
| Provenance/raw archive | Yes | Integrated with latent addresses and replay, not novel alone |
| Closed-reader hybrid | Yes in spirit | Transfer of retrieval only; no proprietary activation claim |
| Full combination | No exact match found | Must be demonstrated by confirmatory ablations |

## Publication language

Prefer:

> We evaluate a persistent activation-feature memory that combines online local plasticity,
> query-anchored completion, source-backed payloads, and verified fixed-capacity replay.

Avoid:

- “the first biologically inspired LLM memory”;
- “the first associative RAG”;
- “the first engram model”;
- “neurons are the densest possible memory”;
- “lossless latent recall”;
- “closed-model activation memory” when only source text is injected.

## Search protocol before submission

Rerun searches over arXiv, OpenReview, ACL Anthology, Semantic Scholar, Google Scholar, GitHub, and
patent indexes. Record date, exact query, inclusion reason, and forward/backward citation decisions.
Queries should combine:

- sparse activation associative memory LLM;
- SAE feature memory retrieval replay;
- Hebbian transformer memory test time;
- recurrent latent RAG hidden states;
- activation cache spreading pattern completion;
- ecphory engram graph memory agent;
- temporal association graph language model;
- K/V memory replay consolidation;
- provenance latent memory raw archive;
- closed-reader surrogate activation retrieval.

Start forward/backward searches from MRAgent, EcphoryRAG, LatentGraphMem, SAGE, TierMem,
Activation Beacon, Neurocache, CAMELoT, Titans, ATLAS, MemoryLLM, Larimar, and modern Hopfield
work. Archive a machine-readable bibliography and frozen PDFs with the submission materials.
