# AAM system specification

## 1. Scope

Activation-Associative Memory (AAM) is a sidecar memory system for a transformer reader. It stores
an episode as:

\[
\mathcal M_e=(z_e,c_e,p_e,t_e,s_e,u_e),
\]

where:

- \(z_e\) is a sparse model-native address;
- \(c_e\) is an optional compressed activation payload;
- \(p_e\) is a source pointer or immutable source text;
- \(t_e\) is time/order metadata;
- \(s_e\) is write salience;
- \(u_e\) contains confidence, access, privacy, and utility metadata.

The system is intended to test a scientific claim: model-internal sparse feature patterns may be a
better address space for persistent memory than text embeddings, especially for associative and
multi-hop recall. It is not intended to replace exact source storage.

## 2. Required invariants

1. **Provenance:** every factual memory returned to a reader has a source identifier.
2. **Query anchoring:** recurrent state always contains a configurable contribution from the query.
3. **Sparse boundedness:** each code and graph row has a hard cardinality bound.
4. **Replay verification:** unverified generations are not potentiated by default.
5. **User isolation:** stores and graphs are scoped to a memory namespace.
6. **Deletion consistency:** deleting, evicting, or merging an episode removes its postings and then
   rebuilds the reference graph from retained episodes by default. A production system may instead
   use per-write edge-contribution logs, but must not leave unreported ghost associations.
7. **Matched accounting:** experiments report source, sparse-address, payload, metadata, posting-index,
   node-state, and graph-edge bytes separately, together with write/read compute and reader context.
8. **Independent trials:** a fresh non-resume run clears both the store and graph; resuming requires an
   explicit graph checkpoint paired with the persistent store.

## 3. Write path

### 3.1 Segmentation

An input stream is segmented at semantically meaningful boundaries when possible: message, event,
paragraph, document section, or model-defined boundary token. Chunk size is an experimental factor.
Token-level storage is allowed only as an ablation because it greatly increases graph writes.

### 3.2 Activation capture

For an open model, choose a model, layer, site, and pooling function:

\[
H_e=\{h_{e,t}^{(\ell,s)}\}_{t=1}^{n_e}.
\]

Candidate sites include residual-pre, residual-post, attention output, MLP output, or an Activation
Beacon state. The capture layer/site is part of the memory schema and cannot silently change between
write and query.

### 3.3 Pattern separation

Preferred SAE path:

\[
f_{e,t}=\operatorname{SAE}_{\ell,s}(h_{e,t}),\qquad
z_e=\operatorname{Normalize}(\operatorname{TopK}(\operatorname{Pool}_t f_{e,t})).
\]

No-SAE control:

\[
z_e=\operatorname{Normalize}(\operatorname{TopK}(R\operatorname{Pool}_t H_e)),
\]

with a seeded fixed random projection \(R\). Dense text embeddings, lexical hashing, and learned
bottlenecks are controls, not interchangeable labels.

### 3.4 Salience gate

A write gate approximates a three-factor plasticity rule:

\[
\delta_e=\operatorname{clip}(b+
\alpha\,\text{surprise}+
\beta\,\text{task relevance}+
\gamma\,\text{user importance}+
\nu\,\text{novelty}-
\kappa\,\text{redundancy}).
\]

No signal is inferred from private user data unless explicitly supplied by the host application.
Synthetic tests set these fields directly.

### 3.5 Coactivation and temporal writes

A symmetric association graph \(A\) receives within-episode updates. A directed graph \(T\) receives
previous-to-current updates. The default centered rule is:

\[
A_{ij}\leftarrow (1-\rho)A_{ij}+
\eta\delta_e(z_{e,i}-\mu_i)(z_{e,j}-\mu_j),
\]

\[
T_{ij}\leftarrow (1-\rho_T)T_{ij}+
\eta_T\delta_e z_{e-1,i}z_{e,j}.
\]

Hebb, Oja, and BCM approximations are implemented as ablations. Rows are top-degree pruned and
norm-capped. Production-scale work should use a sparse tensor/graph engine; the reference Python
implementation prioritizes auditability.

### 3.6 Payload

Supported payload tiers:

1. Source text/pointer only.
2. Pooled hidden activation compressed with symmetric int8.
3. Learned MLP bottleneck.
4. Natural Language Autoencoder code/verbalization.
5. Learned memory tokens.
6. Side cross-attention values.
7. Direct model-specific K/V.

Tiers 5–7 require post-training. Direct K/V must account for GQA/MQA shapes and positional
encoding. For RoPE, a safe design stores pre-rotation keys or reconstructs them, assigns virtual
positions at read time, and applies the target model's rotation then. Reusing arbitrary post-RoPE keys
at new positions is not a valid generic implementation.

## 4. Read path

### 4.1 Cue encoding

The query uses the same encoder schema as writes. A model/SAE version mismatch is a hard error in a
production backend.

### 4.2 Recurrent completion

Initialize \(a^{(0)}=q\). For a small fixed number of iterations:

\[
a^{(r+1)}=\operatorname{TopK}\left[\sigma\left(
\alpha q+\beta D_A^{-1/2}AD_A^{-1/2}a^{(r)}+
\gamma(T+T^\top)a^{(r)}-\theta
\right)\right].
\]

The reference implementation uses ReLU plus L2 normalization for unsigned features and tanh for
signed controls. Degree normalization and an extra hub penalty are independently ablated.

### 4.3 Candidate generation and ranking

An inverted feature index finds episodes containing query or completed features. The score is:

\[
S(e|q)=w_x z_e^\top q+w_a z_e^\top a^{(R)}+
 w_t z_e^\top a_T+w_rR(t_e)+w_cC_e.
\]

Exact and associative terms are logged separately. A result can therefore be diagnosed as a direct
match, a completed association, or a temporal recall.

### 4.4 Reader integration

- **Text mode:** inject selected, source-labelled text. This is the only portable closed-model mode.
- **Memory-token mode:** decode payloads into trainable prefix vectors.
- **Cross-attention mode:** add a gated memory attention block.
- **Direct-K/V mode:** model-specific and position-aware; never claimed by the generic adapter.

The reader prompt states that associative matches are candidate memories and that claims must be
supported by retrieved source content.

## 5. Replay

Replay seeds are selected by random, recency, salience, uncertainty, or associative-centrality
policies. Before strengthening a trace, the default verifier checks an immutable source checksum.
A generated claim without source verification is rejected.

A full replay cycle may:

1. Select a source-backed episode.
2. Recall neighboring features/episodes.
3. Reconstruct a payload or reader prediction.
4. Compare against the original source/full-context teacher.
5. Strengthen verified paths and weaken contradicted paths.
6. Update usage and uncertainty metadata.

The reference `ReplayEngine` implements source-gated graph updates. Teacher distillation and
contradiction-driven LTD belong in the GPU harness.

## 6. Consolidation and capacity

Working memory remains in the ordinary recent KV cache. Fast episodic memory stores individual
traces. Slow semantic memory merges repeated, mutually consistent episodes into prototypes or
slow adapters.

Capacity policies under test:

- unbounded control;
- FIFO;
- LRU;
- salience/utility eviction;
- graph-aware redundancy removal;
- prototype consolidation;
- replay into a LoRA adapter followed by source retention.

A fixed-capacity experiment must count graph and index bytes, not only episode payloads. The
reference implementation supports both episode-count and logical-byte limits. After any eviction or
prototype merge it rebuilds the graph from retained episodes so capacity maintenance and deletion
remain semantically aligned. Logical bytes are portable comparison units; physical SQLite/WAL size
is logged separately when available.

## 7. Interfaces

The stable Python surfaces are:

```python
memory = ActivationAssociativeMemory(config)
memory.write(MemoryEvent(...))
results = memory.query("...")
memory.savegraph("graph.json")
memory.close()
```

Encoder plugins implement `FeatureEncoder.encode`. Stores implement `EpisodeStore`. Readers
implement `Generator`. Model-specific activation injection implements `ActivationInjectionAdapter`.
Pinned external baselines can implement the JSON file contract in `JSONSubprocessBaseline`; the
request and response remain auditable even when the baseline itself is maintained upstream.

## 8. Failure behavior

The system must fail loudly for:

- encoder/graph dimension mismatch;
- unknown dataset schema;
- unavailable optional dependency;
- missing model-specific direct-injection hook;
- source-free replay when verification is required;
- incompatible SAE/model layer/site metadata.

A plausible-looking fallback is more dangerous than an explicit error in this research setting.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
