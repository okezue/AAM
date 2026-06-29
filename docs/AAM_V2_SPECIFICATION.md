# AAM-v2: Hippocampal Activation Memory

## Thesis

AAM-v2 implements memory as sparse model-internal activation engrams rather than text chunks.  Each memory consists of a sparse hidden-state address, sparse sensory/context bindings, an episode-index node, a compressed latent payload, version/provenance metadata, and local plastic graph edges.  Raw text is retained only for audit, deletion, verification, and exact high-entropy fallback; it is not used by the core retrieval scorer in the AAM-v2 condition.

## Core objects

| Object | File | Purpose |
|---|---|---|
| sparse hidden-state encoder | `encoding/*` | SAE/top-k/random/hash/precomputed model-internal features |
| context encoder | `encoding/context.py` | speaker/tool/time/modality/UI/source/state tensor features |
| hidden-context binding | `encoding/binding.py` | address composition `[hidden, context, hashed binding, neuro slots]` |
| Hebbian/temporal graph | `memory/associations.py` | LTP/LTD-like coactivation and STDP-like transitions |
| episode-index graph | `memory/completion.py` | `B` layer preserving identity of overlapping engrams |
| context graph | `memory/contextgraph.py` | hetero-association from context features to activation features |
| dopamine gate | `memory/dopamine.py` | explicit salience/prediction-error/poison-risk write gates |
| allocation gate | `memory/allocation.py` | CREB-like eligibility and anti-hub competitive allocation |
| neurogenesis | `memory/neurogenesis.py` | sidecar feature birth/maturation/pruning |
| prospection | `memory/prospection.py` | low-authority future-query latent traces |
| reconsolidation | `memory/reconsolidation.py` | versioning, supersession, correction, labile recall |
| verified replay | `memory/replayv2.py` | source-backed replay; generated/hypothetical traces quarantined |
| orchestration | `memory/hippocampal.py` | complete AAM-v2 write/query/replay system |

## Write recipe

1. Capture hidden states or sparse feature activations from an open writer model.  In CPU smoke tests this is represented by deterministic sparse feature encoders; GPU experiments should substitute SAE or hidden-state encoders.
2. Encode sensory/context features from metadata: speaker, timestamp, tool/UI state, modality, source/page, correction, task state, state tensors, and trust/risk.
3. Compose a bound address:

\[
z_t = \operatorname{TopK}([\lambda_h h_t; \lambda_r r_t; \lambda_b(h_t\odot Rr_t); n_t])
\]

4. Compute dopamine-like metrics: novelty, redundancy, loss/entropy/KL if available, contradiction, correction, user emphasis, utility, source trust, poison risk.
5. Apply eligibility allocation and anti-hub gain to active features.
6. Optionally allocate immature neurogenesis features when novelty/interference/retrieval error exceed thresholds.
7. Store the activation payload and provenance.  The payload contains sparse hidden/context codes and, in GPU modes, can contain quantized activation packets, memory-token codes, side-cross-attention keys/values, or architecture-specific K/V.
8. Update `A`, `T`, `B`, and `C`: centered Hebbian coactivation, directed temporal association, episode-index connectivity, and context-feature links.
9. Apply decay, degree caps, normalization, correction LTD, and fixed-capacity maintenance.

## Query recipe

1. Encode the query and current context into a sparse activation address.
2. Run query-anchored recurrent completion:

\[
a^{r+1}=\operatorname{TopK}(\alpha q+\beta \tilde A a^r+\gamma \tilde T a^r+\xi B^\top m^r+\chi C^\top r_q-\theta)
\]

3. Score episodes by direct match, completed match, episode-node activation, temporal compatibility, context compatibility, confidence, authority, and poison risk.
4. Decode latent payloads for memory-token, cross-attention, direct-KV, or adapter injection.  Source text fallback is allowed only for audit/exact-recall conditions and is reported separately.
5. Reconsolidate after use: verified retrieval strengthens authority; false retrieval depresses the path; corrections supersede prior versions.

## No-text-primary invariant

A run labeled `variant: aam_v2` must satisfy:

- `primary_memory_substrate == activation_engram` in run records.
- `text_used_for_scoring == false` for every AAM-v2 `QueryResult`.
- retrieved text may be rendered only for generator fallback, audit, exact random-string recovery, or source verification.
- metrics must report text/source bytes separately from latent address, graph, and payload bytes.

`tests/test_aam_v2_prospection_invariant.py` and `scripts/checknotextprimary.py` enforce the invariant for representative runs.
