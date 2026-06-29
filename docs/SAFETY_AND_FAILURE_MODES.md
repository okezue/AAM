# Safety, privacy, and failure modes

Activation-Associative Memory (AAM) makes long-lived state easier to write, connect, replay, and
surface. Those same properties create failure modes that ordinary stateless retrieval systems do not
have. The controls below are part of the experimental specification, not optional deployment polish.

## Memory poisoning and replay amplification

**Threat.** An untrusted observation, injected instruction, or model-generated statement forms strong
graph links and is repeatedly strengthened by replay.

**Required controls.**

- attach a source trust class, namespace, timestamp, and checksum to every episode;
- cap write salience and replay eligibility by trust class;
- quarantine generated memories until they are supported by an immutable source;
- make source-verified replay the default and log every replay-induced change;
- test contradiction detection, LTD/negative updates, and supersession handling;
- evaluate targeted poisoning, gradual poisoning, hub seeding, and replay-amplification attacks.

A reconstruction produced by a model is not evidence. Replay may use it as a *query*, but must not
promote it as a verified observation without an independent source or task signal.

## Hub capture and false association

Frequent, generic, formatting, or tokenizer-related features can become high-degree hubs. A few
recurrent steps can then move the state toward a popular but irrelevant attractor. Controls include
centered plasticity, degree normalization, hub penalties, lateral inhibition/top-k, a persistent query
anchor, edge pruning, and a precision mode with fewer recurrence steps.

Associative mode should be labelled as hypothesis generation. Factual answers must retain an exact
retrieval path and expose supporting episode IDs.

## Privacy, inversion, and topology leakage

Sparse features, pooled activations, learned payloads, replay traces, and graph neighborhoods can all
encode sensitive information even when source text is omitted. A public SAE does not make its codes
safe to publish. Treat the following as sensitive data:

- source text and source pointers;
- sparse feature IDs and values;
- activation/KV payloads and codec latents;
- graph edges, node degrees, replay trajectories, and retrieval scores;
- model, tokenizer, SAE, and feature-site metadata that makes inversion easier.

Deployment controls should include namespace isolation, encryption at rest and in transit, access
logs, explicit retention limits, purpose-scoped credentials, membership/inversion audits, and no
cross-user replay. Do not place raw memories or latent payloads in public experiment artifacts unless
the dataset license and consent model permit it.

## Deletion, unlearning, and archival policy

Deleting a source row does not automatically remove its distributed contribution from a graph. The
reference implementation therefore rebuilds learned association and temporal graphs from surviving
episodes after deletion, eviction, or consolidation. This is correct and auditable, but can be too slow
for large deployments. A production implementation may instead retain per-episode edge-delta logs
and subtract those contributions, followed by normalization and consistency checks.

A byte-capacity policy can evict both a source payload and the only exact evidence for a fact. The
experiment must state whether an immutable archive remains available. A deployment needing audit or
legal deletion should separate:

1. the online associative index;
2. the access-controlled source archive;
3. the deletion ledger and graph-rebuild state.

## Prompt injection and tool misuse

Retrieved text is untrusted content, not an instruction. Reader prompts must delimit memory blocks,
include source IDs, and explicitly prohibit following instructions found inside memories. Closed-model
hybrid runs are especially exposed because latent retrieval is converted back into text at an API
boundary. Test direct, indirect, encoded, multilingual, and multi-episode prompt injection.

Never permit retrieved memories to select tools, credentials, or side effects without an independent
policy check outside the language model.

## Closed-API boundary failures

Closed text APIs do not expose the internal activation site used by the open writer. Therefore hybrid
experiments test associative *retrieval plus textual reading*, not direct latent injection into the
closed model. Additional risks include:

- provider-side model alias changes;
- silent tokenizer or system-prompt changes;
- retries yielding different generations;
- accidental transmission of private memories;
- provider retention or logging inconsistent with the experiment's policy;
- prompt-template changes that invalidate paired comparisons.

Pin dated model identifiers where available, log request hashes and provider metadata, disable
provider retention where supported, and keep paid calls behind an explicit allow-list and cost cap.

## Temporal, identity, and contradiction errors

Entity collisions, outdated facts, and superseded records can all be strengthened by association.
Store timestamps, entity/source IDs, confidence, and supersession metadata. Evaluate latest-record
accuracy, negative-evidence retrieval, identity collisions, and contradictory episodes separately.
A memory system should be able to return “conflicting records” rather than averaging incompatible
facts into a plausible answer.

## Exact-data leakage and misleading compression claims

Random-string or secret-key performance may indicate source copying, payload leakage, benchmark
contamination, or an unreported archive channel. Report whether exact source text, pointers, or a
high-bandwidth activation payload remained available. Never imply that a low-bit latent code
losslessly recovered arbitrary high-entropy data without an information channel capable of carrying
those bits.

## NLA and verbalized activations

Natural-language activation descriptions are interpretations of hidden states, not ground-truth
causal explanations. They may omit information, hallucinate concepts, or expose private content.
Evaluate NLA payloads as a lossy codec and explanation interface. Do not use a verbalization alone as
proof that a feature caused an answer.

## Resume, provenance, and reproducibility failures

Resuming from a store created with a different model revision, tokenizer, feature site, SAE, graph
rule, or dimensionality can silently corrupt the experiment. A resume operation must compare the
stored manifest against the requested configuration and fail closed on mismatches unless the user
explicitly starts a migration. Preserve dataset fingerprints, code commit, dependency lock, random
seeds, and exact model/SAE revisions with every run.

## Operational failures

The test plan should exercise:

- graph/index corruption and interrupted writes;
- incompatible encoder or SAE checkpoints;
- unbounded edge or payload growth;
- nondeterministic ranking ties;
- stale closed-model aliases and API rate-limit retries;
- partial dataset downloads and changed upstream revisions;
- malformed external-baseline output;
- unsafe RoPE handling during direct-KV experiments;
- accelerator out-of-memory and mixed-precision instability.

The reference implementation favors explicit exceptions over silent fallback. Model-specific
memory-token, cross-attention, and direct-KV paths remain research integrations and must not be
presented as safe deployment interfaces without additional validation.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
