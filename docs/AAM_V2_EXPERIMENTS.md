# AAM-v2 experiments, checkpoints, and standalone mini-recipes

## Checkpoint suite

Run all checkpoint configs with:

```bash
PYTHONPATH=src python scripts/runcheckpointsuite.py --dry-run
PYTHONPATH=src python scripts/runcheckpointsuite.py --limit 10
```

| Checkpoint | Config | Mechanism | Stop/go criterion |
|---|---|---|---|
| C0 | `c0_sparse_engram_retrieval.yaml` | sparse activation engrams only | sparse code competitive with dense/text controls at matched bytes |
| C1 | `c1_context_bound_engrams.yaml` | sensory/context binding | lower false cross-context recall without large direct-recall loss |
| C2 | `c2_episode_index_identity.yaml` | synapse-specific episode identity | correct episode wins under high feature overlap |
| C3 | `c3_hebbian_graph.yaml` | centered Hebbian graph | learned graph beats zero/shuffled controls on multi-hop tasks |
| C4 | `c4_stdp_temporal.yaml` | directed temporal/STDP association | update/order tasks improve over symmetric-only graph |
| C5 | `c5_recurrent_completion.yaml` | query-anchored recurrence | depth 1-4 improves multi-hop without high hub capture |
| C6 | `c6_dopamine_gated_writes.yaml` | salience-gated writes | higher retained facts/byte than uniform writes |
| C7 | `c7_latent_payload_codec.yaml` | latent payload codec | payload non-inferiority at lower reader bytes |
| C8 | `c8_memory_injection.yaml` | memory-token/cross-attention/direct-KV hooks | open-model latent injection beats no-memory at fixed retrieval IDs |
| C9 | `c9_verified_replay.yaml` | source-verified replay | retention improves and false replay amplification stays near zero |
| C10 | `c10_reconsolidation.yaml` | versioning/correction/LTD | corrected memories supersede stale traces |
| C11 | `c11_neurogenesis.yaml` | sidecar feature birth | domain-shift/interference improves without feature explosion |
| C12 | `c12_future_simulation.yaml` | prospective traces | future hit rate improves with no factual authority leaks |
| C13 | `c13_multimodal_state_tensor.yaml` | pseudo-multimodal state encoding | state-tensor context beats text-only controls |

## Standalone ablation matrices

| Matrix | Purpose |
|---|---|
| `aam_v2_core_matrix.yaml` | sparse engram vs graph vs shuffled/random controls vs text/dense placeholders |
| `context_binding_matrix.yaml` | hidden-only vs context-bound vs metadata-only vs over-contextualized |
| `overlap_identity_matrix.yaml` | episode-index identity vs vector-only controls |
| `dopamine_gating_matrix.yaml` | full dopamine gate vs uniform, novelty-only, emphasis-only writes |
| `replay_poisoning_matrix.yaml` | verified replay vs no/aggressive/low-confidence replay |
| `neurogenesis_matrix.yaml` | feature birth vs fixed codebook and threshold sweeps |
| `prospection_matrix.yaml` | prospective traces vs no-prospection and authority stress tests |
| `no_text_primary_confirmatory.yaml` | confirms text is not primary memory substrate |

Materialize with:

```bash
PYTHONPATH=src python scripts/materializeablationmatrix.py configs/ablations/aam_v2_core_matrix.yaml --outputdir runs/materialized_aam_v2_core
```

## Dataset modules

Synthetic controlled tasks live in `src/aamemory/data/`:

- `synthetic.py`: paired associates, paraphrase, multi-hop, temporal update, interference, hub distractor, random strings, replay poisoning.
- `contextualmemory.py`: same content under different speaker/tool/source contexts.
- `overlapidentity.py`: overlapping engrams with distinct episode payloads.
- `prospection.py`: delayed future-query/prefetch tasks.
- `neurogenesis.py`: novelty plus interference streams for feature birth.
- `statetensor.py`: pseudo-multimodal numeric state canvas tasks.

Public benchmark adapters remain in the repo for LongBench, LongBench-v2, RULER, LoCoMo, and LongMemEval variants; they require external downloads or local files where licenses demand it.

## Metrics

`eval/metrics.py` includes exact answer matching, token F1, source attribution, hidden-state retrieval accuracy, false association rate, hub capture rate, bytes per retained fact, hallucinated replay amplification rate, and forgetting-curve helpers.  `eval/runner.py` writes per-example retrieval components, graph updates, edge visits, byte accounting, no-text-primary flags, and bootstrap confidence intervals.
