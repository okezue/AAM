# AAM-v2 file map

## New code

- `src/aamemory/encoding/context.py`: sparse sensory/context encoder.
- `src/aamemory/encoding/binding.py`: hidden/context/binding/neurogenesis address composer.
- `src/aamemory/memory/engram.py`: activation engram/version dataclasses.
- `src/aamemory/memory/dopamine.py`: dopamine/salience metrics and LTP/LTD gates.
- `src/aamemory/memory/allocation.py`: eligibility and competitive allocation.
- `src/aamemory/memory/contextgraph.py`: context-to-feature hetero-association graph.
- `src/aamemory/memory/completion.py`: episode-index graph and hippocampal completion network.
- `src/aamemory/memory/reconsolidation.py`: versioning, correction, labile recall updates.
- `src/aamemory/memory/neurogenesis.py`: sidecar feature birth/maturation/pruning.
- `src/aamemory/memory/prospection.py`: future-prompt prospective traces.
- `src/aamemory/memory/replayv2.py`: source-verified replay and quarantine.
- `src/aamemory/memory/hippocampal.py`: full AAM-v2 system.
- `src/aamemory/data/contextualmemory.py`: context association diagnostic dataset.
- `src/aamemory/data/overlapidentity.py`: synapse-specific identity dataset.
- `src/aamemory/data/prospection.py`: future-query simulation dataset.
- `src/aamemory/data/neurogenesis.py`: feature-birth dataset.
- `src/aamemory/data/statetensor.py`: pseudo-multimodal state tensor dataset.

## Updated code

- `src/aamemory/config.py`: adds `memory.variant` and AAM-v2 config namespaces.
- `src/aamemory/eval/runner.py`: instantiates AAM-v2, records no-text-primary flags and new metrics.
- `src/aamemory/eval/metrics.py`: adds source/hidden retrieval, false association, replay amplification, forgetting helpers.
- `src/aamemory/data/registry.py`: registers new datasets.
- `scripts/preparedatasets.py`: includes AAM-v2 diagnostic datasets.

## New configs

- `configs/experiments/aam_v2_cpu_smoke.yaml`.
- `configs/experiments/c0_*.yaml` through `c13_*.yaml`.
- `configs/ablations/*_matrix.yaml` for core, context, identity, dopamine, replay, neurogenesis, prospection, and no-text-primary ablations.
- `configs/datasets/contextual_memory.yaml`, `overlapidentity.yaml`, `prospection.yaml`, `neurogenesis.yaml`, `statetensor.yaml`.

## New scripts

- `scripts/runcheckpointsuite.py` dry-runs or executes all AAM-v2 checkpoint configs.
- `scripts/checknotextprimary.py` validates run artifacts for the no-text-primary invariant.

## New tests

- `tests/test_aam_v2_context_binding.py`.
- `tests/test_aam_v2_episode_identity.py`.
- `tests/test_aam_v2_dopamine_neurogenesis.py`.
- `tests/test_aam_v2_reconsolidation_replay.py`.
- `tests/test_aam_v2_prospection_invariant.py`.
