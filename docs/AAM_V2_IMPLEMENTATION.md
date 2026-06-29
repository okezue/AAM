# AAM-v2 implementation notes

## Open-model path

True AAM-v2 requires model internals: layer/site hidden states, SAE features or a sparse bottleneck, tokenizer/checkpoint compatibility, and a memory injection path.  The repo supports dependency-free hashing/projection smoke tests, precomputed feature artifacts, Hugging Face hidden-state export, SAE-Lens/TransformerLens hooks, Qwen-Scope configs, and placeholders for architecture-specific K/V injection.

Recommended order:

1. `aam_v2_cpu_smoke.yaml` to validate the graph/store/eval path.
2. `c0` and `c1` using precomputed hidden features from a local open model.
3. `c2`/`c3` with SAE features to test sparse feature identity and Hebbian graph utility.
4. `c7` with quantized hidden payloads.
5. `c8` with memory-token or side-cross-attention injection.
6. Direct-K/V only after RoPE, head layout, layer order, and pre/post-rotation conventions are explicitly verified for the model.

## Closed-model path

Closed hosted APIs cannot honestly consume AAM-v2 activation engrams unless they expose compatible internal activations or latent-memory injection.  The supported closed-model mode is therefore explicitly hybrid:

```text
open writer model -> activation engrams -> associative completion -> source-backed compact rendering -> closed reader
```

That tests activation-based retrieval transfer, not closed-model latent memory.

## Activation Beacon baseline positioning

Activation Beacon compresses long context by interleaving beacon tokens, storing their layerwise K/V activations, discarding raw-token activations, and accumulating beacon activations for later chunks.  AAM-v2 treats this as a strong activation-payload/compression baseline.  The distinguishing AAM-v2 mechanism is persistent sparse engram addressing, local plasticity, recurrent completion, source-selected payload reinstatement, replay, reconsolidation, context binding, prospection, and neurogenesis.

## GPU placeholders that are intentional

The following files define contracts without assuming a particular compute cluster:

- `models/direct_kv.py` should be implemented per architecture before direct-KV experiments.
- `models/cross_attention.py` and `models/memory_tokens.py` can be added for trainable injection modules.
- `scripts/exporthfactivations.py` and `scripts/exportencoderfeatures.py` are the entry points for producing precomputed activation artifacts.
- Slurm/AWS templates live under `infrastructure/` and intentionally omit credentials/account IDs.

No CPU fallback is presented as evidence for direct-KV or proprietary closed-model latent consumption.
