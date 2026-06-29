# Model and autoencoder compatibility

Reviewed: 2026-06-21. Public releases are mutable until commit/revision hashes are recorded in a run
artifact.

## Compatibility tiers

| Tier | Address writer | Reader | Payload path | Valid claim |
|---|---|---|---|---|
| A | Same open model + exact SAE | Same open model | text, codec, or model hook | strongest model-native test |
| B | Same open model, no SAE | Same open model | projected activation or codec | activation-space test, not SAE-feature test |
| C | Open model/SAE | Different open reader | source text | retrieval transfer |
| D | Open surrogate | Closed API reader | source text only | hybrid associative retrieval |
| E | Proprietary model internals | Proprietary reader | unsupported by normal APIs | not implemented |

The address representation and the payload are independent experimental factors. An SAE can supply
an address while the payload remains source text; conversely, an activation payload can be indexed
by a dense text or projected address.

## Recommended open-model matrix

### Qwen-Scope: exact runnable starting point

Official Qwen-Scope releases provide TopK residual-stream SAEs with one checkpoint per layer. The
repository includes exact loaders and model fragments for:

| Model fragment | Base model | SAE repository | Default pilot layer |
|---|---|---|---:|
| `qwen35_2b_qwen_scope.yaml` | `Qwen/Qwen3.5-2B-Base` | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` | 16 |
| `qwen35_9b_qwen_scope.yaml` | `Qwen/Qwen3.5-9B-Base` | `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50` | 20 |
| `qwen35_27b_qwen_scope.yaml` | `Qwen/Qwen3.5-27B` | `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` | 40 |

These defaults are pilot sites, not privileged biological layers. Confirmatory work should preserve
at least one pre-registered mid-depth and one late-depth condition. Pin both the base-model revision
and independent SAE-repository revision.

`QwenScopeFeatureEncoder` validates SAE/model widths, loads checkpoints with `weights_only=True`
when supported, applies the checkpoint's token TopK, pools token features, and emits a bounded sparse
episode code. It can also attach the pooled residual as a payload.

### Gemma Scope and Gemma Scope 2

Google's original Gemma Scope covers Gemma 2, while Gemma Scope 2 provides SAEs/transcoders across
Gemma 3 layers. `SAELensFeatureEncoder` supports registry-resolved releases. The supplied Gemma 2
fragment contains a concrete canonical ID; Gemma Scope 2 names must be verified against the
installed SAE-Lens registry before execution.

Use Gemma 2 2B or Gemma 3 4B for the representation pilot, then replicate the winning condition on a
larger Gemma 3 checkpoint only after the mechanism passes.

### Llama Scope

Llama Scope provides TopK SAEs across Llama 3.1 8B layers/sublayers. The fragment intentionally
retains an unresolved SAE ID because layer/site/width names are registry-version dependent. Record
base model access/license acceptance, exact SAE ID, feature width, hook name, and repository commit.

### GPT-2/OpenAI public SAE release

OpenAI's public `sparse_autoencoder` repository contains GPT-2-small activation autoencoders and
analysis code. It is useful for a small mechanistic replication but is not an SAE for current closed
OpenAI production models. The supplied GPT-2 fragment defaults to a hidden-state projection; an
exact OpenAI-SAE converter can be added as a project-specific provider.

## No-SAE controls

Every SAE experiment requires controls that use the same model forward pass where possible:

1. pooled residual + no projection (dense activation kNN);
2. pooled residual + seeded orthogonal/Gaussian projection + top-k;
3. learned top-k bottleneck with equal output width and sparsity;
4. random SAE decoder/encoder with matched dimensions;
5. sentence embedding and lexical/BM25-style retrieval;
6. precomputed feature artifact to guarantee identical inputs across graph conditions.

This distinguishes “internal activations help” from “SAE features help” and “sparsity/graph helps.”

## Natural Language Autoencoders

Anthropic's public NLA code maps a residual activation to text with an activation verbalizer and
reconstructs an activation from that text with an activation reconstructor. Public checkpoint
families represented in this repository include:

- Qwen2.5-7B-Instruct, layer 20;
- Gemma-3-12B-IT, layer 32;
- Gemma-3-27B-IT, layer 41;
- Llama-3.3-70B-Instruct, layer 53.

AAM treats NLA as an optional **payload codec/verbalizer**, not as its associative address graph.
Recommended conditions hold the retrieved episode IDs fixed and compare source text, int8 pooled
activation, MLP codec, and NLA payload reconstruction. NLA text is not automatically trustworthy
evidence; the immutable source pointer remains authoritative.

## Closed models

Ordinary OpenAI Responses, Anthropic Messages, and Gemini GenerateContent APIs do not expose a
supported interface for arbitrary hidden-state capture, exact SAE application, persistent K/V
insertion, or model-internal Hebbian writes. Consequently, the executable closed-reader protocol is:

1. Write observations with an open writer/encoder.
2. Build and query AAM in that open feature space.
3. Select source-backed episodes at fixed top-k and character/token budget.
4. Inject identical source formatting into the closed reader.
5. Compare no-memory, dense RAG, sparse activation kNN, full AAM, and oracle evidence.
6. Record provider model string, response/request ID, usage, timestamp, and refusal/stop metadata.

Even if a public autoencoder were released for a proprietary model, same-model AAM would still
require the exact matching model revision, tokenizer, activation site, normalization convention, and
runtime activation interface. An SAE file alone does not provide that interface.

## Layer/site selection protocol

Use development data only:

1. Candidate depth fractions: 0.25, 0.50, 0.75, and final layer.
2. Candidate sites where available: residual-pre, residual-post, attention output, MLP output.
3. Measure feature sparsity, reconstruction error, episode separability, direct retrieval, and hubness.
4. Select one site by development evidence recall under a fixed byte budget.
5. Preserve one additional pre-registered site for robustness.
6. Never select a site using final test answer accuracy.

For multi-layer addresses, concatenate disjoint feature namespaces or learn a fixed fusion projection;
do not collide feature indices from separate layers.

## Revision and shape checklist

Before any accelerator run, record and validate:

- base model ID and immutable commit;
- tokenizer ID/commit and chat template;
- SAE repository/release/ID and immutable commit;
- layer, hook site, whether the captured residual is pre/post normalization;
- model hidden size, feature width, checkpoint TopK, episode TopK;
- pooling, truncation, boundary-token handling, dtype, quantization;
- library versions and any `trust_remote_code` code hash;
- a 10-example shape, determinism, reconstruction, and nonzero-count check.

## Latent injection checklist

Memory-token, side-cross-attention, and direct-K/V paths are architecture-specific. Record:

- layer count, hidden size, query heads, KV heads, and head dimension;
- pre/post-normalization architecture;
- local/global/linear-attention pattern;
- RoPE variant, base, scaling, multimodal position components, and virtual read positions;
- whether stored keys are pre- or post-rotation;
- cache class and FlashAttention implementation;
- tensor/model parallel sharding;
- payload dtype/quantization and gating initialization;
- behavior when no memory is retrieved.

For RoPE models, arbitrary post-rotation keys cannot be safely reused at a new context position. A
supported implementation should store/reconstruct pre-rotation keys or explicitly transform them to
the new virtual position.

---

## AAM-v2 update

The repository now contains the AAM-v2 hippocampal activation-memory implementation.  See `docs/AAM_V2_SPECIFICATION.md`, `docs/AAM_V2_MATHEMATICS.md`, `docs/AAM_V2_EXPERIMENTS.md`, `docs/AAM_V2_IMPLEMENTATION.md`, and `docs/AAM_V2_FILE_MAP.md` for the complete updated design, math, checkpoint suite, ablations, datasets, metrics, and implementation map.  AAM-v2 treats raw text as provenance/exact-recall fallback only; primary ranking uses activation engrams and graph dynamics.
