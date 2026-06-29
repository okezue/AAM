from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from aamemory.config import EncoderConfig
from aamemory.encoding.base import FeatureEncoder
from aamemory.encoding.hashing import HashingFeatureEncoder
def buildencoder(config: EncoderConfig | Mapping[str, Any]) -> FeatureEncoder:
    if isinstance(config, EncoderConfig):
        encoder_type = config.type
        params = dict(config.params)
    else:
        encoder_type = str(config.get("type", "hashing"))
        params = dict(config.get("params", {}))
    normalized = encoder_type.lower().replace("-", "_")
    if normalized in {"hash", "hashing", "feature_hashing"}:
        return HashingFeatureEncoder(**params)
    if normalized in {"sentence_transformer", "sentence_transformers"}:
        from aamemory.encoding.sentencetransformer import SentenceTransformerFeatureEncoder
        return SentenceTransformerFeatureEncoder(**params)
    if normalized in {"hf_hidden", "hidden_state", "activation"}:
        from aamemory.encoding.hfhidden import HFHiddenStateFeatureEncoder
        return HFHiddenStateFeatureEncoder(**params)
    if normalized in {"transformer_lens_hook", "tl_hook", "hook_activation"}:
        from aamemory.encoding.transformerlenshook import TransformerLensHookFeatureEncoder
        return TransformerLensHookFeatureEncoder(**params)
    if normalized in {"sae", "sae_lens"}:
        from aamemory.encoding.saelens import SAELensFeatureEncoder
        return SAELensFeatureEncoder(**params)
    if normalized in {"qwen_scope", "qwenscope", "qwen_sae"}:
        from aamemory.encoding.qwenscope import QwenScopeFeatureEncoder
        return QwenScopeFeatureEncoder(**params)
    if normalized in {"precomputed", "precomputed_sparse", "feature_artifact"}:
        from aamemory.encoding.precomputed import PrecomputedFeatureEncoder
        return PrecomputedFeatureEncoder(**params)
    raise ValueError(f"unknown encoder type: {encoder_type}")
