from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from aamemory.models.base import Generator
def buildgenerator(config: Mapping[str, Any]) -> Generator | None:
    kind = str(config.get("type", "none")).lower().replace("-", "_")
    params = dict(config.get("params", {}))
    if kind in {"none", "disabled", "retrieval_only"}:
        return None
    if kind in {"hf", "huggingface", "local_hf"}:
        from aamemory.models.hf import HFGenerator
        return HFGenerator(**params)
    if kind in {"openai", "openai_responses"}:
        from aamemory.models.closed import OpenAIResponsesGenerator
        return OpenAIResponsesGenerator(**params)
    if kind in {"anthropic", "anthropic_messages"}:
        from aamemory.models.closed import AnthropicMessagesGenerator
        return AnthropicMessagesGenerator(**params)
    if kind in {"gemini", "google_gemini"}:
        from aamemory.models.closed import GeminiGenerator
        return GeminiGenerator(**params)
    raise ValueError(f"unknown generator type: {kind}")
