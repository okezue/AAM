from __future__ import annotations
import os
from typing import Any
from aamemory.models.base import Generation, Generator
class OpenAIResponsesGenerator(Generator):
    def __init__(self, *, model: str, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAI integration requires `pip install -e .[closed]`") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), **client_kwargs)
        self.model = model
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        content = prompt if system is None else f"System: {system}\n\nUser: {prompt}"
        response = self.client.responses.create(
            model=self.model,
            input=content,
            max_output_tokens=maxtokens,
            temperature=temperature,
        )
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return Generation(
            text=response.output_text,
            model=self.model,
            usage=usage,
            metadata={
                "response_id": getattr(response, "id", None),
                "created_at": getattr(response, "created_at", None),
                "status": getattr(response, "status", None),
            },
            raw=response,
        )
class AnthropicMessagesGenerator(Generator):
    def __init__(self, *, model: str, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Anthropic integration requires `pip install -e .[closed]`") from exc
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), **client_kwargs
        )
        self.model = model
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": maxtokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = self.client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
        }
        return Generation(
            text=text,
            model=self.model,
            usage=usage,
            metadata={
                "response_id": getattr(response, "id", None),
                "stop_reason": getattr(response, "stop_reason", None),
                "stop_sequence": getattr(response, "stop_sequence", None),
            },
            raw=response,
        )
class GeminiGenerator(Generator):
    def __init__(self, *, model: str, api_key: str | None = None, **client_kwargs: Any) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Gemini integration requires `pip install -e .[closed]`") from exc
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"), **client_kwargs)
        self.model = model
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=maxtokens,
            temperature=temperature,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        usage_meta = getattr(response, "usage_metadata", None)
        usage = usage_meta.model_dump() if hasattr(usage_meta, "model_dump") else {}
        return Generation(
            text=response.text or "",
            model=self.model,
            usage=usage,
            metadata={
                "response_id": getattr(response, "response_id", None),
                "model_version": getattr(response, "model_version", None),
            },
            raw=response,
        )
