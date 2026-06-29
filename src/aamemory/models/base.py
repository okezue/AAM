from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw: Any = None
class Generator(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        raise NotImplementedError
