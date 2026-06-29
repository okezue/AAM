from __future__ import annotations
from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class CodecShape:
    input_dimension: int
    bottleneck_dimension: int
    outputdimension: int
def buildmlpactivationcodec(
    *,
    input_dimension: int,
    bottleneck_dimension: int,
    hidden_dimension: int | None = None,
    outputdimension: int | None = None,
    dropout: float = 0.0,
) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Codec training requires `pip install -e .[hf]`") from exc
    hidden = hidden_dimension or max(bottleneck_dimension * 2, input_dimension // 2)
    output = outputdimension or input_dimension
    class ActivationPayloadAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, bottleneck_dimension),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dimension, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output),
            )
        def encode(self, activations: Any) -> Any:
            return self.encoder(activations)
        def decode(self, code: Any) -> Any:
            return self.decoder(code)
        def forward(self, activations: Any) -> tuple[Any, Any]:
            code = self.encode(activations)
            return self.decode(code), code
    model = ActivationPayloadAutoencoder()
    model.shape = CodecShape(input_dimension, bottleneck_dimension, output)
    return model
