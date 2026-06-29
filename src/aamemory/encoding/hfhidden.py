from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.encoding.projection import GaussianRandomProjector
from aamemory.schema import SparseCode
class HFHiddenStateFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        modelname: str,
        layer: int = -1,
        pooling: str = "mean",
        outputdimension: int = 32768,
        topk: int = 128,
        seed: int = 0,
        maxlength: int = 2048,
        device: str | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        payloadmode: str = "pooled",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HFHiddenStateFeatureEncoder requires `pip install -e .[hf]`") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelname, trust_remote_code=trust_remote_code
        )
        torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelname,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device or "auto",
        )
        self.model.eval()
        hidden_size = int(self.model.config.hidden_size)
        self.projector = GaussianRandomProjector(hidden_size, outputdimension, seed=seed)
        self.topk = int(topk)
        self.layer = int(layer)
        self.pooling = pooling
        self.maxlength = int(maxlength)
        self.modelname = modelname
        self.payloadmode = payloadmode
    @property
    def dimension(self) -> int:
        return self.projector.outputdimension
    def pool(self, hidden: Any, attention_mask: Any) -> Any:
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        if self.pooling == "mean":
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if self.pooling == "max":
            masked = hidden.masked_fill(mask == 0, float("-inf"))
            return masked.max(dim=1).values
        if self.pooling == "last":
            positions = attention_mask.sum(dim=1) - 1
            return hidden[self.torch.arange(hidden.shape[0], device=hidden.device), positions]
        if self.pooling == "boundary_mean":
            first = hidden[:, 0]
            positions = attention_mask.sum(dim=1) - 1
            last = hidden[self.torch.arange(hidden.shape[0], device=hidden.device), positions]
            return 0.5 * (first + last)
        raise ValueError(f"unknown pooling mode: {self.pooling}")
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.maxlength,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[self.layer]
        pooled = self.pool(hidden, inputs["attention_mask"])[0].float().cpu().numpy()
        projected = self.projector(pooled)
        code = SparseCode.fromdense(projected, topk=self.topk, normalize=True)
        payload: dict[str, Any] = {}
        if self.payloadmode == "pooled":
            payload["pooled_activation"] = pooled.astype(np.float16).tolist()
        return EncodingResult(
            code=code,
            payload=payload,
            diagnostics={
                "encoder": "hf_hidden",
                "model": self.modelname,
                "layer": self.layer,
                "pooling": self.pooling,
                "tokens": int(inputs["input_ids"].shape[1]),
            },
        )
