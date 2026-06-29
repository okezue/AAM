from __future__ import annotations
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.schema import SparseCode
def infertokentopk(repoid: str, default: int = 50) -> int:
    match = re.search(r"(?:L0[_-]?)(\d+)", repoid, flags=re.IGNORECASE)
    return int(match.group(1)) if match else int(default)
class QwenScopeFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        modelname: str,
        saerepoid: str,
        layer: int,
        sae_revision: str | None = None,
        model_revision: str | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_filename: str | None = None,
        tokentopk: int | None = None,
        topk: int = 128,
        pooling: str = "positive_max",
        maxlength: int = 2048,
        tokenbatchsize: int = 32,
        device: str | None = None,
        saedevice: str = "cpu",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        cachedir: str | Path | None = None,
        payloadmode: str = "none",
    ) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenScopeFeatureEncoder requires `pip install -e .[hf]`"
            ) from exc
        self.torch = torch
        self.modelname = modelname
        self.saerepoid = saerepoid
        self.layer = int(layer)
        if self.layer < 0:
            raise ValueError("Qwen-Scope layer must be non-negative")
        self.sae_revision = sae_revision
        self.model_revision = model_revision
        self.pooling = pooling.lower().replace("-", "_")
        self.maxlength = int(maxlength)
        self.tokenbatchsize = max(1, int(tokenbatchsize))
        self.address_top_k = int(topk)
        self.tokentopk = int(tokentopk or infertokentopk(saerepoid))
        self.payloadmode = payloadmode.lower().replace("-", "_")
        checkpoint = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint is None:
            filename = checkpoint_filename or f"layer{self.layer}.sae.pt"
            checkpoint = Path(
                hf_hub_download(
                    repo_id=saerepoid,
                    filename=filename,
                    revision=sae_revision,
                    cache_dir=str(cachedir) if cachedir else None,
                )
            )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        self.checkpoint_path = checkpoint
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        required = {"W_enc", "b_enc"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"Qwen-Scope checkpoint is missing tensors: {sorted(missing)}")
        w_enc = state["W_enc"].detach()
        b_enc = state["b_enc"].detach()
        if w_enc.ndim != 2 or b_enc.ndim != 1 or w_enc.shape[0] != b_enc.shape[0]:
            raise ValueError("invalid Qwen-Scope W_enc/b_enc shapes")
        self._dimension = int(w_enc.shape[0])
        self._d_model = int(w_enc.shape[1])
        if self.tokentopk <= 0 or self.tokentopk > self._dimension:
            raise ValueError("tokentopk must be between 1 and the SAE width")
        if self.address_top_k <= 0:
            raise ValueError("topk must be positive")
        self.saedevice = torch.device(saedevice)
        self.w_enc = w_enc.to(device=self.saedevice, dtype=torch.float32)
        self.b_enc = b_enc.to(device=self.saedevice, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelname,
            revision=model_revision,
            cache_dir=str(cachedir) if cachedir else None,
            trust_remote_code=trust_remote_code,
        )
        torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            modelname,
            revision=model_revision,
            cache_dir=str(cachedir) if cachedir else None,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device or "auto",
        ).eval()
    @property
    def dimension(self) -> int:
        return self._dimension
    def tokenfeatures(self, residual: Any) -> Any:
        residual = residual.to(device=self.saedevice, dtype=self.torch.float32)
        pre_acts = residual @ self.w_enc.T + self.b_enc
        values, indices = pre_acts.topk(self.tokentopk, dim=-1)
        features = self.torch.zeros(
            (residual.shape[0], self._dimension),
            device=self.saedevice,
            dtype=self.torch.float32,
        )
        features.scatter_(1, indices, values)
        return features
    def poolfeatures(self, residual: Any) -> Any:
        token_count = int(residual.shape[0])
        if token_count == 0:
            return self.torch.zeros(self._dimension, device=self.saedevice)
        if self.pooling == "last":
            return self.tokenfeatures(residual[-1:])[0]
        accumulator = self.torch.zeros(
            self._dimension, device=self.saedevice, dtype=self.torch.float32
        )
        absolute_max = None
        for start in range(0, token_count, self.tokenbatchsize):
            chunk = residual[start : start + self.tokenbatchsize]
            features = self.tokenfeatures(chunk)
            if self.pooling in {"mean", "sum"}:
                accumulator += features.sum(dim=0)
            elif self.pooling == "positive_max":
                accumulator = self.torch.maximum(accumulator, features.max(dim=0).values)
            elif self.pooling in {"max", "max_abs"}:
                chunk_abs, positions = features.abs().max(dim=0)
                columns = self.torch.arange(self._dimension, device=self.saedevice)
                chunk_values = features[positions, columns]
                if absolute_max is None:
                    absolute_max = chunk_abs
                    accumulator = chunk_values
                else:
                    replace = chunk_abs > absolute_max
                    accumulator = self.torch.where(replace, chunk_values, accumulator)
                    absolute_max = self.torch.maximum(absolute_max, chunk_abs)
            else:
                raise ValueError(f"unknown Qwen-Scope pooling mode: {self.pooling}")
        if self.pooling == "mean":
            accumulator /= max(token_count, 1)
        return accumulator
    def poolresidual(self, residual: Any) -> Any:
        if self.payloadmode not in {"pooled", "pooled_residual"}:
            return None
        if self.pooling == "last":
            return residual[-1]
        if self.pooling in {"max", "max_abs"}:
            positions = residual.abs().argmax(dim=0)
            columns = self.torch.arange(residual.shape[1], device=residual.device)
            return residual[positions, columns]
        if self.pooling == "positive_max":
            return residual.max(dim=0).values
        if self.pooling == "sum":
            return residual.sum(dim=0)
        return residual.mean(dim=0)
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.maxlength,
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = output.hidden_states
        hidden_index = self.layer + 1
        if hidden_index >= len(hidden_states):
            raise ValueError(
                f"layer {self.layer} is outside model hidden-state tuple of length "
                f"{len(hidden_states)}"
            )
        residual = hidden_states[hidden_index][0].detach()
        if int(residual.shape[-1]) != self._d_model:
            raise ValueError(
                f"model residual width {int(residual.shape[-1])} does not match SAE d_model "
                f"{self._d_model}; check the exact model and checkpoint revisions"
            )
        pooled_features = self.poolfeatures(residual).float().cpu().numpy()
        code = SparseCode.fromdense(
            pooled_features,
            topk=self.address_top_k,
            positiveonly=self.pooling == "positive_max",
            normalize=True,
        )
        payload: dict[str, Any] = {}
        pooled_residual = self.poolresidual(residual)
        if pooled_residual is not None:
            payload["pooled_activation"] = (
                pooled_residual.float().cpu().numpy().astype(np.float16).tolist()
            )
        return EncodingResult(
            code=code,
            payload=payload,
            diagnostics={
                "encoder": "qwen_scope",
                "model": self.modelname,
                "model_revision": self.model_revision,
                "saerepoid": self.saerepoid,
                "sae_revision": self.sae_revision,
                "checkpoint": self.checkpoint_path.name,
                "layer": self.layer,
                "d_model": self._d_model,
                "d_sae": self._dimension,
                "tokentopk": self.tokentopk,
                "address_top_k": self.address_top_k,
                "pooling": self.pooling,
                "tokens": int(inputs["input_ids"].shape[1]),
            },
        )
