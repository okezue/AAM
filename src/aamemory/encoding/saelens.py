from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.schema import SparseCode
class SAELensFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        modelname: str,
        saerelease: str,
        saeid: str,
        hook_name: str | None = None,
        topk: int = 128,
        pooling: str = "max",
        maxlength: int = 2048,
        device: str = "cpu",
        model_from_pretrained_kwargs: Mapping[str, Any] | None = None,
        sae_from_pretrained_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from sae_lens import SAE
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise ImportError("SAELensFeatureEncoder requires `pip install -e .[sae]`") from exc
        self.torch = torch
        self.model = HookedTransformer.from_pretrained(
            modelname,
            device=device,
            **dict(model_from_pretrained_kwargs or {}),
        )
        loaded = SAE.from_pretrained(
            release=saerelease,
            sae_id=saeid,
            device=device,
            **dict(sae_from_pretrained_kwargs or {}),
        )
        self.sae = loaded[0] if isinstance(loaded, tuple) else loaded
        self.hook_name = hook_name or getattr(self.sae.cfg, "hook_name", None)
        if not self.hook_name:
            raise ValueError("hook_name was not provided and is absent from SAE configuration")
        self._dimension = int(
            getattr(self.sae.cfg, "d_sae", getattr(self.sae, "d_sae", 0))
        )
        if self._dimension <= 0:
            raise ValueError("could not determine SAE feature dimension")
        self.topk = int(topk)
        self.pooling = pooling
        self.maxlength = int(maxlength)
        self.modelname = modelname
        self.saerelease = saerelease
        self.saeid = saeid
    @property
    def dimension(self) -> int:
        return self._dimension
    def pool(self, features: Any) -> Any:
        if self.pooling == "max":
            return features.abs().max(dim=1).values * features.gather(
                1, features.abs().argmax(dim=1, keepdim=True)
            ).squeeze(1).sign()
        if self.pooling == "positive_max":
            return features.max(dim=1).values
        if self.pooling == "mean":
            return features.mean(dim=1)
        if self.pooling == "sum":
            return features.sum(dim=1)
        if self.pooling == "last":
            return features[:, -1]
        raise ValueError(f"unknown SAE pooling mode: {self.pooling}")
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        tokens = self.model.to_tokens(text, prepend_bos=True)
        tokens = tokens[:, : self.maxlength]
        with self.torch.inference_mode():
            _, cache = self.model.run_with_cache(tokens, names_filter=[self.hook_name])
            raw = cache[self.hook_name]
            features = self.sae.encode(raw)
            pooled = self.pool(features)[0].float().cpu().numpy()
        code = SparseCode.fromdense(
            pooled,
            topk=self.topk,
            positiveonly=self.pooling == "positive_max",
            normalize=True,
        )
        return EncodingResult(
            code=code,
            diagnostics={
                "encoder": "sae_lens",
                "model": self.modelname,
                "release": self.saerelease,
                "saeid": self.saeid,
                "hook_name": self.hook_name,
                "tokens": int(tokens.shape[1]),
            },
        )
