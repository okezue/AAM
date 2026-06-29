from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from aamemory.encoding.base import EncodingResult, FeatureEncoder
from aamemory.encoding.projection import GaussianRandomProjector
from aamemory.schema import SparseCode
_SITE_TO_SUFFIX = {
    "residual_pre": "hook_resid_pre",
    "resid_pre": "hook_resid_pre",
    "residual_post": "hook_resid_post",
    "resid_post": "hook_resid_post",
    "mlp_output": "hook_mlp_out",
    "mlp_out": "hook_mlp_out",
    "attention_output": "hook_attn_out",
    "attn_out": "hook_attn_out",
}
class TransformerLensHookFeatureEncoder(FeatureEncoder):
    def __init__(
        self,
        *,
        modelname: str,
        layer: int,
        site: str = "residual_post",
        hook_name: str | None = None,
        pooling: str = "mean",
        outputdimension: int = 32768,
        topk: int = 128,
        seed: int = 0,
        maxlength: int = 2048,
        device: str = "cpu",
        prepend_bos: bool = True,
        payloadmode: str = "pooled",
        model_from_pretrained_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise ImportError(
                "TransformerLensHookFeatureEncoder requires `pip install -e .[sae]`"
            ) from exc
        normalized_site = site.lower().replace("-", "_")
        if hook_name is None:
            try:
                suffix = _SITE_TO_SUFFIX[normalized_site]
            except KeyError as exc:
                raise ValueError(f"unknown TransformerLens activation site: {site}") from exc
            hook_name = f"blocks.{int(layer)}.{suffix}"
        self.torch = torch
        self.model = HookedTransformer.from_pretrained(
            modelname,
            device=device,
            **dict(model_from_pretrained_kwargs or {}),
        )
        hidden_size = int(self.model.cfg.d_model)
        self.projector = GaussianRandomProjector(hidden_size, outputdimension, seed=seed)
        self.modelname = modelname
        self.layer = int(layer)
        self.site = normalized_site
        self.hook_name = hook_name
        self.pooling = pooling
        self.topk = int(topk)
        self.maxlength = int(maxlength)
        self.prepend_bos = bool(prepend_bos)
        self.payloadmode = payloadmode
    @property
    def dimension(self) -> int:
        return self.projector.outputdimension
    def pool(self, hidden: Any) -> Any:
        if self.pooling == "mean":
            return hidden.mean(dim=1)
        if self.pooling == "max":
            positions = hidden.abs().argmax(dim=1, keepdim=True)
            return hidden.gather(1, positions).squeeze(1)
        if self.pooling == "positive_max":
            return hidden.max(dim=1).values
        if self.pooling == "sum":
            return hidden.sum(dim=1)
        if self.pooling == "last":
            return hidden[:, -1]
        if self.pooling == "boundary_mean":
            return 0.5 * (hidden[:, 0] + hidden[:, -1])
        raise ValueError(f"unknown pooling mode: {self.pooling}")
    def encode(self, text: str, *, metadata: Mapping[str, Any] | None = None) -> EncodingResult:
        tokens = self.model.to_tokens(text, prepend_bos=self.prepend_bos)[:, : self.maxlength]
        with self.torch.inference_mode():
            _, cache = self.model.run_with_cache(tokens, names_filter=[self.hook_name])
            if self.hook_name not in cache:
                raise KeyError(
                    f"hook {self.hook_name!r} was not captured; check model/site compatibility"
                )
            hidden = cache[self.hook_name]
            pooled_tensor = self.pool(hidden)[0]
        pooled = pooled_tensor.float().cpu().numpy()
        projected = self.projector(pooled)
        code = SparseCode.fromdense(
            projected,
            topk=self.topk,
            positiveonly=self.pooling == "positive_max",
            normalize=True,
        )
        payload: dict[str, Any] = {}
        if self.payloadmode == "pooled":
            payload["pooled_activation"] = pooled.astype(np.float16).tolist()
        return EncodingResult(
            code=code,
            payload=payload,
            diagnostics={
                "encoder": "transformer_lens_hook",
                "model": self.modelname,
                "layer": self.layer,
                "site": self.site,
                "hook_name": self.hook_name,
                "pooling": self.pooling,
                "tokens": int(tokens.shape[1]),
            },
        )
