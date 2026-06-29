from __future__ import annotations
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
import numpy as np
@dataclass(frozen=True)
class NaturalLanguageAutoencoderCheckpoint:
    name: str
    base_model: str
    layer: int
    d_model: int
    avcheckpoint: str
    archeckpoint: str
KNOWN_PUBLIC_NLA_CHECKPOINTS: tuple[NaturalLanguageAutoencoderCheckpoint, ...] = (
    NaturalLanguageAutoencoderCheckpoint(
        name="qwen2.5-7b-l20",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        layer=20,
        d_model=3584,
        avcheckpoint="kitft/nla-qwen2.5-7b-L20-av",
        archeckpoint="kitft/nla-qwen2.5-7b-L20-ar",
    ),
    NaturalLanguageAutoencoderCheckpoint(
        name="gemma3-12b-l32",
        base_model="google/gemma-3-12b-it",
        layer=32,
        d_model=3840,
        avcheckpoint="kitft/nla-gemma3-12b-L32-av",
        archeckpoint="kitft/nla-gemma3-12b-L32-ar",
    ),
    NaturalLanguageAutoencoderCheckpoint(
        name="gemma3-27b-l41",
        base_model="google/gemma-3-27b-it",
        layer=41,
        d_model=5376,
        avcheckpoint="kitft/nla-gemma3-27b-L41-av",
        archeckpoint="kitft/nla-gemma3-27b-L41-ar",
    ),
    NaturalLanguageAutoencoderCheckpoint(
        name="llama3.3-70b-l53",
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        layer=53,
        d_model=8192,
        avcheckpoint="kitft/Llama-3.3-70B-NLA-L53-av",
        archeckpoint="kitft/Llama-3.3-70B-NLA-L53-ar",
    ),
)
def getnlacheckpoint(name: str) -> NaturalLanguageAutoencoderCheckpoint:
    normalized = name.lower()
    for checkpoint in KNOWN_PUBLIC_NLA_CHECKPOINTS:
        if checkpoint.name.lower() == normalized:
            return checkpoint
    names = ", ".join(checkpoint.name for checkpoint in KNOWN_PUBLIC_NLA_CHECKPOINTS)
    raise KeyError(f"unknown NLA checkpoint family {name!r}; expected one of: {names}")
def loadupstreammodule(path: str | Path) -> ModuleType:
    module_path = Path(path)
    if module_path.is_dir():
        module_path = module_path / "nla_inference.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"NLA upstream inference file not found: {module_path}. Clone kitft/nla-inference "
            "or set `upstreaminferencefile` to its nla_inference.py."
        )
    spec = importlib.util.spec_from_file_location("aamemory_external_nla_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import NLA inference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def snapshotnlacheckpoint(
    repoid: str,
    *,
    revision: str | None = None,
    cachedir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("NLA checkpoint resolution requires `pip install -e .[hf]`") from exc
    return Path(
        snapshot_download(
            repo_id=repoid,
            revision=revision,
            cache_dir=str(cachedir) if cachedir else None,
            local_files_only=local_files_only,
        )
    )
class NaturalLanguageAutoencoderAdapter:
    def __init__(
        self,
        checkpoint: NaturalLanguageAutoencoderCheckpoint,
        *,
        upstreaminferencefile: str | Path,
        av_checkpoint_dir: str | Path,
        ar_checkpoint_dir: str | Path | None = None,
        sglangurl: str = "http://localhost:30000",
        device: str = "cpu",
        injection_scale_override: float | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        module = loadupstreammodule(upstreaminferencefile)
        if not hasattr(module, "NLAClient"):
            raise AttributeError("upstream module does not expose NLAClient")
        self.client = module.NLAClient(
            av_checkpoint_dir,
            sglangurl=sglangurl,
            injection_scale_override=injection_scale_override,
            device=device,
        )
        if int(self.client.cfg.d_model) != checkpoint.d_model:
            raise ValueError(
                f"NLA AV d_model={self.client.cfg.d_model} does not match registered "
                f"{checkpoint.name} d_model={checkpoint.d_model}"
            )
        self.critic = None
        if ar_checkpoint_dir is not None:
            if not hasattr(module, "NLACritic"):
                raise AttributeError("upstream module does not expose NLACritic")
            self.critic = module.NLACritic(ar_checkpoint_dir, device=device)
    def encodeactivation(
        self,
        activation: Any,
        *,
        prompt: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        extract_explanation: bool = True,
    ) -> Mapping[str, Any]:
        vector = np.asarray(activation, dtype=np.float32).reshape(-1)
        if vector.size != self.checkpoint.d_model:
            raise ValueError(
                f"activation has {vector.size} values; {self.checkpoint.name} requires "
                f"{self.checkpoint.d_model}"
            )
        explanation = self.client.generate(
            vector,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            extract_explanation=extract_explanation,
        )
        payload: dict[str, Any] = {
            "codec": "natural_language_autoencoder",
            "checkpoint_family": self.checkpoint.name,
            "avcheckpoint": self.checkpoint.avcheckpoint,
            "layer": self.checkpoint.layer,
            "explanation": explanation,
        }
        if self.critic is not None:
            mse, cosine = self.critic.score(explanation, vector)
            payload["roundtrip_mse"] = float(mse)
            payload["roundtrip_cosine"] = float(cosine)
            payload["archeckpoint"] = self.checkpoint.archeckpoint
        return payload
