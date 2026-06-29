from __future__ import annotations
from typing import Any
from aamemory.models.base import Generation, Generator
class HFGenerator(Generator):
    def __init__(
        self,
        *,
        modelname: str,
        device: str | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HFGenerator requires `pip install -e .[hf]`") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelname, trust_remote_code=trust_remote_code
        )
        torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
        self.model_object = AutoModelForCausalLM.from_pretrained(
            modelname,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
            trust_remote_code=trust_remote_code,
        )
        self.model_object.eval()
        self.model = modelname
        self.use_chat_template = use_chat_template
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        maxtokens: int = 256,
        temperature: float = 0.0,
    ) -> Generation:
        if self.use_chat_template and self.tokenizer.chat_template:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            rendered = prompt if system is None else f"{system}\n\n{prompt}"
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.model_object.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": maxtokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(temperature, 1e-5)
        with self.torch.inference_mode():
            output = self.model_object.generate(**inputs, **generation_kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(
            text=text,
            model=self.model,
            usage={
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": int(generated.shape[0]),
            },
            metadata={
                "model_class": type(self.model_object).__name__,
                "tokenizer_class": type(self.tokenizer).__name__,
                "generation_mode": "sample" if do_sample else "greedy",
            },
            raw=None,
        )
