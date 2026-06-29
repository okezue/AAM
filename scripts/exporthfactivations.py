from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
def main() -> int:
    parser = argparse.ArgumentParser(description="Export pooled hidden activations to an NPZ codec dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-jsonl", required=True, help="Rows must contain a text field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", choices=["mean", "max", "last"], default="mean")
    parser.add_argument("--maxlength", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device
    ).eval()
    device = next(model.parameters()).device
    exported: list[np.ndarray] = []
    ids: list[str] = []
    with Path(args.input_jsonl).open() as source, torch.inference_mode():
        for index, line in enumerate(source):
            if args.limit is not None and index >= args.limit:
                break
            row = json.loads(line)
            text = str(row["text"])
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=args.maxlength
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[args.layer]
            mask = inputs["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            if args.pooling == "mean":
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            elif args.pooling == "max":
                pooled = hidden.masked_fill(mask == 0, float("-inf")).max(1).values
            else:
                position = inputs["attention_mask"].sum(1) - 1
                pooled = hidden[torch.arange(hidden.shape[0], device=device), position]
            exported.append(pooled[0].float().cpu().numpy())
            ids.append(str(row.get("id", index)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, activations=np.stack(exported), ids=np.asarray(ids))
    print(len(exported), output_path)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
