from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from aamemory.models.nla import NaturalLanguageAutoencoderAdapter, getnlacheckpoint
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verbalize and optionally reconstruct activation vectors with a public NLA pair."
    )
    parser.add_argument("input_npz", help="NPZ with `activations` and optional `ids`")
    parser.add_argument("--family", required=True)
    parser.add_argument("--upstreaminferencefile", required=True)
    parser.add_argument("--av-checkpoint-dir", required=True)
    parser.add_argument("--ar-checkpoint-dir")
    parser.add_argument("--sglangurl", default="http://localhost:30000")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.input_npz, allow_pickle=False)
    activations = np.asarray(data["activations"], dtype=np.float32)
    ids = data["ids"] if "ids" in data else np.arange(len(activations)).astype(str)
    checkpoint = getnlacheckpoint(args.family)
    if activations.ndim != 2 or activations.shape[1] != checkpoint.d_model:
        raise ValueError(
            f"expected activations [N, {checkpoint.d_model}], got {activations.shape}"
        )
    adapter = NaturalLanguageAutoencoderAdapter(
        checkpoint,
        upstreaminferencefile=args.upstreaminferencefile,
        av_checkpoint_dir=args.av_checkpoint_dir,
        ar_checkpoint_dir=args.ar_checkpoint_dir,
        sglangurl=args.sglangurl,
        device=args.device,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as sink:
        for index, activation in enumerate(activations):
            if args.limit is not None and index >= args.limit:
                break
            payload = adapter.encodeactivation(
                activation,
                prompt=args.prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            row = {"id": str(ids[index]), **payload}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(destination.resolve())
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
