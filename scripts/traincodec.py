from __future__ import annotations
import argparse
from aamemory.training.trainer import CodecTrainingConfig, trainactivationcodec
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz")
    parser.add_argument("--outputdir", default="runs/codec")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learningrate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    trainactivationcodec(
        CodecTrainingConfig(
            input_npz=args.input_npz,
            outputdir=args.outputdir,
            bottleneck_dimension=args.bottleneck,
            hidden_dimension=args.hidden,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learningrate=args.learningrate,
            device=args.device,
        )
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
