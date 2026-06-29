from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
CHECKPOINT_CONFIGS = [
    "configs/experiments/c0_sparse_engram_retrieval.yaml",
    "configs/experiments/c1_context_bound_engrams.yaml",
    "configs/experiments/c2_episode_index_identity.yaml",
    "configs/experiments/c3_hebbian_graph.yaml",
    "configs/experiments/c4_stdp_temporal.yaml",
    "configs/experiments/c5_recurrent_completion.yaml",
    "configs/experiments/c6_dopamine_gated_writes.yaml",
    "configs/experiments/c7_latent_payload_codec.yaml",
    "configs/experiments/c8_memory_injection.yaml",
    "configs/experiments/c9_verified_replay.yaml",
    "configs/experiments/c10_reconsolidation.yaml",
    "configs/experiments/c11_neurogenesis.yaml",
    "configs/experiments/c12_future_simulation.yaml",
    "configs/experiments/c13_multimodal_state_tensor.yaml",
]
def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run the AAM-v2 checkpoint suite.")
    parser.add_argument("--output-root", default="runs/aam_v2_checkpoints")
    parser.add_argument("--limit", type=int, default=None, help="Override per-config evaluation limit.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate that configs exist and print the plan.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=len(CHECKPOINT_CONFIGS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    plan = []
    for idx, config in enumerate(CHECKPOINT_CONFIGS[args.start : args.stop], start=args.start):
        config_path = root / config
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        out = output_root / f"{idx:02d}_{config_path.stem}"
        plan.append({"index": idx, "config": config, "outputdir": str(out)})
        if not args.dry_run:
            command = [sys.executable, "-m", "aamemory.cli", "run", str(config_path), "--outputdir", str(out)]
            if args.limit is not None:
                command.extend(["--limit", str(args.limit)])
            subprocess.run(command, check=True, cwd=root)
    print(json.dumps({"checkpoints": plan, "dry_run": args.dry_run}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
