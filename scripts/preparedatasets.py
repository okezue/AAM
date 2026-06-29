from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
ALIASES = {"continual": "continual_retention", "longmemeval2": "longmemeval_v2"}
DATASET_CONFIGS = [
    "synthetic",
    "continual_retention",
    "longmemeval",
    "locomo",
    "longbench",
    "longbench_v2",
    "ruler",
    "contextual_memory",
    "overlap_identity",
    "prospection",
    "neurogenesis",
    "state_tensor",
]
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=DATASET_CONFIGS)
    parser.add_argument("--limit", type=int, default=1, help="Index this many rows as a schema check")
    parser.add_argument("--output-root", default="data/prepared")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for requested_name in args.datasets:
        name = ALIASES.get(requested_name, requested_name)
        fragment = root / "configs" / "datasets" / f"{name}.yaml"
        if not fragment.exists():
            raise FileNotFoundError(fragment)
        temp = root / ".dataset_prepare_tmp.yaml"
        dataset_yaml = fragment.read_text()
        temp.write_text(
            "name: prepare_" + name + "\nseed: 0\ndataset:\n"
            + "\n".join("  " + line for line in dataset_yaml.splitlines())
            + "\nmemory: {}\ngenerator: {type: none}\nevaluation: {}\noutput_dir: runs/prepare\n"
        )
        try:
            command = [
                sys.executable,
                "-m",
                "aamemory.cli",
                "prepare",
                str(temp),
                "--outputdir",
                str(Path(args.output_root) / name),
                "--limit",
                str(args.limit),
            ]
            subprocess.run(command, check=True, cwd=root)
        finally:
            temp.unlink(missing_ok=True)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
