from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import yaml
from aamemory.config import loadconfig
def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML fragment must be a mapping: {path}")
    return value
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose a runnable experiment from a base, model fragment, and dataset fragment."
    )
    parser.add_argument("base")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--name")
    parser.add_argument("--outputdir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load(Path(args.base))
    if args.dataset:
        config["dataset"] = load(Path(args.dataset))
    if args.model:
        model = load(Path(args.model))
        config.setdefault("memory", {})
        config.setdefault("evaluation", {})
        if "encoder" in model:
            config["memory"]["encoder"] = model["encoder"]
        elif "type" in model:
            config["memory"]["encoder"] = {
                "type": model["type"],
                "params": model.get("params", {}),
            }
        else:
            raise ValueError("model fragment must contain `type` or `encoder`")
        if "generator" in model:
            config["generator"] = model["generator"]
        if "nlapayload" in model:
            config["evaluation"]["nlapayload"] = model["nlapayload"]
    if args.name:
        config["name"] = args.name
    if args.outputdir:
        config["outputdir"] = args.outputdir
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    resolved = loadconfig(destination)
    print(
        json.dumps(
            {
                "output": str(destination.resolve()),
                "name": resolved.name,
                "dataset": resolved.dataset.get("type"),
                "encoder": resolved.memory.encoder.type,
                "generator": resolved.generator.get("type"),
            },
            indent=2,
        )
    )
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
