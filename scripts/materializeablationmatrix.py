from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any
import yaml
from aamemory.config import loadconfig
def setpath(root: dict[str, Any], path: str, value: Any) -> None:
    cursor = root
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
ALIASES = {
    "plasticityrule": "memory.graph.rule",
    "recurrencesteps": "memory.retrieval.recurrencesteps",
    "hubpenalty": "memory.graph.hubpenalty",
    "queryanchor": "memory.retrieval.queryanchor",
    "topk": "memory.encoder.params.topk",
    "pooling": "memory.encoder.params.pooling",
    "edgecondition": "memory.graph.edgecondition",
    "edgeseed": "memory.graph.edgeseed",
    "temporallearningrate": "memory.graph.temporallearningrate",
    "associationstrength": "memory.retrieval.associationstrength",
    "temporalstrength": "memory.retrieval.temporalstrength",
}
def applymodelfragment(config: dict[str, Any], fragment_path: Path) -> None:
    fragment = yaml.safe_load(fragment_path.read_text(encoding="utf-8")) or {}
    if "encoder" in fragment:
        config.setdefault("memory", {})["encoder"] = fragment["encoder"]
    elif "type" in fragment:
        config.setdefault("memory", {})["encoder"] = {
            "type": fragment["type"],
            "params": fragment.get("params", {}),
        }
    else:
        raise ValueError(f"model fragment has no encoder/type: {fragment_path}")
    if "generator" in fragment:
        config["generator"] = fragment["generator"]
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--outputdir", default="runs/materialized_configs")
    parser.add_argument("--max-runs", type=int, default=100)
    args = parser.parse_args()
    matrix_path = Path(args.matrix)
    matrix = yaml.safe_load(matrix_path.read_text())
    root = matrix_path.resolve().parents[2]
    base = yaml.safe_load((root / matrix["base"]).read_text())
    output = Path(args.outputdir)
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    if "conditions" in matrix:
        combinations = matrix["conditions"]
    else:
        names = list(matrix["factors"])
        products = itertools.product(*(matrix["factors"][name] for name in names))
        combinations = [dict(zip(names, values, strict=True)) for values in products]
    for index, condition in enumerate(combinations):
        if written >= args.max_runs:
            break
        config = copy.deepcopy(base)
        name = condition.get("name", f"ablation_{index:05d}")
        config["name"] = name
        config["outputdir"] = f"runs/ablations/{name}"
        unresolved = {}
        if condition.get("modelfragment"):
            applymodelfragment(config, root / str(condition["modelfragment"]))
        for key, value in condition.items():
            if key in {"name", "modelfragment"}:
                continue
            if key in ALIASES:
                setpath(config, ALIASES[key], value)
            elif "." in key:
                setpath(config, key, value)
            else:
                unresolved[key] = value
        config.setdefault("notes", "")
        if unresolved:
            config["notes"] += "\nUNRESOLVED_DESIGN_FACTORS=" + json.dumps(unresolved, sort_keys=True)
        destination = output / f"{index:05d}_{name}.yaml"
        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        loadconfig(destination)
        written += 1
    print(written, output)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
