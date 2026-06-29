from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any
from aamemory.config import loadconfig
from aamemory.data.registry import builddataset
from aamemory.eval.runner import ExperimentRunner
def printjson(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
def commandrun(args: argparse.Namespace) -> int:
    config = loadconfig(args.config)
    if args.outputdir:
        config.outputdir = args.outputdir
    if args.limit is not None:
        config.evaluation["limit"] = args.limit
    report = ExperimentRunner(config).run()
    printjson(report["aggregate"])
    print(Path(config.outputdir).resolve())
    return 0
def commandsmoke(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config) if args.config else root / "configs" / "experiments" / "l0_cpu_smoke.yaml"
    config = loadconfig(config_path)
    if args.outputdir:
        config.outputdir = args.outputdir
    report = ExperimentRunner(config).run()
    printjson(report["aggregate"])
    return 0
def commandprepare(args: argparse.Namespace) -> int:
    config = loadconfig(args.config)
    dataset = builddataset(config.dataset)
    output = Path(args.outputdir or "data/prepared")
    output.mkdir(parents=True, exist_ok=True)
    limit = args.limit
    rows = []
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        rows.append(
            {
                "example_id": example.example_id,
                "task": example.task,
                "events": len(example.events),
                "query_characters": len(example.query),
                "evidence": len(example.evidence_ids),
            }
        )
    (output / "dataset_index.json").write_text(json.dumps(rows, indent=2))
    printjson({"examples_indexed": len(rows), "output": str(output.resolve())})
    return 0
def commandinspect(args: argparse.Namespace) -> int:
    printjson(loadconfig(args.config).todict())
    return 0
def commanddoctor(_: argparse.Namespace) -> int:
    optional = {}
    for module in [
        "torch",
        "transformers",
        "datasets",
        "sae_lens",
        "transformer_lens",
        "sentence_transformers",
        "openai",
        "anthropic",
        "google.genai",
    ]:
        try:
            __import__(module)
            optional[module] = True
        except ImportError:
            optional[module] = False
    printjson(
        {
            "python": platform.python_version(),
            "executable": sys.executable,
            "optional_dependencies": optional,
        }
    )
    return 0
def buildparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aam", description="Activation-Associative Memory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run an experiment YAML")
    run.add_argument("config")
    run.add_argument("--outputdir")
    run.add_argument("--limit", type=int)
    run.set_defaults(func=commandrun)
    smoke = subparsers.add_parser("smoke", help="run the dependency-free CPU experiment")
    smoke.add_argument("--config")
    smoke.add_argument("--outputdir")
    smoke.set_defaults(func=commandsmoke)
    prepare = subparsers.add_parser("prepare", help="download/load and index a configured dataset")
    prepare.add_argument("config")
    prepare.add_argument("--outputdir")
    prepare.add_argument("--limit", type=int)
    prepare.set_defaults(func=commandprepare)
    inspect = subparsers.add_parser("inspect-config", help="render a resolved YAML config")
    inspect.add_argument("config")
    inspect.set_defaults(func=commandinspect)
    doctor = subparsers.add_parser("doctor", help="report optional integration availability")
    doctor.set_defaults(func=commanddoctor)
    return parser
def main(argv: list[str] | None = None) -> int:
    parser = buildparser()
    args = parser.parse_args(argv)
    return int(args.func(args))
if __name__ == "__main__":
    raise SystemExit(main())
