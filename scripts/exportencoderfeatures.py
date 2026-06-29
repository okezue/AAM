from __future__ import annotations
import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from aamemory.config import loadconfig
from aamemory.data.registry import builddataset
from aamemory.encoding.factory import buildencoder
from aamemory.encoding.precomputed import textsha256
def jsondefault(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)
def exportone(
    *,
    text: str,
    metadata: Mapping[str, Any],
    encoder: Any,
    include_text: bool,
) -> dict[str, Any]:
    result = encoder.encode(text, metadata=metadata)
    row: dict[str, Any] = {
        "textsha256": textsha256(text),
        "dimension": result.code.dimension,
        "indices": list(result.code.indices),
        "values": list(result.code.values),
        "payload": dict(result.payload),
        "diagnostics": dict(result.diagnostics),
    }
    if include_text:
        row["text"] = text
    return row
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export revision-locked sparse features for all event/query texts in an experiment."
    )
    parser.add_argument("experiment", help="Experiment YAML containing dataset and memory.encoder")
    parser.add_argument("--output", required=True, help="Destination JSONL feature artifact")
    parser.add_argument("--limit-examples", type=int)
    parser.add_argument("--events-only", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    config = loadconfig(args.experiment)
    dataset = builddataset(config.dataset)
    encoder = buildencoder(config.memory.encoder)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows = 0
    examples = 0
    with output.open("w", encoding="utf-8") as sink:
        for example_index, example in enumerate(dataset):
            if args.limit_examples is not None and example_index >= args.limit_examples:
                break
            examples += 1
            for event in example.events:
                key = textsha256(event.text)
                if key in seen:
                    continue
                metadata = {
                    **dict(event.metadata),
                    "artifact_role": "event",
                    "event_id": event.event_id,
                    "example_id": example.example_id,
                }
                row = exportone(
                    text=event.text,
                    metadata=metadata,
                    encoder=encoder,
                    include_text=args.include_text,
                )
                sink.write(json.dumps(row, ensure_ascii=False, default=jsondefault) + "\n")
                seen.add(key)
                rows += 1
            if not args.events_only:
                key = textsha256(example.query)
                if key not in seen:
                    row = exportone(
                        text=example.query,
                        metadata={
                            **dict(example.metadata),
                            "artifact_role": "query",
                            "example_id": example.example_id,
                        },
                        encoder=encoder,
                        include_text=args.include_text,
                    )
                    sink.write(json.dumps(row, ensure_ascii=False, default=jsondefault) + "\n")
                    seen.add(key)
                    rows += 1
    manifest = {
        "experiment": str(Path(args.experiment).resolve()),
        "output": str(output.resolve()),
        "examples": examples,
        "unique_texts": rows,
        "encoder": asdict(config.memory.encoder),
        "contains_raw_text": bool(args.include_text),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
