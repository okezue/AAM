from __future__ import annotations
import argparse
import json
from pathlib import Path
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate no-text-primary invariant for AAM-v2 run artifacts.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    records = Path(args.run_dir) / "per_example.jsonl"
    if not records.exists():
        raise FileNotFoundError(records)
    violations = []
    with records.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, start=1):
            row = json.loads(line)
            if row.get("primary_memory_substrate") == "activation_engram" and row.get("text_used_for_scoring"):
                violations.append({"line": line_no, "example_id": row.get("example_id")})
    print(json.dumps({"violations": violations, "ok": not violations}, indent=2))
    return 1 if violations else 0
if __name__ == "__main__":
    raise SystemExit(main())
