from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import yaml
def main() -> int:
    parser = argparse.ArgumentParser(description="Clone selected external baselines at locked commits.")
    parser.add_argument("--names", nargs="*", help="Repository names; default is every manifest row")
    parser.add_argument("--destination", default="third_party/checkouts")
    parser.add_argument("--manifest", default="third_party/baselines.yaml")
    parser.add_argument("--lock", default="third_party/REVISION_LOCK.json")
    parser.add_argument("--allow-unlocked", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / args.manifest).read_text(encoding="utf-8"))
    lock_path = root / args.lock
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    selected = set(args.names or [row["name"] for row in manifest["repositories"]])
    destination = root / args.destination
    destination.mkdir(parents=True, exist_ok=True)
    for row in manifest["repositories"]:
        name = row["name"]
        if name not in selected:
            continue
        commit = lock.get(name, {}).get("commit")
        if not commit and not args.allow_unlocked:
            raise RuntimeError(
                f"{name} has no resolved commit. Run scripts/resolveexternalrevisions.py first."
            )
        target = destination / name
        if not target.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", row["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--all", "--tags"], check=True)
        subprocess.run(
            ["git", "-C", str(target), "checkout", commit or row.get("revision", "HEAD")],
            check=True,
        )
        print(name, target)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
