from __future__ import annotations
import json
import subprocess
from pathlib import Path
import yaml
def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "third_party" / "baselines.yaml").read_text())
    lock = {}
    for item in manifest["repositories"]:
        url = item["url"]
        revision = item.get("revision", "HEAD")
        try:
            output = subprocess.check_output(
                ["git", "ls-remote", url, revision], text=True, stderr=subprocess.STDOUT
            ).strip()
            commit = output.split()[0] if output else None
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            commit = None
            item["resolution_error"] = str(exc)
        lock[item["name"]] = {"url": url, "requested_revision": revision, "commit": commit}
    destination = root / "third_party" / "REVISION_LOCK.json"
    destination.write_text(json.dumps(lock, indent=2))
    print(destination)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
