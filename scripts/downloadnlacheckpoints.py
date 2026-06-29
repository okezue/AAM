from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from aamemory.models.nla import getnlacheckpoint, snapshotnlacheckpoint
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicitly snapshot a released Natural Language Autoencoder AV/AR pair."
    )
    parser.add_argument("family")
    parser.add_argument("--cachedir")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--manifest", default="third_party/NLA_CHECKPOINT_LOCK.json")
    args = parser.parse_args()
    checkpoint = getnlacheckpoint(args.family)
    av = snapshotnlacheckpoint(
        checkpoint.avcheckpoint,
        revision=args.revision,
        cache_dir=args.cachedir,
        local_files_only=args.local_files_only,
    )
    ar = snapshotnlacheckpoint(
        checkpoint.archeckpoint,
        revision=args.revision,
        cache_dir=args.cachedir,
        local_files_only=args.local_files_only,
    )
    manifest = {
        "family": asdict(checkpoint),
        "requested_revision": args.revision,
        "av_local_path": str(av),
        "ar_local_path": str(ar),
    }
    destination = Path(args.manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
