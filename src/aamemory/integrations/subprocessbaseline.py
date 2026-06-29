from __future__ import annotations
import json
import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
@dataclass(frozen=True)
class ExternalBaselineResult:
    command: tuple[str, ...]
    returncode: int
    seconds: float
    output: Mapping[str, Any]
    stdout: str
    stderr: str
@dataclass
class JSONSubprocessBaseline:
    command: str | Sequence[str]
    cwd: str | Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 3600.0
    def argv(self, input_path: Path, output_path: Path) -> list[str]:
        template = shlex.split(self.command) if isinstance(self.command, str) else list(self.command)
        if not template:
            raise ValueError("external baseline command must not be empty")
        argv = [
            part.replace("{input}", str(input_path)).replace("{output}", str(output_path))
            for part in template
        ]
        if not any("{input}" in part for part in template):
            raise ValueError("command must contain an {input} placeholder")
        if not any("{output}" in part for part in template):
            raise ValueError("command must contain an {output} placeholder")
        return argv
    def run(self, request: Mapping[str, Any]) -> ExternalBaselineResult:
        with tempfile.TemporaryDirectory(prefix="aam-baseline-") as temporary:
            root = Path(temporary)
            input_path = root / "request.json"
            output_path = root / "response.json"
            input_path.write_text(
                json.dumps(dict(request), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            argv = self.argv(input_path, output_path)
            environment = os.environ.copy()
            environment.update({str(key): str(value) for key, value in self.env.items()})
            started = time.perf_counter()
            completed = subprocess.run(
                argv,
                cwd=str(self.cwd) if self.cwd is not None else None,
                env=environment,
                capture_output=True,
                text=True,
                timeout=float(self.timeout_seconds),
                check=False,
            )
            seconds = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    "external baseline failed with return code "
                    f"{completed.returncode}: {completed.stderr[-4000:]}"
                )
            if not output_path.exists():
                raise RuntimeError("external baseline completed without writing its output JSON")
            output = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(output, dict):
                raise ValueError("external baseline output must be a JSON object")
            return ExternalBaselineResult(
                command=tuple(argv),
                returncode=completed.returncode,
                seconds=seconds,
                output=output,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
