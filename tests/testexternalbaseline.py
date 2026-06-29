from __future__ import annotations
import sys
from pathlib import Path
from aamemory.integrations.subprocessbaseline import JSONSubprocessBaseline
def testjsonsubprocessbaselineroundtrip(tmp_path: Path) -> None:
    script = tmp_path / "baseline.py"
    script.write_text(
        """
import json
import sys
request = json.load(open(sys.argv[1], encoding='utf-8'))
json.dump({'value': request['value'] * 2}, open(sys.argv[2], 'w', encoding='utf-8'))
""".strip(),
        encoding="utf-8",
    )
    runner = JSONSubprocessBaseline(
        [sys.executable, str(script), "{input}", "{output}"], timeout_seconds=30
    )
    result = runner.run({"value": 7})
    assert result.output == {"value": 14}
    assert result.returncode == 0
    assert result.seconds >= 0
