from pathlib import Path
import re


def test_external_process_creation_is_owned_by_managed_wrapper() -> None:
    root = Path(__file__).parents[2] / "src"
    forbidden = re.compile(
        r"(?:create_subprocess|subprocess\.(?:Popen|run|call|check_call)|os\.system)"
    )
    violations: list[str] = []
    for path in root.rglob("*.py"):
        if path == root / "common" / "managed_process.py":
            continue
        if forbidden.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(root)))
    assert violations == []
