from __future__ import annotations

from pathlib import Path


def test_source_tree_has_no_openwillis_runtime_references() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    blocked_tokens = ("openwillis", "OpenWillis", "OPENWILLIS")
    checked_suffixes = {".py", ".json", ".md", ".toml", ".txt", ".tsv"}
    offenders: list[str] = []

    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        content = path.read_text(encoding="utf-8")
        if any(token in content for token in blocked_tokens):
            offenders.append(str(path.relative_to(source_root.parents[1])))

    assert offenders == []
