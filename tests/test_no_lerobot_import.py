"""The stand-alone package must not import LeRobot at runtime."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _iter_py_files() -> list[Path]:
    files = list((SRC / "mantis_alpha").rglob("*.py"))
    files += list((ROOT / "scripts").glob("*.py"))
    files += list((ROOT / "tests").glob("*.py"))
    return files


def test_source_tree_has_no_lerobot_imports() -> None:
    offenders: list[str] = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "lerobot" or alias.name.startswith("lerobot."):
                        offenders.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "lerobot" or node.module.startswith("lerobot."):
                    offenders.append(f"{path}:{node.lineno}: from {node.module} import ...")
    assert offenders == [], "LeRobot imports are forbidden:\n" + "\n".join(offenders)


def test_package_import_does_not_load_lerobot() -> None:
    import sys

    leaked_before = [k for k in list(sys.modules) if k == "lerobot" or k.startswith("lerobot.")]
    for key in leaked_before:
        del sys.modules[key]

    import mantis_alpha  # noqa: F401

    leaked = [k for k in sys.modules if k == "lerobot" or k.startswith("lerobot.")]
    assert leaked == []
