"""Syntax checks for all Python source files.

Uses py_compile so no ML dependencies are required in CI.
"""

import glob
import py_compile
import pytest


def _collect_sources():
    patterns = [
        "model/*.py",
        "dataset/*.py",
        "utils/*.py",
        "*.py",
    ]
    seen = set()
    files = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


@pytest.mark.parametrize("filepath", _collect_sources())
def test_syntax(filepath):
    """All .py files must parse without syntax errors."""
    py_compile.compile(filepath, doraise=True)
