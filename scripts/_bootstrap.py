"""Make the repo's ``python/`` package directory importable from a plain script run.

Importing this module (first, before any ``alpha_agent`` import) inserts
``<repo>/python`` onto ``sys.path`` when it is not already importable.

Why this is needed: CPython's ``site.py`` silently skips ``.pth`` files that
carry the macOS ``UF_HIDDEN`` flag, which some Desktop/iCloud/security tooling
applies to files pip writes. When that happens the editable install of
``alpha_agent`` is present on disk but not on ``sys.path``. ``pytest`` is
unaffected (``tool.pytest.ini_options.pythonpath``); standalone scripts are not,
hence this shim. ``bash scripts/fix_venv_flags.sh`` clears the flag for tools
that import the installed package directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_PYTHON = Path(__file__).resolve().parents[1] / "python"
if _REPO_PYTHON.is_dir() and str(_REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_REPO_PYTHON))
