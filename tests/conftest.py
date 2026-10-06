"""Makes the repo root importable when the suite is run under pytest.

Tests carry their own sys.path bootstrap so they work with bare `python3`
(pytest is not a dependency of this repo). This file covers the pytest
path as well, so `pytest tests/` works if pytest is ever installed.
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
