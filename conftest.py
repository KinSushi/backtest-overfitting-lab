"""Make tools/ and demo/ importable by pytest from the repository root."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("tools", "demo"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
