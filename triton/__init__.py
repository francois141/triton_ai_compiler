from __future__ import annotations

from pathlib import Path


_VENDORED_TRITON_ROOT = (
    Path(__file__).resolve().parents[1] / "triton_ptx" / "python" / "triton"
)
_VENDORED_INIT = _VENDORED_TRITON_ROOT / "__init__.py"

if not _VENDORED_INIT.is_file():
    raise ImportError(f"Vendored Triton package not found at {_VENDORED_INIT}")

__file__ = str(_VENDORED_INIT)
__path__ = [str(_VENDORED_TRITON_ROOT)]

with _VENDORED_INIT.open("r", encoding="utf-8") as vendored_init_file:
    exec(compile(vendored_init_file.read(), __file__, "exec"))
