"""
Streamlit Community Cloud entrypoint.

Set **Main file** to `streamlit_app.py`, or use `streamlit run streamlit_app.py` locally.
Alternatively: `streamlit run ui/dashboard.py` (same app).

Double-clicking this file or running `python streamlit_app.py` re-launches via
`python -m streamlit run` so the browser can open.
"""
from __future__ import annotations

import sys
from pathlib import Path

_APP_FILE = Path(__file__).resolve()

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        _streamlit_ctx = get_script_run_ctx() is not None
    except Exception:
        _streamlit_ctx = False
    if not _streamlit_ctx:
        import subprocess

        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(_APP_FILE), *sys.argv[1:]]
            )
        )

import mediapipe_vision_stub

mediapipe_vision_stub.apply()

import importlib.util

ROOT = _APP_FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "gesture_dashboard", ROOT / "ui" / "dashboard.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
try:
    _spec.loader.exec_module(_mod)
    _mod.main()
except Exception:
    import traceback

    traceback.print_exc()
    raise
