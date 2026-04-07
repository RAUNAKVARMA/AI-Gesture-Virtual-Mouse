"""
Streamlit Community Cloud entrypoint.

Set **Main file** to `streamlit_app.py`, or use `streamlit run streamlit_app.py` locally.
Alternatively: `streamlit run ui/dashboard.py` (same app).
"""
from __future__ import annotations

import mediapipe_vision_stub

mediapipe_vision_stub.apply()

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
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
