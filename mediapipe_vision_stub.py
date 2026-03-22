"""
Prevent MediaPipe's vision/__init__.py from loading drawing_utils (which imports cv2).

The stock ``mediapipe.tasks.python.vision`` package runs ``import drawing_styles`` and
``import drawing_utils``; ``drawing_utils`` requires OpenCV and thus libgthread on Linux.
Streamlit Cloud often lacks those libs. We only need ``hand_landmarker`` and ``core``.

Call :func:`apply` once before any ``import mediapipe`` (e.g. first line of ``ui/dashboard.py``).

Safe on local dev: skipping optional drawing helpers does not affect our pipeline.
"""
from __future__ import annotations

import os
import sys
import types

_APPLIED = False


def apply() -> None:
    """Register a minimal ``mediapipe.tasks.python.vision`` package (submodules load from disk)."""
    global _APPLIED
    if _APPLIED:
        return
    name = "mediapipe.tasks.python.vision"
    if name in sys.modules:
        return

    try:
        import site
    except ImportError:
        return

    roots: list[str] = list(site.getsitepackages())
    us = site.getusersitepackages()
    if us:
        roots.append(us)

    vision_dir: str | None = None
    for root in roots:
        d = os.path.join(root, "mediapipe", "tasks", "python", "vision")
        if os.path.isdir(d):
            vision_dir = d
            break

    if not vision_dir:
        return

    mod = types.ModuleType(name)
    mod.__path__ = [vision_dir]
    mod.__doc__ = "Stub: submodules load without executing stock vision/__init__.py (no cv2)."
    sys.modules[name] = mod
    _APPLIED = True
