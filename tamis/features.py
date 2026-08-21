"""Which optional extras this build actually has.

Both optional features degrade silently by design: without their dependencies
the tabs, scores and controls are simply absent, which is right for a user who
never installed them and useless for telling a *packaged* build apart from a
broken one. v2.3.0 and v2.3.1 shipped executables whose release workflow never
installed the quality extra, so the feature switched itself off exactly as
intended and the downloads looked simply featureless. This module exists so
`--version` can report what is present, making that verifiable from outside
the app.

Deliberately uses `find_spec` rather than importing anything: importing torch
takes seconds and pulls hundreds of MB into memory, which is far too much for
a `--version` call. The specs checked here are the same ones MainWindow's
guarded imports ultimately depend on, so the answers agree.
"""

from __future__ import annotations

import importlib.util


def _has(*modules: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in modules)


def recognition_available() -> bool:
    """Mirrors main_window.RECOGNITION_AVAILABLE."""
    return _has("torch", "torchvision", "facenet_pytorch")


def quality_available() -> bool:
    """Mirrors main_window.QUALITY_AVAILABLE."""
    return _has("torch", "open_clip")


def summary() -> list[str]:
    """One line per optional feature, for `--version` and the About dialog."""
    return [
        f"Face recognition: {'enabled' if recognition_available() else 'not installed'}",
        f"Quality scoring:  {'enabled' if quality_available() else 'not installed'}",
    ]
