"""Crash-safe file writes, shared by every layer that persists a sidecar
file (models, recognition, editing). No dependency on the rest of the app,
so it can't create an import cycle with any of them.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` without ever leaving a truncated/corrupt file
    behind if the process is killed or the disk fills up mid-write.

    Writes to a same-directory temp file first, then renames into place --
    os.replace() is atomic on both POSIX and Windows as long as source and
    destination share a filesystem, which a same-directory temp file
    guarantees.
    """
    temp_path = path.with_name(f".tamis_write_{uuid.uuid4().hex}{path.suffix}")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
