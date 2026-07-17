"""Validated, process-local scratch directory for Media temporary files."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def resolve_media_temp_dir(configured_path: str) -> str:
    """Create and prove write access to the one Media scratch directory."""
    path = Path(configured_path).expanduser() if configured_path else Path(tempfile.gettempdir())
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"MEDIA_TEMP_DIR is not a directory: {path}")
    fd, probe = tempfile.mkstemp(prefix=".media-probe-", dir=path)
    os.close(fd)
    os.unlink(probe)
    return str(path)
