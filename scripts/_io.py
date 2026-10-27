"""Atomic file writes for every script.

A write goes to `<name>.tmp` beside the target and is renamed over it only
once complete. On a full disk (ENOSPC) or a crash mid-write the call fails
but the previous file stays whole; an in-place write truncates it first and
fails after — which is how a refresh on a full EC2 disk emptied a store file
and could have emptied the paper logs.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path):
    """Yield a temp path; on success it replaces `path`, on failure it is removed."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def atomic_open(path, mode: str = "w", newline: str = "", **kw):
    """`open(path, "w")` that never leaves `path` half-written."""
    with atomic_path(path) as tmp, open(tmp, mode, newline=newline, **kw) as f:
        yield f


def write_csv(df, path, **kw) -> None:
    """`df.to_csv(path, **kw)`, atomically. Pass `compression` explicitly:
    the temp name's `.tmp` suffix defeats inference from `.gz`."""
    with atomic_path(path) as tmp:
        df.to_csv(tmp, **kw)


def append_csv(df, path, **kw) -> None:
    """`df.to_csv(path, mode="a", **kw)` on a copy, swapped in when complete,
    so a failed append cannot leave a torn last row."""
    with atomic_path(path) as tmp:
        shutil.copy2(path, tmp)
        df.to_csv(tmp, mode="a", **kw)


def write_text(path, text: str) -> None:
    with atomic_path(path) as tmp:
        Path(tmp).write_text(text)
