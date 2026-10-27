"""
Environment and KiteTicker helpers shared by the live scripts.

- load_dotenv(): reads KEY=VALUE pairs from a .env file without overriding
  variables already exported in the shell.
- patch_twisted_signals(): lets KiteTicker run its Twisted reactor off the
  main thread without breaking Ctrl+C / SIGTERM shutdown.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv() -> Optional[Path]:
    """Auto-load a .env file at startup so users can manage env vars without
    Docker, shell exports, or systemd Environment= directives.

    Search order (first match wins):
      1. ./.env             (current working directory — where you run from)
      2. <repo>/../.env  (parent folder)
      3. <repo>/.env

    Format: KEY=VALUE per line. Quoted values strip one layer of single OR
    double quotes. Lines starting with # are comments. Blank lines ignored.
    Existing os.environ entries WIN — an explicit `export FOO=bar` before
    launch is never overridden by the file.

    Zero external deps (no python-dotenv install needed). Returns the Path
    that was loaded, or None if no .env was found.
    """
    package_dir = Path(__file__).resolve().parent              # <repo>/src
    repo_dir = package_dir.parent.parent                       # <repo>/..
    pkg_root = package_dir.parent                              # <repo>
    for candidate in (Path.cwd() / ".env", repo_dir / ".env", pkg_root / ".env"):
        if not candidate.is_file():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export "):]
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Strip a single matching pair of quotes (one layer only).
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1]
                    # Don't overwrite existing env (explicit exports win).
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            # Best-effort — never let a bad .env crash startup. Real errors
            # surface later as "ZERODHA_ACCESS_TOKEN not set" type messages.
            continue
        return candidate
    return None



# Our own SIGINT handling (Ctrl+C in __main__) is unaffected because it
# runs on the main thread and registers before any reactor call.
def patch_twisted_signals() -> None:
    try:
        from twisted.internet import reactor
    except Exception:
        return  # twisted not installed — nothing to patch
    if getattr(reactor, '_signals_patched', False):
        return
    _orig_run = reactor.run

    def _run_no_signals(*args, **kwargs):
        kwargs.setdefault('installSignalHandlers', False)
        return _orig_run(*args, **kwargs)

    reactor.run = _run_no_signals  # type: ignore[method-assign]
    reactor._signals_patched = True  # type: ignore[attr-defined]
