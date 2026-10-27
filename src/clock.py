"""Session clock.

NSE trades in IST. Every time-of-day decision in this codebase — market
hours, entry windows, the VWAP warm-up, session-minute maths — is written
against IST wall-clock, so any of them that reads the HOST clock is wrong by
however many hours the host is shifted.

That is not hypothetical here. The deployment targets in the project notes
(Oracle Cloud Always Free, AWS Lightsail) default to UTC, and on a UTC host
`datetime.now().time()` is IST minus 5:30 — so a check written as "after
10:15 IST" only becomes true at 15:45 IST, half an hour AFTER the 15:15
forced flat close. The filter does not misfire; it never fires at all, and
nothing in the logs says so.

Import `ist_now()` rather than reaching for `datetime.now()` whenever the
question is "what time is it in the session".
"""

from __future__ import annotations

import logging
import time as _time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo          # stdlib >= 3.9
    IST: Optional[object] = ZoneInfo("Asia/Kolkata")
except Exception:                          # pragma: no cover - tzdata missing
    IST = None                             # fall back to host-local time


def ist_now() -> datetime:
    """Current time in IST, tz-aware when the zone could be resolved.

    Falls back to host-local time when `zoneinfo` has no tzdata to read.
    That fallback is correct only on an IST host, which is what
    `warn_if_not_ist()` exists to say out loud at startup.

    Honours the replay clock override when one is installed — see
    `set_clock_override`. Live runs never install one, so this is the wall
    clock in production.
    """
    if _clock_override is not None:
        return _clock_override()
    return datetime.now(IST) if IST is not None else datetime.now()


def host_is_ist() -> bool:
    """Whether the host clock itself is IST (UTC+5:30, or +6:30 in DST terms).

    Only meaningful when `IST` is None — otherwise the zone does the work.
    """
    offset_hours = -_time.timezone // 3600
    return offset_hours in (5, 6)


def warn_if_not_ist() -> bool:
    """Warn once when neither zoneinfo nor the host clock can give us IST.

    Returns True when the clock is trustworthy for session decisions.
    """
    if IST is not None:
        return True
    if host_is_ist():
        return True
    logger.warning(
        f"⚠️  zoneinfo unavailable and host TZ offset UTC+{-_time.timezone // 3600} "
        f"is not IST. Every time-of-day gate — market hours, entry window, the "
        f"VWAP warm-up — will be off by that shift. Run under Asia/Kolkata or "
        f"install tzdata."
    )
    return False


# ── Replay seam ─────────────────────────────────────────────────────────────
# `ist_now()` is the single place the codebase asks "what time is it in the
# session". An offline replay has to answer that question with the simulated
# bar time rather than the wall clock, otherwise every time-of-day rule —
# the entry window, the RULE_D morning band, session quality — reads the
# machine's clock and the replay measures a different system from the live
# one.
#
# The override is process-global and OFF unless something sets it. Nothing on
# the trading path calls `set_clock_override`; it exists for `src/replay.py`
# and for tests. Kept here rather than threaded through every call site so
# the production signature of `ist_now()` does not change.

_clock_override = None


def set_clock_override(fn) -> None:
    """Make `ist_now()` return `fn()` instead of the wall clock.

    Args:
        fn: zero-argument callable returning a datetime, or None to clear.

    Not thread-safe by design: a replay is single-threaded over its clock,
    and a live process must never call this.
    """
    global _clock_override
    _clock_override = fn


def clear_clock_override() -> None:
    """Restore `ist_now()` to the wall clock."""
    set_clock_override(None)


@contextmanager
def clock_at(fn):
    """Scope a clock override, restoring whatever was set before.

    Nesting is safe — the previous override is restored, not cleared, so a
    replay step inside a replay session does not silently unpin the clock.
    """
    global _clock_override
    prev = _clock_override
    _clock_override = fn
    try:
        yield
    finally:
        _clock_override = prev
