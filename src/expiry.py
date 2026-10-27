"""
Derivative expiry calendar
==========================

Replaces the one-line rule this bot ran on since "Change 19"::

    def is_expiry_day(current_date) -> bool:
        return current_date.weekday() == 1   # Tuesday

That rule is wrong in three separate, measurable ways, all confirmed on the
Jul 9-31 2026 tape (169 trades, 17 sessions):

1. **It does not know BSE exists.** BSE weekly expiry (Thursday) was the only
   cohort in the whole month with a positive mean R: +0.252R (n=32, 3
   sessions) vs -0.106R on ordinary days, t=+2.00. Median MFE 0.98 ATR vs
   0.73, target-hit 34.4% vs 16.0%, stop-out 21.9% vs 40.6%. The one cohort
   that behaved differently in a *useful* direction got no handling at all.

2. **It cannot tell weekly from monthly.** Jul 14 / Jul 21 (NSE weekly,
   -0.117R) and Jul 28 (NSE monthly, +0.160R) are all Tuesdays and all got
   identical treatment. It also fires on Tuesdays that are not expiries at
   all once a holiday shifts the series.

3. **It widens the stop on the day ranges compress.** ``get_expiry_adjustments``
   returns 1.3x (1.5x after 14:30). Measured on the tape: stops on NSE weekly
   Tuesdays sat at 3.04 ATR vs 2.05 ATR on ordinary days, with the target
   3.37 ATR away against a *median favourable excursion of 0.43 ATR* — the
   target is 7.8x the median move, against 3.7x on a normal day. One trade in
   22 ever reached it; 22.7% ran to the 15:20 forced close (vs 5.7%),
   neither winning nor losing, just paying two legs of charges. The
   multipliers were an assumption ("expiry is volatile"), never a
   measurement. On this tape expiry-Tuesday is the opposite: pinned.

This module only *classifies* and *looks up policy*. It ships with the policy
table set to reproduce the old behaviour byte-for-byte (NSE Tuesdays widen
1.3x/1.5x, nothing else is touched), so installing it changes no trade. What
it buys is the ability to state a per-expiry-type policy in YAML, and a
correct label on every entry for the forward analysis.

Holiday handling: a nominal expiry that falls on a listed holiday rolls BACK
to the previous working day, which is the exchange convention. With an empty
holiday list (the default) the classification is pure weekday arithmetic.
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date as _date, datetime as _datetime, time as _time, timedelta
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple


logger = logging.getLogger(__name__)


# ── Expiry type constants ────────────────────────────────────────────────────
NONE = "none"
NSE_WEEKLY = "nse_weekly"
NSE_MONTHLY = "nse_monthly"
BSE_WEEKLY = "bse_weekly"
BSE_MONTHLY = "bse_monthly"

ALL_TYPES: Tuple[str, ...] = (NSE_MONTHLY, BSE_MONTHLY, NSE_WEEKLY, BSE_WEEKLY)

#: When a date carries more than one label (e.g. an NSE monthly that is also a
#: BSE weekly after a holiday shift), the FIRST match here becomes ``primary``
#: and drives the policy lookup. Monthly outranks weekly; NSE outranks BSE
#: because the NSE series is the one the index names in this watchlist track.
PRECEDENCE: Tuple[str, ...] = ALL_TYPES


@dataclass(frozen=True)
class ExpiryInfo:
    """What kind of expiry (if any) a calendar date is."""

    date: _date
    types: Tuple[str, ...] = ()

    @property
    def is_expiry(self) -> bool:
        return bool(self.types)

    @property
    def primary(self) -> str:
        """The single label used for policy lookup and logging."""
        for t in PRECEDENCE:
            if t in self.types:
                return t
        return NONE

    @property
    def label(self) -> str:
        """All labels joined, for audit columns. ``'none'`` when not an expiry."""
        return "+".join(t for t in PRECEDENCE if t in self.types) or NONE


def _parse_date(value) -> Optional[_date]:
    """Accept ``date``/``datetime``/ISO string. Returns None if unusable."""
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, _date):
        return value
    if isinstance(value, str):
        try:
            return _date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _holiday_set(holidays: Optional[Iterable]) -> Set[_date]:
    out: Set[_date] = set()
    for h in holidays or ():
        d = _parse_date(h)
        if d is not None:
            out.add(d)
    return out


def _roll_back(d: _date, holidays: Set[_date]) -> _date:
    """Move to the previous working day while `d` is a weekend or a holiday.

    Bounded at 10 steps so a mis-entered holiday list can never spin.
    """
    for _ in range(10):
        if d.weekday() < 5 and d not in holidays:
            return d
        d -= timedelta(days=1)
    return d


def _series_for_month(year: int, month: int, weekday: int,
                      holidays: Set[_date]) -> Tuple[Set[_date], Optional[_date]]:
    """Expiry dates for one exchange series in one month.

    Returns ``(weekly_dates, monthly_date)``. The LAST nominal occurrence of
    ``weekday`` in the month is the monthly expiry; the rest are weeklies.
    Every date is rolled back over holidays/weekends first, which is why the
    monthly can land on a Monday (or, rarely, in the previous month).
    """
    days_in_month = calendar.monthrange(year, month)[1]
    nominal = [
        _date(year, month, day)
        for day in range(1, days_in_month + 1)
        if _date(year, month, day).weekday() == weekday
    ]
    if not nominal:
        return set(), None
    rolled = [_roll_back(d, holidays) for d in nominal]
    return set(rolled[:-1]), rolled[-1]


def _neighbour_months(d: _date) -> Tuple[Tuple[int, int], ...]:
    """(year, month) for d and its two neighbours.

    A holiday roll-back can push an expiry across a month boundary, so a date
    must be tested against the neighbouring months' series too.
    """
    first = d.replace(day=1)
    prev = (first - timedelta(days=1))
    nxt = (first + timedelta(days=32)).replace(day=1)
    return ((prev.year, prev.month), (d.year, d.month), (nxt.year, nxt.month))


class ExpiryCalendar:
    """Classifies dates against the NSE and BSE expiry series.

    Driven entirely by config so a series change (NSE moved its weekly expiry
    more than once in recent years) is a YAML edit rather than a code change.

    Args mirror ``config.risk.expiry_calendar``:
      nse_weekday / bse_weekday : 0=Mon .. 4=Fri, or None to disable that series
      holidays                  : trading holidays; expiries roll BACK over them
      overrides                 : explicit ``{date: type}`` map, wins outright.
                                  Use ``"none"`` to declare a date is NOT an
                                  expiry despite the weekday rule.
    """

    def __init__(self, nse_weekday: Optional[int] = 1,
                 bse_weekday: Optional[int] = 3,
                 holidays: Optional[Iterable] = None,
                 overrides: Optional[Dict] = None) -> None:
        self.nse_weekday = nse_weekday
        self.bse_weekday = bse_weekday
        self.holidays = _holiday_set(holidays)
        self.overrides: Dict[_date, str] = {}
        for k, v in (overrides or {}).items():
            d = _parse_date(k)
            if d is None:
                logger.warning("expiry calendar: ignoring unparseable override date %r", k)
                continue
            label = str(v).strip().lower()
            if label not in ALL_TYPES + (NONE,):
                logger.warning("expiry calendar: ignoring unknown override type %r for %s", v, d)
                continue
            self.overrides[d] = label
        self._cache: Dict[_date, ExpiryInfo] = {}

    def classify(self, value) -> ExpiryInfo:
        """Label a date. Never raises — an unusable input classifies as NONE."""
        d = _parse_date(value)
        if d is None:
            return ExpiryInfo(date=_date.min, types=())
        cached = self._cache.get(d)
        if cached is not None:
            return cached

        override = self.overrides.get(d)
        if override is not None:
            info = ExpiryInfo(date=d, types=() if override == NONE else (override,))
            self._cache[d] = info
            return info

        types = []
        for weekday, weekly, monthly in (
            (self.nse_weekday, NSE_WEEKLY, NSE_MONTHLY),
            (self.bse_weekday, BSE_WEEKLY, BSE_MONTHLY),
        ):
            if weekday is None:
                continue
            for (yr, mo) in _neighbour_months(d):
                weeklies, month_end = _series_for_month(yr, mo, weekday, self.holidays)
                if d == month_end:
                    types.append(monthly)
                    break
                if d in weeklies:
                    types.append(weekly)
                    break

        info = ExpiryInfo(date=d, types=tuple(t for t in PRECEDENCE if t in types))
        self._cache[d] = info
        return info

    # Kept for the old call sites; prefer classify().
    def is_expiry_day(self, value) -> bool:
        return self.classify(value).is_expiry


# ── Policy lookup ────────────────────────────────────────────────────────────

#: Reproduces the pre-2026-08 behaviour exactly: NSE Tuesdays (weekly AND
#: monthly, because the old code could not tell them apart) widen the stop
#: 1.3x, 1.5x from 14:30; the size cut is 0.7x/0.5x but is separately gated
#: off by ``expiry_day_size_cut_enabled: false``. BSE days were never touched.
#: Change these in YAML, not here.
DEFAULT_POLICY: Dict[str, Dict[str, float]] = {
    NSE_WEEKLY:  {"sl_multiplier": 1.3, "late_sl_multiplier": 1.5,
                  "size_multiplier": 0.7, "late_size_multiplier": 0.5, "skip": False},
    NSE_MONTHLY: {"sl_multiplier": 1.3, "late_sl_multiplier": 1.5,
                  "size_multiplier": 0.7, "late_size_multiplier": 0.5, "skip": False},
    BSE_WEEKLY:  {"sl_multiplier": 1.0, "late_sl_multiplier": 1.0,
                  "size_multiplier": 1.0, "late_size_multiplier": 1.0, "skip": False},
    BSE_MONTHLY: {"sl_multiplier": 1.0, "late_sl_multiplier": 1.0,
                  "size_multiplier": 1.0, "late_size_multiplier": 1.0, "skip": False},
}

NEUTRAL = {"sl_multiplier": 1.0, "size_multiplier": 1.0,
           "skip": False, "expiry_type": NONE, "expiry_label": NONE}

#: The hour/minute from which the "late" multipliers apply. 14:30 is the
#: legacy boundary; it is a config knob so the late tier can be retired.
DEFAULT_LATE_CUTOFF = _time(14, 30)


def get_adjustments(value_date, value_time, calendar_: ExpiryCalendar,
                    policy: Optional[Dict[str, Dict]] = None,
                    late_cutoff: _time = DEFAULT_LATE_CUTOFF) -> Dict:
    """Risk adjustments for a date/time.

    Returns ``{sl_multiplier, size_multiplier, skip, expiry_type,
    expiry_label}``. ``skip=True`` means the policy says do not trade this
    expiry type at all — the caller decides how to honour it.
    """
    info = calendar_.classify(value_date)
    if not info.is_expiry:
        return dict(NEUTRAL)

    table = policy if policy is not None else DEFAULT_POLICY
    rule = table.get(info.primary) or {}
    late = bool(value_time is not None and value_time >= late_cutoff)

    def pick(base: str, late_key: str) -> float:
        base_val = float(rule.get(base, 1.0))
        if not late:
            return base_val
        late_val = rule.get(late_key)
        return float(base_val if late_val is None else late_val)

    return {
        "sl_multiplier": pick("sl_multiplier", "late_sl_multiplier"),
        "size_multiplier": pick("size_multiplier", "late_size_multiplier"),
        "skip": bool(rule.get("skip", False)),
        "expiry_type": info.primary,
        "expiry_label": info.label,
    }


def calendar_from_config(config) -> ExpiryCalendar:
    """Build a calendar from ``config.risk.expiry_calendar``.

    Falls back to the legacy NSE-Tuesday / BSE-Thursday defaults when the
    section is absent, so an older config.yaml keeps working.
    """
    section = getattr(getattr(config, "risk", None), "expiry_calendar", None)
    if section is None:
        return ExpiryCalendar()
    return ExpiryCalendar(
        nse_weekday=getattr(section, "nse_weekday", 1),
        bse_weekday=getattr(section, "bse_weekday", 3),
        holidays=getattr(section, "holidays", None),
        overrides=getattr(section, "dates", None),
    )


def policy_from_config(config) -> Tuple[Dict[str, Dict], _time]:
    """Build ``(policy_table, late_cutoff)`` from ``config.risk.expiry_policy``."""
    risk = getattr(config, "risk", None)
    section = getattr(risk, "expiry_policy", None)
    if section is None:
        return DEFAULT_POLICY, DEFAULT_LATE_CUTOFF
    table: Dict[str, Dict] = {}
    for t in ALL_TYPES:
        rule = getattr(section, t, None)
        if rule is None:
            table[t] = dict(DEFAULT_POLICY[t])
            continue
        table[t] = {
            "sl_multiplier": getattr(rule, "sl_multiplier", 1.0),
            "late_sl_multiplier": getattr(rule, "late_sl_multiplier", None),
            "size_multiplier": getattr(rule, "size_multiplier", 1.0),
            "late_size_multiplier": getattr(rule, "late_size_multiplier", None),
            "skip": getattr(rule, "skip", False),
        }
    cutoff = _time(int(getattr(section, "late_cutoff_hour", 14)),
                   int(getattr(section, "late_cutoff_minute", 30)))
    return table, cutoff
