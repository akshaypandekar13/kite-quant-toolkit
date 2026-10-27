"""Session clock — IST, not host-local.

The failure this guards against produces no error and no log line. A gate
written as "after 10:15 IST" that reads the HOST clock simply never becomes
true during the session on a UTC box, and the only symptom is that a filter
you believe is protecting you has never once fired.

That is not a hypothetical deployment. The project notes recommend Oracle
Cloud Always Free and AWS Lightsail, both of which default to UTC.

Run with: pytest tests/test_clock.py -v
"""

import sys
from datetime import datetime, time as dtime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.clock import IST, host_is_ist, ist_now, warn_if_not_ist


class TestIstNow:

    def test_resolves_the_zone(self):
        assert IST is not None, "tzdata missing — every time gate is host-local"

    def test_is_timezone_aware(self):
        assert ist_now().utcoffset() is not None

    def test_offset_is_five_thirty(self):
        off = ist_now().utcoffset()
        assert off.total_seconds() == 5.5 * 3600

    def test_it_is_not_the_host_clock_on_a_utc_box(self):
        """The whole point. On UTC these differ by 5:30; a check written for
        IST and fed the host clock is simply a different check."""
        naive = datetime.now()
        aware = ist_now()
        if not host_is_ist():
            delta_min = abs((aware.replace(tzinfo=None) - naive).total_seconds()) / 60
            assert delta_min > 300, "expected a ~330-minute shift from UTC"

    def test_warn_is_quiet_when_the_zone_resolves(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            assert warn_if_not_ist() is True
        assert not [m for m in caplog.messages if "not IST" in m]

    def test_warn_fires_when_there_is_no_zone_and_no_ist_host(self, caplog):
        import logging
        with patch('src.clock.IST', None), patch('src.clock.host_is_ist',
                                                 return_value=False):
            with caplog.at_level(logging.WARNING):
                assert warn_if_not_ist() is False
            assert any("not IST" in m for m in caplog.messages)

    def test_no_zone_but_an_ist_host_is_acceptable(self):
        with patch('src.clock.IST', None), patch('src.clock.host_is_ist',
                                                 return_value=True):
            assert warn_if_not_ist() is True


class TestVwapFilterWindow:
    """The analyzer's 10:15 VWAP warm-up must be decided in IST, and
    preferably from the CANDLE rather than from when the process ran."""

    def _bar(self, ts):
        return pd.Series({"Close": 100.0, "VWAP": 100.0}, name=ts)




class TestNoNaiveTimeOfDayComparisons:
    """A structural guard. Elapsed-time arithmetic on naive datetimes is
    fine — both ends shift together. Comparing a naive clock against an IST
    WALL-CLOCK constant is the bug, and it reads as harmless code.
    """

    FILES = ["analyzer.py", "fast_scanner.py", "realtime_engine.py"]


    def test_there_is_exactly_one_ist_definition(self):
        """Two copies of a timezone helper is how they drift apart."""
        defs = []
        for f in (Path(__file__).parent.parent / 'src').glob("*.py"):
            if f.name == "clock.py":
                continue
            if 'ZoneInfo("Asia/Kolkata")' in f.read_text():
                defs.append(f.name)
        assert not defs, f"IST defined outside clock.py: {defs}"
