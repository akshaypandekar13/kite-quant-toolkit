"""
Index export
============
`structural_edge.py` showed the basket's overnight leg in all three size
terciles while NIFTY_50 showed almost none — but NIFTY_50 is large-cap, so it
cannot speak to the mid and small tiers where most of the drift sits. A
midcap/smallcap index can, and it is the only survivorship-free measurement
available for those segments.

That makes this plumbing load-bearing: a mistyped index that quietly exported
nothing would look identical to "that segment has no premium", which is the
one conclusion the export exists to make trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_kite_history import (  # noqa: E402
    DEFAULT_INDICES, NIFTY_50_NAME, NIFTY_50_TOKEN, KiteHistoryExporter,
    index_store_name,
)


class _FakeKite:
    """The two rows that matter: an equity and an index sharing a namespace."""

    ROWS = [
        {"tradingsymbol": "RELIANCE", "instrument_token": 111,
         "segment": "NSE-EQ", "instrument_type": "EQ"},
        {"tradingsymbol": "NIFTY 50", "instrument_token": 256265,
         "segment": "INDICES", "instrument_type": "EQ"},
        {"tradingsymbol": "NIFTY MIDCAP 100", "instrument_token": 222,
         "segment": "INDICES", "instrument_type": "EQ"},
        {"tradingsymbol": "NIFTY SMLCAP 100", "instrument_token": 333,
         "segment": "INDICES", "instrument_type": "EQ"},
    ]

    def instruments(self, exchange):
        return list(self.ROWS)


def _exporter(tmp_path):
    ex = KiteHistoryExporter(_FakeKite(), tmp_path)
    ex.load_instruments()
    return ex


# ── the store name ──────────────────────────────────────────────────────────


def test_spaces_become_underscores_so_the_filename_is_quotable():
    assert index_store_name("NIFTY MIDCAP 100") == "NIFTY_MIDCAP_100"
    assert index_store_name("nifty 50") == "NIFTY_50"
    # Idempotent: a name already in store form must not gain underscores.
    assert index_store_name("NIFTY_50") == "NIFTY_50"


def test_the_default_index_still_files_under_the_name_the_store_already_uses():
    """Every existing parquet, manifest entry and `--index-symbol` default
    says NIFTY_50. A rename here silently orphans the whole store."""
    assert [index_store_name(i) for i in DEFAULT_INDICES] == [NIFTY_50_NAME]


# ── token resolution ────────────────────────────────────────────────────────


def test_indices_and_equities_do_not_share_a_map(tmp_path):
    ex = _exporter(tmp_path)
    assert ex.token_for("RELIANCE") == 111
    assert ex.token_for("NIFTY_MIDCAP_100") == 222
    assert ex.token_for("NIFTY_SMLCAP_100") == 333
    assert "NIFTY 50" not in ex._tokens, "an index leaked into the equity map"
    assert "RELIANCE" not in ex._index_tokens


def test_an_index_resolves_from_its_underscored_store_name(tmp_path):
    """The store files it with underscores; Kite lists it with spaces. If
    only one spelling resolved, the export would fetch nothing and say
    nothing."""
    ex = _exporter(tmp_path)
    assert ex.token_for("NIFTY_MIDCAP_100") == ex.token_for("NIFTY MIDCAP 100")


def test_nifty_50_keeps_its_hardcoded_token_when_the_dump_has_no_indices(tmp_path):
    """A dump without the INDICES segment must still export exactly what it
    always did, or a re-run silently drops the one control that exists."""
    class _NoIndices(_FakeKite):
        def instruments(self, exchange):
            return [r for r in self.ROWS if r["segment"] != "INDICES"]

    ex = KiteHistoryExporter(_NoIndices(), tmp_path)
    ex.load_instruments()
    assert ex._index_tokens == {}
    assert ex.token_for(NIFTY_50_NAME) == NIFTY_50_TOKEN


def test_an_unknown_index_is_unresolvable_rather_than_matching_an_equity(tmp_path):
    ex = _exporter(tmp_path)
    assert ex.token_for("NIFTY MIDCAPP 100") is None


def test_available_indices_lists_only_indices(tmp_path):
    ex = _exporter(tmp_path)
    assert ex.available_indices() == ["NIFTY 50", "NIFTY MIDCAP 100",
                                      "NIFTY SMLCAP 100"]


# ── the loud failure ────────────────────────────────────────────────────────


def test_a_mistyped_index_stops_the_run_before_the_multi_hour_pull(tmp_path):
    """THE TEST THAT MATTERS. A typo that exported nothing would read as
    'that size segment shows no overnight premium' — a wrong conclusion,
    reached silently, about the exact question the export is for."""
    ex = _exporter(tmp_path)
    from datetime import date
    with pytest.raises(SystemExit, match="unknown index"):
        ex.run([], date(2025, 1, 1), date(2025, 2, 1),
               timeframes=("day",), indices=("NIFTY MIDCAPP 100",))
