"""
Round-trip cost by instrument
=============================
The overnight strategy is decided by this arithmetic, not by a t-stat. The
measured overnight drift is ~18 bps on the stock basket and ~2.5 bps on the
index; whether either is harvestable depends entirely on what a round trip
costs in the instrument that can actually hold a gap.

The error that matters is a table that makes an unprofitable vehicle look
profitable, so every test here is aimed at one of the three ways that could
happen: the wrong STT schedule, a net figure printed against a vehicle that
cannot hold the position overnight, and an index vehicle netted against the
basket's drift instead of the index's own.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.costs import CostModel  # noqa: E402
from scripts._stats import CHARGE_BPS  # noqa: E402
from scripts._vehicles import (  # noqa: E402
    CNC_DELIVERY, INDEX_FUTURES, MIS_EQUITY, POSITION_NOTIONAL,
    STOCK_FUTURES_225K, STOCK_FUTURES_LOT, VEHICLES, Vehicle, vehicle_table,
)


# ── the live path must be untouched by the new STT field ────────────────────


def test_the_default_cost_model_is_byte_identical_after_adding_buy_side_stt():
    """`stt_rate_buy` was added to a module the LIVE journal uses.

    A non-zero default would have silently restated every charge the bot has
    ever reported, and the change would look like a data revision rather than
    a bug. Pinned at the exact rupee figure, not a tolerance.
    """
    m = CostModel()
    assert m.stt_rate_buy == 0.0
    assert m.round_trip_cost("BUY", 1.0, POSITION_NOTIONAL,
                             POSITION_NOTIONAL) == pytest.approx(126.50, abs=0.01)
    # And a BUY leg on its own pays no STT, which is the definition of intraday.
    assert m.leg_cost(1.0, 100_000.0, is_sell=False) == pytest.approx(
        CostModel(stt_rate_buy=0.0).leg_cost(1.0, 100_000.0, is_sell=False))


def test_intraday_reconciles_with_the_measured_friction_constant():
    """`_stats.CHARGE_BPS` = 5.94 was measured off the live journal; this
    model says 5.62 for a two-leg round trip. The gap is the 50% partial's
    THIRD leg, which pays its own brokerage. If these two ever diverge
    further than that, one of them has drifted and every bps comparison in
    the analysis is off.
    """
    m = CostModel()
    two_leg = m.round_trip_cost("BUY", 1.0, POSITION_NOTIONAL, POSITION_NOTIONAL)
    half = POSITION_NOTIONAL / 2.0
    three_leg = (m.entry_cost("BUY", 1.0, POSITION_NOTIONAL)
                 + m.exit_cost("BUY", 1.0, half) + m.exit_cost("BUY", 1.0, half))
    lo, hi = two_leg / POSITION_NOTIONAL * 1e4, three_leg / POSITION_NOTIONAL * 1e4
    assert lo < CHARGE_BPS < hi, (
        f"measured {CHARGE_BPS} should sit between the two-leg {lo:.2f} and "
        f"three-leg {hi:.2f} round trip")


# ── the schedules themselves ────────────────────────────────────────────────


def test_delivery_costs_about_20_bps_more_than_intraday_and_it_is_all_stt():
    """The single fact the overnight-in-cash question turns on.

    Delivery pays STT 0.1% on BOTH legs where intraday pays 0.025% on one.
    That difference alone is larger than the entire measured overnight drift
    on the survivorship-free control.
    """
    extra = CNC_DELIVERY.round_trip_bps - MIS_EQUITY.round_trip_bps
    # 17.5 bps more STT, +1.2 stamp, -2.1 because delivery pays no brokerage.
    # The first bounds (17-20) were fitted to a delivery figure that wrongly
    # carried the intraday Rs20 brokerage cap (24.32 instead of 22.22).
    assert 15.0 < extra < 20.0, extra
    assert CNC_DELIVERY.round_trip_bps == pytest.approx(22.22, abs=0.05)
    # Strip STT from both and the rest is within a couple of bps.
    bare_mis = CostModel(stt_rate_sell=0.0)
    bare_cnc = CostModel(brokerage_rate=0.0, stt_rate_sell=0.0, stt_rate_buy=0.0,
                         stamp_duty_rate_buy=0.00015)
    d = (bare_cnc.round_trip_cost("BUY", 1, POSITION_NOTIONAL, POSITION_NOTIONAL)
         - bare_mis.round_trip_cost("BUY", 1, POSITION_NOTIONAL, POSITION_NOTIONAL))
    assert abs(d / POSITION_NOTIONAL * 1e4) < 2.0, "STT is not the whole story"


def test_futures_are_the_only_cheap_way_to_hold_a_gap():
    """Everything the overnight thesis depends on is in this ordering."""
    assert STOCK_FUTURES_225K.round_trip_bps < MIS_EQUITY.round_trip_bps
    assert STOCK_FUTURES_225K.round_trip_bps < CNC_DELIVERY.round_trip_bps / 4
    assert STOCK_FUTURES_225K.holds_overnight and not MIS_EQUITY.holds_overnight
    # And futures are NOT available everywhere — the constraint that makes
    # the size split load-bearing.
    assert "F&O" in STOCK_FUTURES_225K.available_on


def test_the_brokerage_cap_makes_a_bigger_futures_position_cheaper_per_bp():
    """The one place size genuinely helps: Rs20 is a cap, not a rate."""
    assert STOCK_FUTURES_LOT.round_trip_bps < STOCK_FUTURES_225K.round_trip_bps
    # ...but not without limit; STT is a rate and does not amortise.
    assert STOCK_FUTURES_LOT.round_trip_bps > 2.0


# ── the two ways the table could mislead ────────────────────────────────────


def test_a_vehicle_that_cannot_hold_overnight_gets_no_overnight_net():
    """MIS is auto-squared-off before the gap happens. Printing
    '18.02 - 5.62 = +12.40' on its row would advertise a return from an
    instrument that is structurally incapable of earning it."""
    t = vehicle_table(18.02, 2.46)
    row = t[t["vehicle"] == MIS_EQUITY.name].iloc[0]
    assert row["overnight?"] == "NO"
    assert math.isnan(row["net bps"]) and row["verdict"] == "n/a"


def test_index_vehicles_net_against_the_index_drift_not_the_baskets():
    """An index future cannot capture a basket-specific premium.

    With the real measured pair (basket +18.02, index +2.46) the index row
    must come back NEGATIVE — it is the row that says the survivorship-free
    version of this trade loses money.
    """
    t = vehicle_table(18.02, 2.46)
    idx = t[t["vehicle"] == INDEX_FUTURES.name].iloc[0]
    assert idx["net bps"] == pytest.approx(2.46 - INDEX_FUTURES.round_trip_bps,
                                           abs=0.01)
    assert idx["net bps"] < 0 and idx["verdict"] == "loses"
    stock = t[t["vehicle"] == STOCK_FUTURES_225K.name].iloc[0]
    assert stock["net bps"] == pytest.approx(18.02 - STOCK_FUTURES_225K.round_trip_bps,
                                             abs=0.01)


def test_an_index_row_with_no_index_measurement_is_blank_not_the_basket_figure():
    t = vehicle_table(18.02, None)
    idx = t[t["vehicle"] == INDEX_FUTURES.name].iloc[0]
    assert math.isnan(idx["net bps"]) and idx["verdict"] == "n/a"


def test_delivery_loses_at_the_measured_basket_drift():
    """+18 bps gross is a real number and cash still loses on it."""
    t = vehicle_table(18.02, 2.46)
    cnc = t[t["vehicle"] == CNC_DELIVERY.name].iloc[0]
    assert cnc["net bps"] < 0 and cnc["verdict"] == "loses"


def test_no_gross_supplied_means_no_net_column_at_all():
    t = vehicle_table()
    assert "net bps" not in t.columns and "verdict" not in t.columns
    assert len(t) == len(VEHICLES)
    assert (t["bps"] > 0).all()


# ── the arithmetic itself ───────────────────────────────────────────────────


def test_net_bps_is_gross_minus_cost_and_nothing_else():
    v = Vehicle(name="x", model=CostModel(), notional=100_000.0,
                available_on="", holds_overnight=True)
    assert v.net_bps(10.0) == pytest.approx(10.0 - v.round_trip_bps)
    assert v.net_bps(0.0) == pytest.approx(-v.round_trip_bps)


def test_round_trip_bps_matches_the_rupee_figure():
    for v in VEHICLES:
        assert v.round_trip_bps == pytest.approx(
            v.round_trip_rupees / v.notional * 1e4)


def test_a_disabled_cost_model_is_free_so_the_table_cannot_hide_a_zero():
    """If charges are ever switched off, the table must read 0 rather than
    quietly falling back to a default schedule."""
    v = Vehicle(name="free", model=CostModel(enabled=False),
                notional=100_000.0, available_on="", holds_overnight=True)
    assert v.round_trip_bps == 0.0
    assert v.net_bps(3.0) == pytest.approx(3.0)


# ── the spread, and the gross it is netted against ──────────────────────────


def test_the_spread_is_deducted_from_every_tradeable_net():
    """The defect that read as good news for weeks: the table netted gross
    against broker charges ONLY, so it reported 'stock futures +14.88, pays'
    on a mid-to-mid number with no spread in it at all."""
    no_spread = vehicle_table(9.56, 7.09)
    with_spread = vehicle_table(9.56, 7.09, spread_bps=2.98)
    for name in (STOCK_FUTURES_LOT.name, CNC_DELIVERY.name):
        a = float(no_spread.loc[no_spread["vehicle"] == name, "net bps"].iloc[0])
        b = float(with_spread.loc[with_spread["vehicle"] == name, "net bps"].iloc[0])
        assert b == pytest.approx(a - 2.98, abs=0.01), name


def test_a_spread_large_enough_flips_the_verdict_to_loses():
    """The whole point of measuring it."""
    t = vehicle_table(9.56, 7.09, spread_bps=7.0)
    row = t[t["vehicle"] == STOCK_FUTURES_LOT.name].iloc[0]
    assert row["net bps"] < 0 and row["verdict"] == "loses"


def test_a_negative_spread_is_not_treated_as_a_rebate():
    """An index reads a small negative spread because it has no quoted market.
    Feeding that in must not ADD to the net — nobody is paid to trade."""
    base = vehicle_table(9.56, 7.09)
    neg = vehicle_table(9.56, 7.09, spread_bps=-3.0)
    a = float(base.loc[base["vehicle"] == STOCK_FUTURES_LOT.name, "net bps"].iloc[0])
    b = float(neg.loc[neg["vehicle"] == STOCK_FUTURES_LOT.name, "net bps"].iloc[0])
    assert b == pytest.approx(a, abs=0.01)


def test_net_bps_defaults_to_no_spread_so_old_callers_are_unchanged():
    v = STOCK_FUTURES_LOT
    assert v.net_bps(10.0) == pytest.approx(10.0 - v.round_trip_bps)
    assert v.net_bps(10.0, 0.0) == v.net_bps(10.0)


def test_the_table_says_which_gross_it_was_given():
    """An unlabelled table is what let an open-print number be read as a
    tradeable one. A caller that does not say gets told off in the attrs."""
    labelled = vehicle_table(9.56, 7.09, gross_label="09:16-09:25 plateau")
    assert labelled.attrs["gross_label"] == "09:16-09:25 plateau"
    assert "UNLABELLED" in vehicle_table(9.56, 7.09).attrs["gross_label"]
    assert vehicle_table(9.56, 7.09, spread_bps=2.98).attrs["spread_bps"] == 2.98


def test_the_measured_chain_reproduces_the_published_decomposition():
    """Pins the numbers quoted in the structural-edge write-up. If any drifts,
    the write-up explaining how +14.88 became +0.97 becomes wrong.
    """
    fut = STOCK_FUTURES_LOT
    assert fut.net_bps(15.01) == pytest.approx(11.87, abs=0.02)   # as printed
    assert fut.net_bps(9.56) == pytest.approx(6.42, abs=0.02)     # reachable
    assert fut.net_bps(7.09) == pytest.approx(3.95, abs=0.02)     # no selection
    assert fut.net_bps(7.09, 2.98) == pytest.approx(0.97, abs=0.02)  # spread
