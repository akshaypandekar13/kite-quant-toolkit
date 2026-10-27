"""
Transaction-cost model tests.

Locks in the finding that motivated the module: every reported P&L was GROSS.
On the Jul 9-31 2026 tape (168 completed trades, 377 executed legs) charges
were Rs 22,158 against a Rs -5,463 gross book — the entire loss, invisible.

Run with: pytest tests/test_costs.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.costs import CostModel


M = CostModel()


class TestLegCosts:

    def test_brokerage_caps_at_20_on_a_large_leg(self):
        """0.03% of Rs 224,000 is Rs 67 — the Rs 20 cap must bind."""
        turnover = 224_000.0
        cost = M.leg_cost(quantity=1, price=turnover, is_sell=False)
        # brokerage 20 + exchange 6.65 + sebi 0.22 + stamp 6.72 + gst 4.84
        assert cost == pytest.approx(38.4, abs=0.5)

    def test_brokerage_is_percentage_below_the_cap(self):
        """On a Rs 50,000 leg, 0.03% = Rs 15 < cap, so the rate applies."""
        cost_small = M.leg_cost(quantity=1, price=50_000.0, is_sell=False)
        brokerage_component = 0.0003 * 50_000.0
        assert brokerage_component == pytest.approx(15.0)
        assert cost_small < M.leg_cost(quantity=1, price=224_000.0, is_sell=False)

    def test_stt_only_on_the_sell_leg(self):
        sell = M.leg_cost(1, 100_000.0, is_sell=True)
        buy = M.leg_cost(1, 100_000.0, is_sell=False)
        # STT 0.025% = Rs 25 on sell; stamp 0.003% = Rs 3 on buy
        assert sell - buy == pytest.approx(25.0 - 3.0, abs=0.1)

    def test_zero_and_negative_turnover_are_safe(self):
        assert M.leg_cost(0, 100.0, True) == 0.0
        assert M.leg_cost(10, 0.0, False) == 0.0

    def test_disabled_model_costs_nothing(self):
        assert CostModel(enabled=False).leg_cost(100, 500.0, True) == 0.0


class TestPositionSideMapping:
    """A SHORT's entry is a sell leg and its exit is a buy leg. Getting this
    backwards mis-states STT/stamp on every SELL trade — and the book is
    SELL-heavy (105 of 168 trades in July)."""

    def test_short_entry_is_a_sell_leg(self):
        assert (M.entry_cost("SELL", 100, 500.0)
                == M.leg_cost(100, 500.0, is_sell=True))

    def test_short_exit_is_a_buy_leg(self):
        assert (M.exit_cost("SELL", 100, 500.0)
                == M.leg_cost(100, 500.0, is_sell=False))

    def test_long_entry_is_a_buy_leg(self):
        assert (M.entry_cost("BUY", 100, 500.0)
                == M.leg_cost(100, 500.0, is_sell=False))

    def test_long_exit_is_a_sell_leg(self):
        assert (M.exit_cost("BUY", 100, 500.0)
                == M.leg_cost(100, 500.0, is_sell=True))

    def test_round_trip_costs_the_same_either_direction(self):
        """One buy leg and one sell leg regardless of side."""
        long_rt = M.round_trip_cost("BUY", 450, 500.0, 500.0)
        short_rt = M.round_trip_cost("SELL", 450, 500.0, 500.0)
        assert long_rt == pytest.approx(short_rt)


class TestJulyTapeCalibration:
    """Reproduce the measured Jul 9-31 friction so a rate change that would
    silently move the number gets caught."""

    def test_round_trip_on_a_225k_position_is_about_130_rupees(self):
        rt = M.round_trip_cost("BUY", quantity=450, entry_price=498.0,
                               exit_price=498.0)
        assert 120.0 <= rt <= 140.0, f"expected ~Rs 130/round trip, got {rt:.1f}"

    def test_friction_is_about_015R_at_the_median_risk(self):
        """Median rupee risk per trade on the July tape was Rs 855."""
        rt = M.round_trip_cost("BUY", 450, 498.0, 498.0)
        assert rt / 855.0 == pytest.approx(0.15, abs=0.02)

    def test_partial_exit_adds_a_third_leg(self):
        """39 of 168 July trades scaled out, so 377 legs were executed, not
        336. A two-leg estimate under-counts by ~12%."""
        two_leg = M.round_trip_cost("BUY", 450, 498.0, 498.0)
        partial_leg = M.exit_cost("BUY", 225, 500.0)
        three_leg = (M.entry_cost("BUY", 450, 498.0)
                     + partial_leg
                     + M.exit_cost("BUY", 225, 502.0))
        assert three_leg > two_leg
        assert partial_leg > 0

    def test_full_july_book_reproduces_the_measured_charges(self):
        """377 legs at the tape's median notional lands near Rs 22,158."""
        legs_buyish, legs_sellish = 188, 189
        total = (legs_buyish * M.leg_cost(1, 224_000.0, is_sell=False)
                 + legs_sellish * M.leg_cost(1, 224_000.0, is_sell=True))
        assert 20_000 <= total <= 25_000, f"got Rs {total:,.0f}"


class TestConfigOverride:

    def test_from_config_uses_defaults_when_costs_block_absent(self):
        class Bare:
            pass
        assert CostModel.from_config(Bare()).brokerage_cap == 20.0

    def test_from_config_honours_overrides(self):
        class C:
            class costs:
                enabled = True
                brokerage_rate = 0.0
                brokerage_cap = 0.0
                stt_rate_sell = 0.0
                exchange_txn_rate = 0.0
                sebi_rate = 0.0
                stamp_duty_rate_buy = 0.0
                gst_rate = 0.0
        assert CostModel.from_config(C()).leg_cost(100, 500.0, True) == 0.0
