"""
Transaction Cost Model
======================
Per-leg Zerodha charges for NSE equity INTRADAY (MIS/product-squared-off).

WHY THIS EXISTS
---------------
Every number the bot reported — journal PnL, equity curve, session report,
the daily-loss circuit breaker, and every gate's exp_R — was GROSS. Measured
on the Jul 9-31 2026 tape:

    gross P&L, 168 completed trades, 17 sessions   Rs  -5,463
    charges (377 legs)                             Rs -22,158
    NET                                            Rs -27,621   (-11.0% of equity)

Friction is ~Rs 130 per round trip = **0.15R** at the median Rs 855 of risk
per trade. That is not a rounding error against a book whose gross edge is
~zero: at 168 trades/month the charges ARE the entire loss. Reporting gross
made a -11% month look roughly flat.

Note the leg count: 377 legs for 168 trades, not 336. The 50% partial at 1R
adds a third leg to every scaled-out trade, and each partial pays its own
brokerage + STT. A "per round trip" estimate that assumes two legs
under-counts by ~12% here.

SCOPE
-----
This module is OBSERVATIONAL. It annotates the journal and the session
report; it deliberately does NOT debit ``_current_balance``, because balance
feeds position sizing and the daily-loss halt — changing those is a
behaviour change that belongs in its own decision, not a reporting fix.

RATE SOURCE
-----------
Zerodha equity-intraday slabs (https://zerodha.com/charges/), as of 2026-08.
All rates are config-overridable under ``costs:`` so a plan change (or a
different broker) is a YAML edit, not a code change. Rates are applied to
turnover = quantity x price for the leg.

    brokerage    0.03% of turnover, capped at Rs 20 per order
    STT          0.025% on the SELL leg only (intraday; delivery and futures
                 differ — see `scripts/_vehicles.py`, which builds those
                 schedules from this same model rather than restating rates)
    exchange     0.00297% (NSE) both legs
    SEBI         0.0001% both legs  (Rs 10 per crore)
    stamp duty   0.003% on the BUY leg only
    GST          18% on (brokerage + exchange + SEBI)

Slippage is NOT included — it is not a charge and is not knowable from the
journal. `estimate_slippage` is provided separately for scenario work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Broker charge schedule. Rates are fractions of turnover, not percents."""

    brokerage_rate: float = 0.0003          # 0.03%
    brokerage_cap: float = 20.0             # Rs per order
    stt_rate_sell: float = 0.00025          # 0.025%, sell leg only
    #: STT on the BUY leg. ZERO for intraday equity, which is why this field
    #: did not exist until the overnight question was asked: MIS pays STT once.
    #: DELIVERY pays it on BOTH legs at 0.1%, and that 20 bps is the single
    #: number that decides whether an overnight strategy can be run in cash.
    #: Defaulting to 0.0 leaves every existing caller byte-identical.
    stt_rate_buy: float = 0.0
    exchange_txn_rate: float = 0.0000297    # 0.00297% NSE
    sebi_rate: float = 0.000001             # Rs 10 per crore
    stamp_duty_rate_buy: float = 0.00003    # 0.003%, buy leg only
    gst_rate: float = 0.18                  # on brokerage + exchange + SEBI
    enabled: bool = True

    def leg_cost(self, quantity: float, price: float, is_sell: bool) -> float:
        """Total charges for ONE executed leg.

        Args:
            quantity: shares traded on this leg
            price: fill price
            is_sell: True when this leg SELLS (STT applies), False when it
                BUYS (stamp duty applies). For a short trade the ENTRY is the
                sell leg and the exit is the buy leg — pass the leg's own
                direction, not the position's side.
        """
        if not self.enabled:
            return 0.0
        turnover = abs(float(quantity)) * abs(float(price))
        if turnover <= 0:
            return 0.0
        brokerage = min(self.brokerage_rate * turnover, self.brokerage_cap)
        stt = (self.stt_rate_sell if is_sell else self.stt_rate_buy) * turnover
        exchange = self.exchange_txn_rate * turnover
        sebi = self.sebi_rate * turnover
        stamp = 0.0 if is_sell else self.stamp_duty_rate_buy * turnover
        gst = self.gst_rate * (brokerage + exchange + sebi)
        return brokerage + stt + exchange + sebi + stamp + gst

    def entry_cost(self, side: str, quantity: float, price: float) -> float:
        """Charges on the entry leg of a position. `side` is BUY or SELL."""
        return self.leg_cost(quantity, price, is_sell=(side or "").upper() == "SELL")

    def exit_cost(self, side: str, quantity: float, price: float) -> float:
        """Charges on an exit leg (full or partial). `side` is the POSITION's
        side, so the leg itself trades the opposite way."""
        return self.leg_cost(quantity, price, is_sell=(side or "").upper() == "BUY")

    def round_trip_cost(self, side: str, quantity: float,
                        entry_price: float, exit_price: float) -> float:
        """Entry + single exit. Partial exits add legs — sum them separately."""
        return (self.entry_cost(side, quantity, entry_price)
                + self.exit_cost(side, quantity, exit_price))

    @staticmethod
    def estimate_slippage(quantity: float, price: float, bps: float) -> float:
        """Scenario helper — NOT applied anywhere automatically."""
        return abs(float(quantity)) * abs(float(price)) * (bps / 10000.0)

    @classmethod
    def from_config(cls, config) -> "CostModel":
        """Build from AppConfig.costs, tolerating older configs with no
        ``costs:`` block (defaults are the shipped Zerodha slabs)."""
        c = getattr(config, "costs", None)
        if c is None:
            return cls()
        return cls(
            brokerage_rate=getattr(c, "brokerage_rate", cls.brokerage_rate),
            brokerage_cap=getattr(c, "brokerage_cap", cls.brokerage_cap),
            stt_rate_sell=getattr(c, "stt_rate_sell", cls.stt_rate_sell),
            stt_rate_buy=getattr(c, "stt_rate_buy", cls.stt_rate_buy),
            exchange_txn_rate=getattr(c, "exchange_txn_rate", cls.exchange_txn_rate),
            sebi_rate=getattr(c, "sebi_rate", cls.sebi_rate),
            stamp_duty_rate_buy=getattr(c, "stamp_duty_rate_buy", cls.stamp_duty_rate_buy),
            gst_rate=getattr(c, "gst_rate", cls.gst_rate),
            enabled=getattr(c, "enabled", True),
        )
