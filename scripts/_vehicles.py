"""
What a round trip costs, by instrument — the arithmetic the overnight question
turns on.

WHY THIS EXISTS
---------------
`structural_edge.py` measured where in the day the drift lives and found the
only positive leg is overnight. The obvious follow-up — "then only trade
overnight" — is not a strategy question, it is a charge-schedule question,
because the bot's whole problem is the identity

    rupees = notional x (gross_bps - charge_bps)

and the charge term is NOT 5.94 bps once you stop being intraday. MIS pays
Securities Transaction Tax on the sell leg only. Hold a cash position
overnight and it becomes a DELIVERY trade, which pays STT of 0.1% on BOTH
legs: 20 bps before anything else. That one line is larger than the entire
measured overnight drift on the survivorship-free control, and larger than the
intraday leg the strategy would be avoiding.

So the escape, if there is one, is not cash. It is futures, where STT is 0.02%
on the sell leg alone. And stock futures exist only on the F&O-eligible list —
roughly the largest couple of hundred names — which is exactly the segment
where `structural_edge.py`'s own size split found the SMALLEST overnight drift.
That tension is the finding; this module makes it arithmetic instead of
argument.

RATES
-----
Built from `src.costs.CostModel` rather than restating percentages, so there is
one charge schedule in this codebase and not two. Zerodha slabs as of 2026-08.

Two rates here are less certain than the rest and are called out in the table
rather than buried: equity-delivery brokerage (Zerodha has changed it), and the
exact F&O exchange-transaction rate. Both are worth under 0.4 bps on a Rs225K
position. STT dominates every row by an order of magnitude, so no conclusion in
this module moves if either is slightly wrong — which is the only reason it is
acceptable to publish the table at all.

BTST is billed BOTH ways in the wild: some brokers treat the next-day sale of
undelivered stock as delivery (STT 0.1% both legs), others as intraday on the
sell (0.025%). Both bounds are reported. Zerodha bills the delivery way.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.costs import CostModel  # noqa: E402

__all__ = ["Vehicle", "VEHICLES", "vehicle_table", "LOT_NOTIONAL",
           "POSITION_NOTIONAL"]

#: The bot's own position size: Rs75K x 3x MIS margin.
POSITION_NOTIONAL = 225_000.0

#: A realistic stock-futures lot. Lot sizes are set per symbol so this is an
#: order of magnitude, not a quote — it is here to show that the brokerage cap
#: makes the per-bps cost FALL as the position grows, which is the one place
#: size helps.
LOT_NOTIONAL = 1_000_000.0


@dataclass(frozen=True)
class Vehicle:
    """One instrument's round-trip cost, and where it can be traded."""

    name: str
    model: CostModel
    notional: float
    available_on: str
    holds_overnight: bool
    #: Which measured drift this vehicle can actually capture. An index future
    #: tracks the INDEX, so netting it against the basket's +18 bps would
    #: promise a return no index future can deliver — the single most
    #: inviting mistake this whole table could produce.
    tracks: str = "basket"
    caveat: str = ""

    @property
    def round_trip_rupees(self) -> float:
        """Entry + one exit. No partial ladder: an overnight gap trade is one
        in and one out, unlike the intraday book whose 50% partial adds a
        third leg (and ~1 bp) to every scaled trade."""
        return self.model.round_trip_cost("BUY", 1.0, self.notional, self.notional)

    @property
    def round_trip_bps(self) -> float:
        return self.round_trip_rupees / self.notional * 1e4

    def net_bps(self, gross_bps: float, spread_bps: float = 0.0) -> float:
        """Gross minus broker charges minus the spread actually crossed.

        `spread_bps` defaults to 0 only so existing callers keep working; a
        zero spread is a MID-TO-MID fiction, never a trade. Entering at 15:25
        and exiting at 09:16 crosses the spread twice at half each time, so
        one full spread is the right deduction.
        """
        return gross_bps - self.round_trip_bps - max(spread_bps, 0.0)


#: Equity intraday. The bot's current vehicle. Cannot hold overnight at all —
#: MIS is auto-squared-off, so this row exists only as the baseline the
#: measured 5.94 bps reconciles against.
MIS_EQUITY = Vehicle(
    name="equity MIS (intraday)",
    model=CostModel(),
    notional=POSITION_NOTIONAL,
    available_on="every symbol",
    holds_overnight=False,
    caveat="auto-squared-off; cannot hold a gap",
)

#: Equity delivery. What a cash overnight position actually is.
CNC_DELIVERY = Vehicle(
    name="equity delivery / BTST",
    # brokerage_rate=0: Zerodha's published equity-delivery brokerage is Rs0.
    # The first version kept the intraday Rs20 cap and printed 24.32 bps while
    # the analysis quoted 22.2 — two numbers for one cost. No conclusion
    # depends on the 2.1 bps difference; the STT does all the work.
    model=CostModel(brokerage_rate=0.0, stt_rate_sell=0.001, stt_rate_buy=0.001,
                    stamp_duty_rate_buy=0.00015),
    notional=POSITION_NOTIONAL,
    available_on="every symbol",
    holds_overnight=True,
    caveat="STT 0.1% on BOTH legs = 20 bps before anything else",
)

#: The favourable reading of BTST, where the sell leg is billed as intraday.
#: Reported so the conclusion does not rest on which way a broker bills it.
BTST_INTRADAY_SELL = Vehicle(
    name="BTST, sell billed intraday",
    model=CostModel(brokerage_rate=0.0, stt_rate_sell=0.00025,
                    stt_rate_buy=0.001, stamp_duty_rate_buy=0.00015),
    notional=POSITION_NOTIONAL,
    available_on="every symbol",
    holds_overnight=True,
    caveat="the best case; Zerodha does not bill it this way",
)

#: Stock futures, NRML. STT 0.02% sell-side only.
_FUT = CostModel(stt_rate_sell=0.0002, stt_rate_buy=0.0,
                 exchange_txn_rate=0.000019, stamp_duty_rate_buy=0.00002)

STOCK_FUTURES_225K = Vehicle(
    name="stock futures @ Rs225K",
    model=_FUT,
    notional=POSITION_NOTIONAL,
    available_on="F&O-eligible names only (~200)",
    holds_overnight=True,
    caveat="the Rs20 brokerage cap is 1.8 bps at this size",
)

STOCK_FUTURES_LOT = Vehicle(
    name="stock futures @ Rs10L lot",
    model=_FUT,
    notional=LOT_NOTIONAL,
    available_on="F&O-eligible names only (~200)",
    holds_overnight=True,
    caveat="brokerage cap amortised; ~4x the bot's per-trade risk",
)

INDEX_FUTURES = Vehicle(
    name="index futures @ Rs18L lot",
    model=_FUT,
    notional=1_800_000.0,
    available_on="NIFTY / BANKNIFTY only",
    holds_overnight=True,
    tracks="index",
    caveat="tracks the INDEX's drift, not the basket's",
)

VEHICLES: List[Vehicle] = [
    MIS_EQUITY, CNC_DELIVERY, BTST_INTRADAY_SELL,
    STOCK_FUTURES_225K, STOCK_FUTURES_LOT, INDEX_FUTURES,
]


def vehicle_table(gross_bps: Optional[float] = None,
                  index_gross_bps: Optional[float] = None,
                  vehicles: Optional[List[Vehicle]] = None,
                  spread_bps: float = 0.0,
                  gross_label: str = "") -> pd.DataFrame:
    """Round-trip cost per vehicle, netted against a gross drift.

    `gross_bps` is a MEASURED drift. `index_gross_bps` is the same leg on the
    index, and index vehicles net against it rather than the basket, since an
    index future cannot capture a basket-specific premium. Vehicles that
    cannot be held overnight get no net at all.

    WHICH GROSS YOU PASS IS THE WHOLE ANSWER, and an earlier version of this
    table got it wrong in a way that read as good news for weeks. Passing the
    OPEN-PRINT overnight mean of the survivorship-affected basket produced
    "stock futures ... +14.88 ... pays", and every one of the three
    measurements that followed cut it:

        basket, open print, no spread          +15.01
          - futures round trip                  -3.14  -> +11.87  (as printed)
          - first-minute cliff + decay          -5.45  -> +6.42
          - selection (basket vs its index)     -2.47  -> +3.95
          - effective spread, LOWER bound       -2.98  -> +0.97

    So pass the REACHABLE plateau, not the open print; the INDEX, not the
    basket; and a real `spread_bps`. `gross_label` is printed in the result's
    attrs so a reader can always see which of those they are looking at.
    """
    rows = []
    for v in (vehicles or VEHICLES):
        r = {
            "vehicle": v.name,
            "Rs/round trip": round(v.round_trip_rupees, 1),
            "bps": round(v.round_trip_bps, 2),
            "overnight?": "yes" if v.holds_overnight else "NO",
            "available on": v.available_on,
        }
        gross = index_gross_bps if v.tracks == "index" else gross_bps
        if gross_bps is not None or index_gross_bps is not None:
            # A vehicle that cannot be held overnight has no overnight net,
            # and printing one invites reading "+12 bps" off the row for the
            # instrument that is auto-squared-off before the gap happens.
            # Likewise an index row with no index drift supplied: blank, not
            # the basket's number.
            if not v.holds_overnight or gross is None:
                r["net bps"], r["verdict"] = float("nan"), "n/a"
            else:
                n = v.net_bps(gross, spread_bps)
                r["net bps"] = round(n, 2)
                r["verdict"] = "pays" if n > 0 else "loses"
        r["caveat"] = v.caveat
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs["gross_label"] = gross_label or (
        "UNLABELLED GROSS — say which leg this is")
    out.attrs["spread_bps"] = spread_bps
    return out
