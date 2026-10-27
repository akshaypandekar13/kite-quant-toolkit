"""Event-driven market-making simulator (Route B, gate G2). Research only.

Answers one question for a quoting strategy: after queue position, latency,
adverse selection and every statutory charge, is there money left?

    python -m scripts.mm_sim --day data/depth/2026-10-12 --symbol NIFTY --kind fut

THE MODEL

Time is exchange time. Three latencies separate us from the exchange:
  feed_lat   exchange event -> our strategy sees it
  order_lat  our decision   -> the order (or cancel) reaches the exchange
  (fills reach the strategy feed_lat after they happen)
Everything the strategy knows is stale by feed_lat; everything it does lands
order_lat later. A cancel racing a fill can lose, as in reality.

Fills for a resting order, `fill_model`:
  "through" (default, most conservative): filled only when a trade prints
            STRICTLY through our price. Queue position never helps us.
  "queue":  we join the back of the displayed queue at our price. The queue
            ahead shrinks only by trades at our price (cancellations are
            assumed to be BEHIND us, the pessimistic choice); once it is used
            up, further trades at our price fill us. A trade through our price
            fills us in full.
An order that is marketable on arrival takes liquidity at the displayed
touch, walking the displayed levels; any remainder rests.

With 5-level snapshots (Route A) the aggressor of a trade is unknown;
`events_from_depth` classifies it by the Lee-Ready rule against the mid
BEFORE the trade. With exchange tick-by-tick data the side is known and the
same simulator applies.

Charges per fill come from a `src.costs.CostModel`. MEMBER_FUT is a member
desk on stock/index futures: zero brokerage, STT on sells, exchange, SEBI,
stamp, GST (rates VERIFY; same futures schedule as scripts/_vehicles.py).

Reported: gross and net P&L, fills, spread captured (vs the mid at the fill),
adverse selection (the mid's move against us h seconds after each fill),
inventory, messages and order-to-trade ratio, max drawdown, and the cost of
flattening at the end.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import CostModel  # noqa: E402

MEMBER_FUT = CostModel(brokerage_rate=0.0, brokerage_cap=0.0, stt_rate_sell=0.0002,
                       exchange_txn_rate=0.000019, stamp_duty_rate_buy=0.00002)
ADVERSE_HORIZONS = (1.0, 5.0, 30.0)


# ── events ──────────────────────────────────────────────────────────────────

@dataclass
class Book:
    t: float
    bids: List[Tuple[float, float]]          # [(price, qty)] best first
    asks: List[Tuple[float, float]]

    @property
    def bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        return (self.bid + self.ask) / 2 if self.bids and self.asks else None

    def qty_at(self, side: str, price: float) -> float:
        for p, q in (self.bids if side == "BUY" else self.asks):
            if abs(p - price) < 1e-9:
                return q
        return 0.0


@dataclass
class Trade:
    t: float
    price: float
    qty: float
    aggressor: str                            # "BUY" lifts asks, "SELL" hits bids


@dataclass
class Order:
    oid: int
    side: str                                 # BUY / SELL
    price: float
    qty: float
    sent: float
    arrived: Optional[float] = None
    ahead: float = 0.0
    filled: float = 0.0
    live: bool = False
    cancelled: bool = False

    @property
    def open_qty(self) -> float:
        return self.qty - self.filled


@dataclass
class Fill:
    t: float
    oid: int
    side: str
    price: float
    qty: float
    maker: bool
    mid: Optional[float]
    fee: float


# ── strategy interface ──────────────────────────────────────────────────────

@dataclass
class View:
    """What the strategy may use: the book as seen (stale by feed_lat), the
    position as known from fills it has been told about, and its own orders."""
    t: float
    book: Book
    position: float
    orders: Dict[int, Order]


class SymmetricQuoter:
    """Baseline: quote `half_ticks` either side of mid, skewed against
    inventory, one level each side, within a position limit. Joins the touch
    when the spread is one tick. Not a claim of edge: the reference strategy
    a real one must beat."""

    def __init__(self, tick: float, size: float, max_pos: float, half_ticks: int = 1,
                 skew_ticks_per_unit: float = 0.0, min_spread_ticks: int = 1):
        self.tick, self.size, self.max_pos = tick, size, max_pos
        self.half, self.skew, self.min_spread = half_ticks, skew_ticks_per_unit, min_spread_ticks

    def quotes(self, v: View) -> Dict[str, Optional[Tuple[float, float]]]:
        b = v.book
        if b.bid is None or b.ask is None or b.ask - b.bid < self.min_spread * self.tick - 1e-9:
            return {"BUY": None, "SELL": None}
        shift = -self.skew * (v.position / self.size) * self.tick
        bid = min(b.bid, _floor(b.mid - self.half * self.tick + shift, self.tick))
        ask = max(b.ask, _ceil(b.mid + self.half * self.tick + shift, self.tick))
        out = {"BUY": (bid, self.size), "SELL": (ask, self.size)}
        if v.position + self.size > self.max_pos:
            out["BUY"] = None
        if v.position - self.size < -self.max_pos:
            out["SELL"] = None
        return out


def _floor(x, tick):
    return round(np.floor(x / tick + 1e-9) * tick, 10)


def _ceil(x, tick):
    return round(np.ceil(x / tick - 1e-9) * tick, 10)


# ── simulator ───────────────────────────────────────────────────────────────

@dataclass
class Result:
    fills: List[Fill]
    messages: int
    gross: float
    fees: float
    flatten_cost: float
    net: float
    position_path: List[Tuple[float, float]]
    mids: pd.Series

    def summary(self) -> dict:
        f = pd.DataFrame([x.__dict__ for x in self.fills])
        out = dict(fills=len(f), messages=self.messages,
                   otr=self.messages / max(len(f), 1),
                   gross=self.gross, fees=self.fees, flatten_cost=self.flatten_cost,
                   net=self.net)
        if len(f):
            sgn = np.where(f.side == "BUY", 1.0, -1.0)
            out["spread_captured"] = float(np.nansum(sgn * (f["mid"] - f.price) * f.qty))
            out["maker_share"] = float(f.maker.mean())
            for h in ADVERSE_HORIZONS:
                m = adverse(self.mids, f, h)
                out[f"adverse_{h:g}s"] = float(np.nansum(m * f.qty))
        pos = np.array([p for _, p in self.position_path]) if self.position_path else np.zeros(1)
        out["max_abs_position"] = float(np.abs(pos).max())
        return out


def adverse(mids: pd.Series, fills: pd.DataFrame, h: float) -> np.ndarray:
    """Per unit: how far the mid moved AGAINST each fill h seconds later
    (positive = adverse). Uses the last mid at or before t + h."""
    idx = mids.index.to_numpy()
    j = np.searchsorted(idx, fills.t.to_numpy() + h, side="right") - 1
    later = np.where(j >= 0, mids.to_numpy()[np.clip(j, 0, None)], np.nan)
    sgn = np.where(fills.side == "BUY", 1.0, -1.0)
    return -sgn * (later - fills["mid"].to_numpy())


class Simulator:
    def __init__(self, strategy, feed_lat: float = 0.05, order_lat: float = 0.1,
                 fill_model: str = "through", costs: CostModel = MEMBER_FUT,
                 requote_min: float = 0.0, flatten: bool = True):
        assert fill_model in ("through", "queue")
        self.s = strategy
        self.feed_lat, self.order_lat = feed_lat, order_lat
        self.fill_model, self.costs = fill_model, costs
        self.requote_min, self.flatten = requote_min, flatten

    def run(self, events: List) -> Result:
        heap: list = []
        seq = itertools.count()
        push = lambda t, kind, x: heapq.heappush(heap, (t, next(seq), kind, x))
        for e in events:
            push(e.t, "mkt", e)

        book: Optional[Book] = None
        orders: Dict[int, Order] = {}
        oid = itertools.count(1)
        fills: List[Fill] = []
        pos = 0.0          # true position at the exchange
        known_pos = 0.0    # what the strategy has been told
        cash = 0.0
        fees = 0.0
        msgs = 0
        mids: Dict[float, float] = {}
        path: List[Tuple[float, float]] = []
        last_quote = -np.inf

        def fill(o: Order, t: float, price: float, qty: float, maker: bool):
            nonlocal pos, cash, fees
            if qty <= 0:
                return
            o.filled += qty
            sgn = 1.0 if o.side == "BUY" else -1.0
            pos += sgn * qty
            cash -= sgn * price * qty
            fee = self.costs.leg_cost(qty, price, is_sell=o.side == "SELL")
            fees += fee
            fills.append(Fill(t, o.oid, o.side, price, qty, maker,
                              book.mid if book else None, fee))
            path.append((t, pos))
            push(t + self.feed_lat, "fill_seen", sgn * qty)
            if o.open_qty <= 1e-12:
                o.live = False

        def take(o: Order, t: float):
            """Marketable on arrival: walk the displayed opposite levels."""
            levels = book.asks if o.side == "BUY" else book.bids
            for p, q in levels:
                if o.open_qty <= 0:
                    break
                ok = p <= o.price + 1e-9 if o.side == "BUY" else p >= o.price - 1e-9
                if not ok:
                    break
                fill(o, t, p, min(q, o.open_qty), maker=False)

        while heap:
            t, _, kind, x = heapq.heappop(heap)
            if kind == "mkt" and isinstance(x, Book):
                book = x
                if book.mid is not None:
                    mids[t] = book.mid
                # The queue ahead of us cannot exceed what is displayed at our
                # price: if it shrank below, some of it was cancelled.
                for o in orders.values():
                    if o.live and o.open_qty > 0:
                        o.ahead = min(o.ahead, book.qty_at(o.side, o.price))
                push(t + self.feed_lat, "see", x)
            elif kind == "mkt" and isinstance(x, Trade):
                for o in list(orders.values()):
                    if not o.live or o.open_qty <= 0:
                        continue
                    hits = (o.side == "BUY" and x.aggressor == "SELL") or \
                           (o.side == "SELL" and x.aggressor == "BUY")
                    through = (o.side == "BUY" and x.price < o.price - 1e-9) or \
                              (o.side == "SELL" and x.price > o.price + 1e-9)
                    at = abs(x.price - o.price) < 1e-9
                    if through:
                        fill(o, t, o.price, o.open_qty, maker=True)
                    elif self.fill_model == "queue" and at and hits:
                        rem = x.qty - o.ahead
                        o.ahead = max(o.ahead - x.qty, 0.0)
                        if rem > 0:
                            fill(o, t, o.price, min(rem, o.open_qty), maker=True)
            elif kind == "see":
                v = View(t, x, known_pos, {k: o for k, o in orders.items() if o.live or o.arrived is None})
                if t - last_quote < self.requote_min:
                    continue
                want = self.s.quotes(v)
                for side in ("BUY", "SELL"):
                    mine = [o for o in orders.values() if o.side == side and not o.cancelled
                            and (o.live or o.arrived is None)]
                    target = want.get(side)
                    keep = [o for o in mine if target and abs(o.price - target[0]) < 1e-9]
                    for o in mine:
                        if o not in keep:
                            o.cancelled = True
                            msgs += 1
                            push(t + self.order_lat, "cancel", o)
                    if target and not keep:
                        o = Order(next(oid), side, target[0], target[1], sent=t)
                        orders[o.oid] = o
                        msgs += 1
                        push(t + self.order_lat, "arrive", o)
                last_quote = t
            elif kind == "arrive":
                o = x
                o.arrived = t
                # Live even if its cancel is already on the way: until that
                # cancel lands the order rests at the exchange and can fill.
                o.live = True
                if book is not None:
                    opp = book.ask if o.side == "BUY" else book.bid
                    marketable = opp is not None and (
                        (o.side == "BUY" and o.price >= opp - 1e-9) or
                        (o.side == "SELL" and o.price <= opp + 1e-9))
                    if marketable:
                        take(o, t)
                    o.ahead = book.qty_at(o.side, o.price)
            elif kind == "cancel":
                x.live = False           # a fill that landed first is already booked
            elif kind == "fill_seen":
                known_pos += x

        flatten_cost = 0.0
        if self.flatten and abs(pos) > 1e-12 and book and book.bids and book.asks:
            px = book.bid if pos > 0 else book.ask        # cross the spread to exit
            flatten_cost = abs(pos) * abs(book.mid - px) + self.costs.leg_cost(abs(pos), px, pos > 0)
            cash += pos * px
            fees += self.costs.leg_cost(abs(pos), px, pos > 0)
            pos = 0.0
            gross = cash
        else:
            gross = cash + (pos * book.mid if book and book.mid is not None else 0.0)
        return Result(fills, msgs, gross, fees, flatten_cost, gross - fees, path,
                      pd.Series(mids, dtype=float).sort_index())


# ── loader: Route A depth recordings -> events ──────────────────────────────

def events_from_depth(rows: pd.DataFrame, levels: int = 5) -> List:
    """One instrument's recorder rows (sorted by exch_ts) -> Book and Trade
    events. A volume increase between snapshots is a trade of that size at
    `ltp`, classified by Lee-Ready against the PREVIOUS snapshot's mid
    (above: buyer-initiated; below: seller; at the mid: the tick rule)."""
    rows = rows.sort_values(["exch_ts", "recv_ns"])
    out: List = []
    prev_vol, prev_mid, prev_px = None, None, None
    for r in rows.itertuples(index=False):
        t = float(r.exch_ts)
        vol = r.volume
        if prev_vol is not None and pd.notna(vol) and vol > prev_vol and pd.notna(r.ltp):
            if prev_mid is not None and r.ltp > prev_mid + 1e-9:
                side = "BUY"
            elif prev_mid is not None and r.ltp < prev_mid - 1e-9:
                side = "SELL"
            else:
                side = "BUY" if prev_px is None or r.ltp >= prev_px else "SELL"
            out.append(Trade(t, float(r.ltp), float(vol - prev_vol), side))
            prev_px = float(r.ltp)
        if pd.notna(vol):
            prev_vol = vol
        bids = [(getattr(r, f"b{i}p"), getattr(r, f"b{i}q")) for i in range(1, levels + 1)]
        asks = [(getattr(r, f"a{i}p"), getattr(r, f"a{i}q")) for i in range(1, levels + 1)]
        bids = [(float(p), float(q)) for p, q in bids if pd.notna(p) and p > 0 and q > 0]
        asks = [(float(p), float(q)) for p, q in asks if pd.notna(p) and p > 0 and q > 0]
        b = Book(t, bids, asks)
        out.append(b)
        if b.mid is not None:
            prev_mid = b.mid
    return out


def load_day(day_dir: Path, symbol: str, kind: str) -> Tuple[List, dict]:
    inst = pd.read_csv(day_dir / "instruments.csv")
    m = inst[(inst.symbol == symbol) & (inst.kind == kind)]
    if m.empty:
        raise SystemExit(f"{symbol}/{kind} not in {day_dir}/instruments.csv")
    meta = m.iloc[0].to_dict()
    parts = [pd.read_csv(f) for f in sorted(day_dir.glob("ticks_*.csv.gz"))]
    ticks = pd.concat(parts, ignore_index=True)
    rows = ticks[ticks.token == meta["token"]].dropna(subset=["exch_ts"])
    return events_from_depth(rows), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True, help="data/depth/<date>")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--kind", default="fut", choices=["fut", "cash", "opt"])
    ap.add_argument("--feed-lat", type=float, default=0.05)
    ap.add_argument("--order-lat", type=float, default=0.10)
    ap.add_argument("--fill-model", default="through", choices=["through", "queue"])
    ap.add_argument("--lots", type=float, default=1)
    ap.add_argument("--max-lots", type=float, default=3)
    a = ap.parse_args(argv)
    events, meta = load_day(Path(a.day), a.symbol, a.kind)
    size = a.lots * float(meta.get("lot") or 1)
    strat = SymmetricQuoter(tick=float(meta.get("tick") or 0.05), size=size,
                            max_pos=a.max_lots * float(meta.get("lot") or 1))
    res = Simulator(strat, a.feed_lat, a.order_lat, a.fill_model).run(events)
    for k, v in res.summary().items():
        print(f"{k:18s} {v:,.2f}" if isinstance(v, float) else f"{k:18s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
