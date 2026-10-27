"""Measure real order round-trip latency with orders that cannot fill.

Account-holder approval required. PLACES REAL ORDERS, but only ones far from the
market, and cancels each one at once:

    python -m scripts.latency_probe                # dry run: prints the plan, places nothing
    python -m scripts.latency_probe --live         # during market hours, 09:20-15:20 IST

Per trial, on two instruments:
  - cash:   BUY 1 share of a liquid stock, LIMIT 3% below the best bid (CNC)
  - nfo:    BUY 1 lot of a far-OTM NIFTY call at Rs0.05 (NRML); a fill costs ~Rs3
it records, from this machine's clock (ns):
  place_rtt     place_order() call -> order id returned
  ack           place_order() call -> OPEN status seen on the order WebSocket
  cancel_rtt    cancel_order() call -> returned
  cancel_ack    cancel_order() call -> CANCELLED seen
plus the exchange/broker timestamps from the order updates.

Safety rails (each one is tested):
  - an order is placed only if its limit is strictly below the live best bid,
    so it cannot be marketable when sent
  - at most MAX_ORDERS per run, at most one every SPACING seconds (Zerodha
    allows 10/s, 400/min, 5,000/day; SEBI's algo-registration threshold is
    10 orders/s per exchange)
  - every order carries TAG; on ANY exception, every open TAG order is cancelled
  - if any order fills, the run stops at once and says so
  - outside 09:20-15:20 IST it refuses
  - without --live it places nothing

API orders may need a broker-registered static IP under SEBI's retail-algo
framework; a rejection is reported, never retried.

Output: data/latency/<date>_<HHMM>.csv and a printed summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("latency_probe")

TAG = "latprobe"
MAX_ORDERS = 30
SPACING = 3.0
WAIT_STATUS = 10.0          # seconds to wait for an OPEN / CANCELLED update
CASH_SYMBOL = "ITC"
CASH_DISCOUNT = 0.03
OPT_PRICE = 0.05
OPT_OTM = 0.05
OUT = Path("data/latency")
WINDOW = ((9, 20), (15, 20))


class ProbeAbort(RuntimeError):
    pass


@dataclass
class Leg:
    name: str
    exchange: str
    tradingsymbol: str
    quantity: int
    product: str
    price: float


def round_down(price: float, tick: float) -> float:
    return np.floor(price / tick + 1e-9) * tick


def cash_leg(best_bid: float, tick: float = 0.05) -> Leg:
    return Leg("cash", "NSE", CASH_SYMBOL, 1, "CNC",
               round(round_down(best_bid * (1 - CASH_DISCOUNT), tick), 2))


def check_not_marketable(leg: Leg, best_bid: Optional[float]) -> None:
    """A BUY limit must sit strictly below the best bid; otherwise abort."""
    if best_bid is None or not np.isfinite(best_bid) or best_bid <= 0:
        raise ProbeAbort(f"{leg.tradingsymbol}: no live best bid, refusing to place")
    if leg.price >= best_bid:
        raise ProbeAbort(f"{leg.tradingsymbol}: limit {leg.price} not below best bid {best_bid}")


def in_window(now: datetime) -> bool:
    (h0, m0), (h1, m1) = WINDOW
    t = (now.hour, now.minute)
    return (h0, m0) <= t < (h1, m1) and now.weekday() < 5


class Updates:
    """Order updates from the WebSocket, timestamped on arrival."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seen: Dict[str, List[tuple]] = {}

    def on_update(self, _ws, data):
        ns = time.time_ns()
        with self.lock:
            self.seen.setdefault(str(data.get("order_id")), []).append(
                (ns, data.get("status"), data.get("exchange_timestamp"),
                 data.get("order_timestamp"), data.get("status_message")))

    def first(self, order_id: str, statuses, after_ns: int) -> Optional[tuple]:
        with self.lock:
            for u in self.seen.get(str(order_id), []):
                if u[1] in statuses and u[0] >= after_ns:
                    return u
        return None


def wait_for(upd: Updates, order_id: str, statuses, after_ns: int, timeout: float,
             sleep=time.sleep) -> Optional[tuple]:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        u = upd.first(order_id, statuses, after_ns)
        if u:
            return u
        sleep(0.005)
    return None


def cancel_all_tagged(kite) -> int:
    n = 0
    for o in kite.orders():
        if o.get("tag") == TAG and o.get("status") in ("OPEN", "TRIGGER PENDING", "PUT ORDER REQ RECEIVED",
                                                         "VALIDATION PENDING", "OPEN PENDING"):
            try:
                kite.cancel_order(variety="regular", order_id=o["order_id"])
                n += 1
            except Exception as e:
                logger.error(f"cleanup cancel failed for {o['order_id']}: {e}")
    return n


def trial(kite, upd: Updates, leg: Leg, best_bid: Optional[float], clock=time.time_ns,
          wait=wait_for) -> dict:
    check_not_marketable(leg, best_bid)
    t0 = clock()
    oid = kite.place_order(variety="regular", exchange=leg.exchange,
                           tradingsymbol=leg.tradingsymbol, transaction_type="BUY",
                           quantity=leg.quantity, product=leg.product, order_type="LIMIT",
                           price=leg.price, validity="DAY", tag=TAG)
    t1 = clock()
    rec = dict(leg=leg.name, symbol=leg.tradingsymbol, price=leg.price, best_bid=best_bid,
               order_id=oid, place_rtt_ms=(t1 - t0) / 1e6)
    ack = wait(upd, oid, ("OPEN", "COMPLETE", "REJECTED"), t0, WAIT_STATUS)
    if ack is None:
        rec.update(ack_ms=np.nan, status="NO_UPDATE")
    else:
        rec.update(ack_ms=(ack[0] - t0) / 1e6, status=ack[1], exch_ts=ack[2],
                   order_ts=ack[3], message=ack[4])
    if rec["status"] == "COMPLETE":
        raise ProbeAbort(f"order {oid} FILLED ({leg.tradingsymbol} @ {leg.price}); stopping")
    if rec["status"] == "REJECTED":
        return rec
    t2 = clock()
    kite.cancel_order(variety="regular", order_id=oid)
    t3 = clock()
    rec["cancel_rtt_ms"] = (t3 - t2) / 1e6
    done = wait(upd, oid, ("CANCELLED", "COMPLETE"), t2, WAIT_STATUS)
    if done and done[1] == "COMPLETE":
        raise ProbeAbort(f"order {oid} FILLED before cancel; stopping")
    rec["cancel_ack_ms"] = (done[0] - t2) / 1e6 if done else np.nan
    return rec


def run(kite, upd: Updates, legs_fn, n_trials: int, max_orders: int = MAX_ORDERS,
        spacing: float = SPACING, sleep=time.sleep, **kw) -> pd.DataFrame:
    """legs_fn() -> [(Leg, best_bid)] refreshed every trial."""
    rows, placed = [], 0
    try:
        for i in range(n_trials):
            for leg, bid in legs_fn():
                if placed >= max_orders:
                    raise ProbeAbort(f"order cap {max_orders} reached")
                placed += 1
                r = trial(kite, upd, leg, bid, **kw)
                r["trial"] = i
                rows.append(r)
                if r.get("status") == "REJECTED":
                    raise ProbeAbort(f"order rejected: {r.get('message')}. Not retrying.")
                sleep(spacing)
    except ProbeAbort as e:
        logger.error(str(e))
        rows.append(dict(abort=str(e)))
    finally:
        n = cancel_all_tagged(kite)
        if n:
            logger.warning(f"cleanup cancelled {n} open {TAG} order(s)")
    return pd.DataFrame(rows)


# ── live wiring ─────────────────────────────────────────────────────────────

def _nifty_far_call(kite, spot: float) -> tuple:
    """A NIFTY call >= OPT_OTM above spot, on the nearest expiry at least 7
    days out: close to expiry a far call can be bid at Rs0.05 itself, and
    the not-marketable rail would (correctly) refuse a Rs0.05 order."""
    nfo = pd.DataFrame(kite.instruments("NFO"))
    c = nfo[(nfo.name == "NIFTY") & (nfo.instrument_type == "CE")]
    c = c[pd.to_datetime(c.expiry) >= pd.Timestamp.today().normalize() + pd.Timedelta(days=7)]
    first = c.expiry.min()
    c = c[(c.expiry == first) & (c.strike >= spot * (1 + OPT_OTM))].sort_values("strike")
    return c.tradingsymbol.iloc[0], int(c.lot_size.iloc[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="actually place (and cancel) orders")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--legs", default="cash,nfo")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from src.kite_env import load_dotenv as _load_dotenv
    from src.clock import ist_now
    _load_dotenv()
    from kiteconnect import KiteConnect, KiteTicker
    from src.kite_env import patch_twisted_signals
    patch_twisted_signals()
    key, tok = os.getenv("ZERODHA_API_KEY"), os.getenv("ZERODHA_ACCESS_TOKEN")
    kite = KiteConnect(api_key=key)
    kite.set_access_token(tok)
    now = ist_now().replace(tzinfo=None)
    spot = kite.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    opt_sym, opt_lot = _nifty_far_call(kite, spot)
    want = set(a.legs.split(","))

    def legs_fn():
        q = kite.quote([f"NSE:{CASH_SYMBOL}", f"NFO:{opt_sym}"])
        out = []
        if "cash" in want:
            bids = q[f"NSE:{CASH_SYMBOL}"]["depth"]["buy"]
            bid = bids[0]["price"] if bids and bids[0]["price"] > 0 else None
            out.append((cash_leg(bid or 0.0), bid))
        if "nfo" in want:
            bids = q[f"NFO:{opt_sym}"]["depth"]["buy"]
            bid = bids[0]["price"] if bids and bids[0]["price"] > 0 else None
            out.append((Leg("nfo", "NFO", opt_sym, opt_lot, "NRML", OPT_PRICE), bid))
        return out

    plan = legs_fn()
    print(f"NIFTY {spot}; plan per trial ({a.trials} trials, <= {MAX_ORDERS} orders):")
    for leg, bid in plan:
        print(f"  BUY {leg.quantity} {leg.exchange}:{leg.tradingsymbol} LIMIT {leg.price} "
              f"(best bid {bid}) product {leg.product}")
    if not a.live:
        print("dry run: nothing placed. Re-run with --live during 09:20-15:20 IST.")
        return 0
    if not in_window(now):
        raise SystemExit(f"refusing: {now:%a %H:%M} IST is outside 09:20-15:20 on a weekday")

    upd = Updates()
    ws = KiteTicker(key, tok)
    ws.on_order_update = upd.on_update
    ws.connect(threaded=True)
    time.sleep(3)
    df = run(kite, upd, legs_fn, a.trials)
    try:
        ws.close()
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{now:%Y-%m-%d_%H%M}.csv"
    df.to_csv(path, index=False)
    ok = df.dropna(subset=["place_rtt_ms"]) if "place_rtt_ms" in df else df.iloc[0:0]
    print(f"\n{len(ok)} orders measured -> {path}")
    for col in ("place_rtt_ms", "ack_ms", "cancel_rtt_ms", "cancel_ack_ms"):
        if col in ok and ok[col].notna().any():
            for leg, g in ok.groupby("leg"):
                x = g[col].dropna()
                print(f"  {leg:4s} {col:14s} median {x.median():7.1f} ms   p90 {x.quantile(.9):7.1f}"
                      f"   max {x.max():7.1f}   n {len(x)}")
    if "abort" in df and df["abort"].notna().any():
        print("ABORTED:", df["abort"].dropna().iloc[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
