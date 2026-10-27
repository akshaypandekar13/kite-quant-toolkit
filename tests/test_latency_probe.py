"""scripts/latency_probe.py safety rails, with a fake Kite (no network, no orders)."""

from datetime import datetime

import pytest

from scripts import latency_probe as lp


class FakeKite:
    def __init__(self, upd, ack="OPEN", fail_on=None):
        self.upd, self.ack, self.fail_on = upd, ack, fail_on
        self.placed, self.cancelled, self.book = [], [], {}

    def place_order(self, **kw):
        if self.fail_on is not None and len(self.placed) == self.fail_on:
            raise RuntimeError("network down")
        oid = str(len(self.placed) + 1)
        self.placed.append(kw)
        self.book[oid] = dict(order_id=oid, tag=kw["tag"], status=self.ack)
        self.upd.on_update(None, dict(order_id=oid, status=self.ack))
        return oid

    def cancel_order(self, variety, order_id):
        self.cancelled.append(order_id)
        self.book[order_id]["status"] = "CANCELLED"
        self.upd.on_update(None, dict(order_id=order_id, status="CANCELLED"))

    def orders(self):
        return list(self.book.values())


def fast_wait(upd, oid, statuses, after, timeout):
    return upd.first(oid, statuses, 0)


def legs(bid=400.0, price=None):
    leg = lp.cash_leg(bid) if price is None else lp.Leg("cash", "NSE", "ITC", 1, "CNC", price)
    return lambda: [(leg, bid)]


def go(kite, upd, legs_fn, n=3, **kw):
    return lp.run(kite, upd, legs_fn, n, sleep=lambda s: None, wait=fast_wait, **kw)


def test_normal_trial_places_then_cancels_with_tag_and_measures():
    upd = lp.Updates()
    k = FakeKite(upd)
    ticks = iter(range(0, 10**10, 10**6))                # 1 ms per clock call
    df = go(k, upd, legs(), n=2, clock=lambda: next(ticks))
    assert len(k.placed) == 2 and k.cancelled == ["1", "2"]
    assert all(p["tag"] == lp.TAG and p["transaction_type"] == "BUY" for p in k.placed)
    assert df.place_rtt_ms.tolist() == [1.0, 1.0] and df.cancel_rtt_ms.tolist() == [1.0, 1.0]


def test_cash_limit_is_below_the_bid_on_the_tick_grid():
    leg = lp.cash_leg(401.37)
    assert leg.price < 401.37 and leg.price == pytest.approx(389.30)


def test_marketable_order_is_never_sent():
    upd = lp.Updates()
    k = FakeKite(upd)
    df = go(k, upd, legs(bid=400.0, price=400.0))
    assert k.placed == [] and "not below best bid" in df["abort"].dropna().iloc[0]
    df = go(k, upd, lambda: [(lp.cash_leg(400.0), None)])
    assert k.placed == [] and "no live best bid" in df["abort"].dropna().iloc[0]


def test_order_cap():
    upd = lp.Updates()
    k = FakeKite(upd)
    two = lambda: [(lp.cash_leg(400.0), 400.0), (lp.cash_leg(400.0), 400.0)]
    df = go(k, upd, two, n=10, max_orders=3)
    assert len(k.placed) == 3 and "cap" in df["abort"].dropna().iloc[0]


def test_a_fill_stops_the_run():
    upd = lp.Updates()
    k = FakeKite(upd, ack="COMPLETE")
    df = go(k, upd, legs(), n=5)
    assert len(k.placed) == 1 and "FILLED" in df["abort"].dropna().iloc[0]
    assert k.cancelled == []                             # never cancels a filled order


def test_rejection_stops_without_retry():
    upd = lp.Updates()
    k = FakeKite(upd, ack="REJECTED")
    df = go(k, upd, legs(), n=5)
    assert len(k.placed) == 1 and "rejected" in df["abort"].dropna().iloc[0]


def test_any_error_cancels_every_open_tagged_order():
    upd = lp.Updates()
    k = FakeKite(upd, fail_on=1)                         # 2nd placement raises
    lost = []
    k.cancel_order = lambda variety, order_id: lost.append(order_id)   # cancels never land
    with pytest.raises(RuntimeError):
        lp.run(k, upd, legs(), 3, sleep=lambda s: None, wait=fast_wait)
    # order 1 stayed OPEN: the trial's cancel, then the cleanup's, both sent
    assert lost == ["1", "1"]


def test_window():
    assert lp.in_window(datetime(2026, 10, 12, 9, 20))
    assert not lp.in_window(datetime(2026, 10, 12, 9, 19))
    assert not lp.in_window(datetime(2026, 10, 12, 15, 20))
    assert not lp.in_window(datetime(2026, 10, 10, 11, 0))   # Saturday
