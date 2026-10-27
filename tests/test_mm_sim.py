"""scripts/mm_sim.py on hand-computed event streams."""

import numpy as np
import pandas as pd
import pytest

from scripts import mm_sim as mm
from src.costs import CostModel

FREE = CostModel(enabled=False)


class Script:
    """Strategy that returns quotes from a function of the view."""

    def __init__(self, fn):
        self.fn, self.views = fn, []

    def quotes(self, v):
        self.views.append(v)
        return self.fn(v)


def book(t, bid=100.0, ask=101.0, bq=10.0, aq=10.0, depth=None):
    if depth:
        return mm.Book(t, *depth)
    return mm.Book(t, [(bid, bq)], [(ask, aq)])


def bid_only(price, qty=1.0):
    return Script(lambda v: {"BUY": (price, qty), "SELL": None})


def bid_once(price, qty=1.0):
    """Bid until the strategy KNOWS it is filled."""
    return Script(lambda v: {"BUY": (price, qty) if v.position == 0 else None, "SELL": None})


def sim(strategy, events, **kw):
    kw.setdefault("costs", FREE)
    kw.setdefault("flatten", False)
    return mm.Simulator(strategy, feed_lat=0.05, order_lat=0.1, **kw).run(events)


def test_order_is_live_only_after_feed_plus_order_latency():
    s = bid_only(100.0)
    ev = [book(0.0), mm.Trade(0.10, 99.0, 5, "SELL"),         # before arrival (0.15)
          mm.Trade(0.20, 99.0, 5, "SELL")]
    r = sim(s, ev)
    assert [(f.t, f.price, f.qty) for f in r.fills] == [(0.20, 100.0, 1.0)]


def test_through_model_needs_a_print_through_our_price():
    s = bid_only(100.0)
    r = sim(s, [book(0.0), mm.Trade(0.2, 100.0, 50, "SELL")])
    assert r.fills == []
    r = sim(bid_only(100.0), [book(0.0), mm.Trade(0.2, 99.95, 1, "SELL")])
    assert len(r.fills) == 1 and r.fills[0].maker


def test_queue_model_waits_behind_the_displayed_queue():
    ev = [book(0.0, bq=10), mm.Trade(0.2, 100.0, 4, "SELL"),   # ahead 10 -> 6
          mm.Trade(0.3, 100.0, 8, "SELL")]                     # 6 used, 2 left -> fill 1 (size 1)
    r = sim(bid_only(100.0), ev, fill_model="queue")
    assert [(f.t, f.qty) for f in r.fills] == [(0.3, 1.0)]
    r = sim(bid_only(100.0, 5), ev, fill_model="queue")
    assert [(f.t, f.qty) for f in r.fills] == [(0.3, 2.0)]
    # buyer-initiated prints at our bid do not consume the bid queue
    r = sim(bid_only(100.0), [book(0.0, bq=1), mm.Trade(0.2, 100.0, 50, "BUY")], fill_model="queue")
    assert r.fills == []


def test_queue_ahead_is_capped_by_the_displayed_quantity():
    ev = [book(0.0, bq=10), book(0.2, bq=3), mm.Trade(0.3, 100.0, 5, "SELL")]
    r = sim(bid_only(100.0, 5), ev, fill_model="queue")
    assert [f.qty for f in r.fills] == [2.0]                    # 3 ahead, 5 traded


def test_marketable_order_walks_the_displayed_levels():
    d = ([(100.0, 5)], [(100.5, 1), (101.0, 2), (101.5, 9)])
    s = Script(lambda v: {"BUY": (101.0, 2.0), "SELL": None})
    r = sim(s, [book(0.0, depth=d)])
    assert [(f.price, f.qty, f.maker) for f in r.fills] == [(100.5, 1.0, False), (101.0, 1.0, False)]


def test_a_fill_can_beat_its_own_cancel():
    # quote at 0.05, arrives 0.15; at 0.25 the strategy pulls it (book seen at
    # 0.20 shows a lower bid), cancel lands 0.35; a print through at 0.30 fills.
    s = Script(lambda v: {"BUY": (100.0, 1.0) if v.book.bid >= 100 else None, "SELL": None})
    ev = [book(0.0), book(0.20, bid=99.0), mm.Trade(0.30, 99.0, 3, "SELL")]
    r = sim(s, ev)
    assert len(r.fills) == 1 and r.fills[0].t == 0.30
    ev = [book(0.0), book(0.20, bid=99.0), mm.Trade(0.40, 99.0, 3, "SELL")]   # after the cancel
    assert sim(s, ev).fills == []


def test_accounting_round_trip_and_charges():
    s = Script(lambda v: {"BUY": (100.0, 1.0), "SELL": (101.0, 1.0)})
    ev = [book(0.0), mm.Trade(0.2, 99.5, 1, "SELL"), mm.Trade(0.3, 101.5, 1, "BUY")]
    r = sim(s, ev)
    assert r.gross == pytest.approx(1.0) and r.net == pytest.approx(1.0)
    cm = mm.MEMBER_FUT
    r = sim(s, ev, costs=cm)
    fee = cm.leg_cost(1, 100.0, False) + cm.leg_cost(1, 101.0, True)
    assert r.fees == pytest.approx(fee) and r.net == pytest.approx(1.0 - fee)
    assert cm.leg_cost(1, 100.0, True) > cm.leg_cost(1, 100.0, False)   # STT on the sell


def test_flatten_crosses_the_spread():
    r = sim(bid_once(100.0), [book(0.0), mm.Trade(0.2, 99.0, 1, "SELL"), book(0.5, 98.0, 99.0)],
            flatten=True)
    assert r.gross == pytest.approx(98.0 - 100.0)                      # sold at the bid
    assert r.flatten_cost == pytest.approx(0.5)                        # half the 1-point spread


def test_adverse_selection_measures_the_later_mid():
    s = bid_once(100.0)
    ev = [book(0.0), mm.Trade(0.2, 99.5, 1, "SELL"), book(0.26, 98.0, 99.0), book(5.0, 99.0, 100.0)]
    out = sim(s, ev).summary()
    # fill at mid 100.5; 1s later mid 98.5 (-2 against a buy); 5s later mid 99.5
    assert out["adverse_1s"] == pytest.approx(2.0) and out["adverse_5s"] == pytest.approx(1.0)
    assert out["spread_captured"] == pytest.approx(0.5)


def test_strategy_sees_a_stale_book_and_position():
    s = bid_only(100.0)
    ev = [book(0.0), book(0.16), mm.Trade(0.2, 99.0, 1, "SELL"), book(0.22)]
    sim(s, ev)
    seen = {round(v.t, 2): v.position for v in s.views}
    assert seen[0.21] == 0.0                     # filled at 0.20, but reported only at 0.25
    assert seen[0.27] == 1.0
    assert all(abs(v.t - v.book.t - 0.05) < 1e-12 for v in s.views)


def test_messages_count_places_and_cancels():
    flip = iter([100.0, 99.5, 100.0])
    s = Script(lambda v: {"BUY": (next(flip), 1.0), "SELL": None})
    r = sim(s, [book(0.0), book(1.0), book(2.0)])
    assert r.messages == 5                       # 3 places + 2 cancels


def test_symmetric_quoter_limits_and_skew():
    q = mm.SymmetricQuoter(tick=0.05, size=1, max_pos=2, half_ticks=1, skew_ticks_per_unit=1)
    b = mm.Book(0, [(100.0, 5)], [(100.2, 5)])
    v = lambda pos: mm.View(0, b, pos, {})
    assert q.quotes(v(0)) == {"BUY": (100.0, 1), "SELL": (100.2, 1)}
    assert q.quotes(v(2))["BUY"] is None and q.quotes(v(-2))["SELL"] is None
    assert q.quotes(v(1))["SELL"][0] == 100.2             # skew can't cross inside the touch
    tight = mm.Book(0, [(100.0, 5)], [(100.05, 5)])
    assert mm.SymmetricQuoter(0.05, 1, 2, min_spread_ticks=2).quotes(mm.View(0, tight, 0, {})) == \
        {"BUY": None, "SELL": None}


def test_loader_infers_trades_with_lee_ready():
    cols = {"recv_ns": 0, "token": 1}
    def row(t, ltp, vol, bid, ask):
        r = dict(cols, exch_ts=t, ltp=ltp, volume=vol)
        for i in range(1, 6):
            r.update({f"b{i}p": bid - (i - 1) * 0.05 if i == 1 else np.nan, f"b{i}q": 10 if i == 1 else np.nan,
                      f"a{i}p": ask if i == 1 else np.nan, f"a{i}q": 10 if i == 1 else np.nan})
        return r
    rows = pd.DataFrame([row(1, 100.0, 1000, 100.0, 100.1), row(2, 100.1, 1005, 100.0, 100.1),
                         row(3, 100.0, 1012, 100.0, 100.1), row(4, 100.0, 1012, 100.0, 100.1)])
    ev = mm.events_from_depth(rows)
    trades = [e for e in ev if isinstance(e, mm.Trade)]
    assert [(t.t, t.qty, t.aggressor) for t in trades] == [(2.0, 5.0, "BUY"), (3.0, 7.0, "SELL")]
    assert sum(isinstance(e, mm.Book) for e in ev) == 4


def test_an_order_cancelled_in_flight_still_rests_until_the_cancel_lands():
    # quoted at 0.05 (arrives 0.15); book at 0.06 seen 0.11 -> cancel decided
    # while the order is still in flight, lands 0.21. A print through at 0.18
    # finds it resting.
    s = Script(lambda v: {"BUY": (100.0, 1.0) if v.book.bid >= 100 else None, "SELL": None})
    ev = [book(0.0), book(0.06, bid=99.0), mm.Trade(0.18, 99.0, 3, "SELL")]
    r = sim(s, ev)
    assert [f.t for f in r.fills] == [0.18]


def test_taking_stops_at_the_limit_and_rests_the_remainder():
    d = ([(100.0, 5)], [(100.5, 1), (101.0, 2), (101.5, 9)])
    s = Script(lambda v: {"BUY": (101.0, 5.0), "SELL": None} if not v.orders else
               {"BUY": (101.0, 5.0), "SELL": None})
    r = sim(s, [book(0.0, depth=d)])
    assert [(f.price, f.qty) for f in r.fills] == [(100.5, 1.0), (101.0, 2.0)]
