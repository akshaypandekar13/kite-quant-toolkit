"""scripts/condor_test.py against CONDOR_PREREG §7, on a synthetic NIFTY chain."""

import numpy as np
import pandas as pd
import pytest

from scripts import condor_test as ct
from scripts import fno_test as ft
from scripts.fno_features import b76_delta, b76_price

SESS = pd.bdate_range("2024-01-01", periods=120)
EXP = list(SESS[19::20])
F0, SIG, LOT = 22000.0, 0.15, 75.0


def market(F=None, tweak=None, extra_futs=None):
    """Flat forward F0 unless `F` (date -> forward) is given; every expiry's
    chain is priced at SIG on a 50-point grid. `tweak(opts)` edits the
    options frame before the build."""
    F = F or {}
    futs, opts = [], []
    for e in EXP:
        for t in SESS[(SESS <= e)]:
            f = F.get(t, F0)
            futs.append(dict(date=t, symbol="NIFTY", instrument="FUTIDX", expiry=e, close=f, lot=LOT))
            T = max((e - t).days, 1) / 365
            for k in np.arange(17000, 27050, 50.0):
                for typ in ("CE", "PE"):
                    p = float(b76_price(f, k, T, SIG, typ == "CE"))
                    opts.append(dict(date=t, expiry=e, strike=k, opt_type=typ,
                                     close=p, settle=p, contracts=10))
    futs = pd.DataFrame(futs + (extra_futs or []))
    opts = pd.DataFrame(opts)
    if tweak:
        opts = tweak(opts)
    carry = pd.Series(0.06, index=SESS)
    idx = pd.Series([F.get(t, F0) for t in SESS], index=SESS)
    return ct.Market.build(futs, opts, carry, idx)


def first_cycle(m):
    return ft.cycles(m.sessions, m.expiries)[0]


def test_delta_targets_and_wings():
    m = market()
    c = first_cycle(m)
    k = ct.choose(m, c)
    T = (c.next_expiry - c.signal).days / 365
    grid = np.arange(17000, 27050, 50.0)
    want_p = grid[np.argmin(np.abs(b76_delta(F0, grid, T, SIG, False) + 0.15))]
    want_c = grid[np.argmin(np.abs(b76_delta(F0, grid, T, SIG, True) - 0.15))]
    assert (k.short_put, k.short_call) == (want_p, want_c)
    W = 0.04 * F0
    assert k.long_put < k.short_put and k.long_call > k.short_call
    assert k.long_put == grid[np.argmin(np.abs(grid - (want_p - W)))]
    assert k.long_call == grid[np.argmin(np.abs(grid - (want_c + W)))]


def test_strikes_use_only_s_k():
    m0 = market()
    c = first_cycle(m0)
    base = ct.choose(m0, c)

    def later_prices_change(o):
        o = o.copy()
        later = o.date > c.signal
        o.loc[later, "close"] *= 3
        o.loc[later, "contracts"] = 0
        return o
    assert ct.choose(market(tweak=later_prices_change), c) == base


def test_entry_is_the_next_close_and_exit_is_5_before_expiry():
    m0 = market()
    c = first_cycle(m0)
    k = ct.choose(m0, c)
    assert m0.sessions[m0.sessions.get_loc(c.signal) + 1] == c.entry
    assert m0.sessions.get_loc(c.next_expiry) - m0.sessions.get_loc(c.exit) == 5

    def bump_entry(o):
        o = o.copy()
        hit = (o.date == c.entry) & (o.strike == k.short_put) & (o.opt_type == "PE")
        o.loc[hit, "close"] += 10.0
        return o
    r0 = ct.run_cycle(market(), c)
    r1 = ct.run_cycle(market(tweak=bump_entry), c)
    assert r1.credit - r0.credit == pytest.approx(10.0)


def test_settle_fallback_only_for_untraded_legs_and_counted():
    m0 = market()
    c = first_cycle(m0)
    k = ct.choose(m0, c)

    def untraded_wing(o):
        o = o.copy()
        hit = (o.date == c.entry) & (o.strike == k.long_call) & (o.opt_type == "CE")
        o.loc[hit, "contracts"] = 0
        o.loc[hit, "close"] = 999.0                       # stale close must be ignored
        o.loc[hit, "settle"] = 5.0
        return o
    r0 = ct.run_cycle(m0, c)
    r = ct.run_cycle(market(tweak=untraded_wing), c)
    assert r.fallbacks == 1 and r0.fallbacks == 0
    base_wing = m0.chains[(c.entry, c.next_expiry)].loc[(k.long_call, "CE")].close
    assert r.credit - r0.credit == pytest.approx(base_wing - 5.0)


def test_pnl_at_expiry_worthless_and_beyond_the_wing():
    m0 = market()
    c = first_cycle(m0)
    r = ct.run_cycle(m0, c, to_expiry=True, spread=0)
    assert r.debit == 0 and r.net == pytest.approx(r.credit - r.costs)
    crash = market(F={c.next_expiry: 15000.0})          # index far below the put wing
    k = ct.choose(crash, c)
    r = ct.run_cycle(crash, c, to_expiry=True, spread=0)
    put_width = k.short_put - k.long_put
    assert r.net == pytest.approx(r.credit - put_width - r.costs)


def test_costs_stt_on_sells_stamp_on_buys_eight_orders():
    p = 100.0
    base = ct.BROKERAGE_ORDER / LOT + ct.EXCHANGE * p + ct.SEBI * p
    base *= 1 + ct.GST
    assert ct.order_cost(p, True, LOT) == pytest.approx(base + ct.STT_SELL * p)
    assert ct.order_cost(p, False, LOT) == pytest.approx(base + ct.STAMP_BUY * p)
    m = market()
    c = first_cycle(m)
    r = ct.run_cycle(m, c, spread=0)
    k = ct.choose(m, c)
    ch_in, ch_out = m.chains[(c.entry, c.next_expiry)], m.chains[(c.exit, c.next_expiry)]
    want = sum(ct.order_cost(ch_in.loc[(s, t)].close, sign < 0, LOT)
               + ct.order_cost(ch_out.loc[(s, t)].close, sign > 0, LOT) for s, t, sign in k.legs())
    assert r.costs == pytest.approx(want)
    assert ct.run_cycle(m, c, spread=0.5).costs == pytest.approx(want + 8 * 0.5)


def test_expiry_alias_merges_a_moved_series():
    moved = EXP[2] + pd.Timedelta(days=2)               # same series, relabelled
    extra = [dict(date=t, symbol="NIFTY", instrument="FUTIDX", expiry=moved, close=F0, lot=LOT)
             for t in SESS[(SESS <= EXP[2]) & (SESS > EXP[1])][:-1]]
    m = market(extra_futs=extra)
    assert moved not in m.expiries
    gaps = [(b - a).days for a, b in zip(m.expiries[:-1], m.expiries[1:])]
    assert min(gaps) > 7


def test_absent_leg_skips_at_entry_and_is_intrinsic_at_exit():
    m0 = market()
    c = first_cycle(m0)
    k = ct.choose(m0, c)
    drop_entry = lambda o: o[~((o.date == c.entry) & (o.strike == k.long_put) & (o.opt_type == "PE"))]
    assert ct.run_cycle(market(tweak=drop_entry), c) is None
    drop_exit = lambda o: o[~((o.date == c.exit) & (o.strike == k.long_put) & (o.opt_type == "PE"))]
    r = ct.run_cycle(market(tweak=drop_exit), c)
    assert r.missing_exit == 1


def test_development_cannot_read_the_holdout():
    m = market()
    c = ft.Cycle(*(pd.Timestamp("2025-09-01"),) * 3, pd.Timestamp("2025-10-01"),
                 pd.Timestamp("2025-10-20"))
    with pytest.raises(ft.HoldoutError):
        ct.run_cycle(m, c)


def test_risk_uses_the_wider_wing():
    m0 = market()
    c = first_cycle(m0)
    k0 = ct.choose(m0, c)
    gap = 0.04 * F0 + 200

    def thin_calls(o):                                   # no calls just past the target wing
        drop = (o.opt_type == "CE") & (o.strike > k0.short_call) & (o.strike < k0.short_call + gap)
        return o[~drop]
    m = market(tweak=thin_calls)
    k = ct.choose(m, c)
    r = ct.run_cycle(m, c)
    call_w, put_w = k.long_call - k.short_call, k.short_put - k.long_put
    assert call_w > put_w
    assert r.risk == pytest.approx(call_w - r.credit)
