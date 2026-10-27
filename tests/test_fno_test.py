"""scripts/fno_test.py against FNO_PREREG §10, on a synthetic market."""

import numpy as np
import pandas as pd
import pytest

from scripts import fno_test as ft

SESS = pd.bdate_range("2024-01-01", periods=340)
EXP = list(SESS[19::20])                         # an expiry every 20 sessions


def world(rets=None, n=10, seed=0, skew=None, drop=None, actions=None, lot=100.0, value=None,
          parity=None):
    """n stocks; closes compound `rets` (date x symbol); every expiry's
    contract trades for the 60 sessions up to it. `drop` = {symbol: last date
    it has contracts}."""
    syms = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(seed)
    if rets is None:
        rets = pd.DataFrame(rng.normal(0, 0.01, (len(SESS), n)), index=SESS, columns=syms)
    px = 100 * np.cumprod(1 + rets)
    fut, oi = [], []
    for e in EXP:
        live = SESS[(SESS <= e) & (SESS > e - pd.Timedelta(days=90))]
        for s in syms:
            for t in live:
                if drop and s in drop and t > drop[s]:
                    continue
                fut.append(dict(date=t, instrument="FUTSTK", symbol=s, expiry=e,
                                close=px.at[t, s], lot=lot,
                                value_lakh=1.0 if value is None else value.at[t, s]))
    for s in syms:
        for t in SESS:
            oi.append(dict(date=t, symbol=s, ret=rets.at[t, s], cal_spread_ann=0.06))
    oi += [dict(date=t, symbol="NIFTY", ret=0.0, cal_spread_ann=0.0) for t in SESS]
    sk = skew if skew is not None else pd.DataFrame(
        np.tile(np.arange(n, dtype=float), (len(SESS), 1)), index=SESS, columns=syms)
    ivd = sk.stack().rename("skew30").reset_index().set_axis(["date", "symbol", "skew30"], axis=1)
    f = pd.DataFrame(fut)
    pdev = parity if parity is not None else sk / 1000      # same ranks as skew by default
    exiv = f[["date", "symbol", "expiry"]].assign(
        fut_close=100.0, parity=[100.0 * (1 + pdev.at[t, s]) for t, s in zip(f.date, f.symbol)])
    acts = actions if actions is not None else pd.DataFrame(
        columns=["date", "symbol", "kind", "adj_factor", "dividend"])
    return ft.Data.build(pd.DataFrame(fut), acts, pd.DataFrame(oi), ivd, exiv), rets, px


def dev_cycles(d):
    cs = ft.cycles(d.sessions, d.expiries)
    return [c for c in cs if c.signal >= d.sessions[ft.MIN_HISTORY + 1]]


def test_calendar_entry_after_signal_and_exit_before_expiry_week():
    d, _, _ = world()
    for c in ft.cycles(d.sessions, d.expiries):
        i = d.sessions.get_loc(c.signal)
        assert d.sessions[i - 1] == c.expiry                 # s_k = first session after E_k
        assert d.sessions[i + 1] == c.entry                  # entry = next session
        assert d.sessions.get_loc(c.next_expiry) - d.sessions.get_loc(c.exit) == 5


def test_signal_dated_after_s_k_does_not_change_the_book():
    d, _, _ = world()
    c = dev_cycles(d)[0]
    base = ft.run_quantile(d, [c], ft.signal_skew, long_low=True)
    # from the entry day on, ranks reverse hard enough to reorder any window
    # that reads them
    d.skew.loc[d.sessions > c.signal] = -100 * d.skew.loc[d.sessions > c.signal]
    after = ft.run_quantile(d, [c], ft.signal_skew, long_low=True)
    assert base.longs.iloc[0] == after.longs.iloc[0] == "S0 S1"


def test_entry_is_at_the_next_close_not_the_signal_close():
    rets = pd.DataFrame(0.0, index=SESS, columns=[f"S{i}" for i in range(10)])
    d0, _, _ = world(rets.copy())
    c = dev_cycles(d0)[0]
    rets.loc[c.entry, "S0"] = 0.10                       # jump into the entry close
    d, _, _ = world(rets)
    df = ft.run_quantile(d, [c], ft.signal_skew, long_low=True, spread_bps=0)
    assert df.long.iloc[0] == pytest.approx(0.0)         # S0 is long; the jump is not earned


def _act(day, sym, kind, f=1.0, div=0.0):
    return pd.DataFrame({"date": [day], "symbol": [sym], "kind": [kind],
                         "adj_factor": [f], "dividend": [div]})


def test_bonus_and_dividend_inside_the_hold_and_relisted_close():
    rets = pd.DataFrame(0.0, index=SESS, columns=[f"S{i}" for i in range(10)])
    d0, _, px0 = world(rets.copy())
    c = dev_cycles(d0)[0]
    mid = d0.sessions[d0.sessions.get_loc(c.entry) + 3]
    # bonus: the contract halves on `mid`, flat otherwise
    r = rets.copy(); r.loc[mid, "S0"] = -0.5
    d, _, _ = world(r, actions=_act(mid, "S0", "multiply", 0.5))
    assert ft.leg_return(d, "S0", c.next_expiry, c.entry, c.exit) == pytest.approx(0.0)
    # dividend Rs4 on a Rs100 contract
    r = rets.copy(); r.loc[mid, "S1"] = -0.04
    d, _, _ = world(r, actions=_act(mid, "S1", "subtract", 1.0, 4.0))
    assert ft.leg_return(d, "S1", c.next_expiry, c.entry, c.exit) == pytest.approx(0.0)
    # relisted: closed at the previous close; later moves are not earned
    r = rets.copy(); r.loc[mid, "S2"] = -0.4
    r.loc[d0.sessions[d0.sessions.get_loc(mid) + 1], "S2"] = 0.2
    d, _, _ = world(r, actions=_act(mid, "S2", "relisted", np.nan))
    assert ft.leg_return(d, "S2", c.next_expiry, c.entry, c.exit) == pytest.approx(0.0)


def test_universe_needs_the_contract_and_keeps_later_leavers():
    d0, _, _ = world()
    cs = dev_cycles(d0)
    c0, c_last = cs[0], cs[-1]
    d, _, _ = world(drop={"S3": c0.exit, "S4": c0.signal})
    assert "S3" in ft.universe(d, c0)                    # leaves later: still in
    assert "S3" not in ft.universe(d, c_last)
    assert "S4" not in ft.universe(d, c0)                # no contract on the entry day


def test_costs_on_both_books_both_sides():
    d, _, _ = world()
    c = dev_cycles(d)[0]
    df = ft.run_quantile(d, [c], ft.signal_skew, long_low=True, spread_bps=5)
    one = ft.cost_frac(d, "S0", c.next_expiry, c.entry, "BUY", 5)
    other = ft.cost_frac(d, "S9", c.next_expiry, c.entry, "SELL", 5)
    assert df.cost.iloc[0] == pytest.approx(one + other)
    assert one > 10 / 1e4                                # spread both sides + charges


def test_swapping_long_and_short_flips_the_sign():
    d, _, _ = world()
    cs = dev_cycles(d)
    a = ft.run_quantile(d, cs, ft.signal_skew, long_low=True)
    b = ft.run_quantile(d, cs, ft.signal_skew, long_low=False)
    assert np.allclose(a.gross, -b.gross) and abs(a.gross).sum() > 0


def test_pair_formation_uses_only_the_window_ending_s_k():
    d0, rets, _ = world()
    c = dev_cycles(d0)[0]
    syms = ft.universe(d0, c)
    base = ft.form_pairs(d0, c, syms, 3)
    win = ft._window(d0.sessions, c.signal, ft.FORMATION)
    r = rets.copy()
    r.loc[r.index < win[0]] = 0.05                       # before the window
    r.loc[r.index > c.signal] = -0.03                    # after s_k
    d1, _, _ = world(r)
    assert ft.form_pairs(d1, c, syms, 3) == base


def test_pair_that_never_diverges_contributes_zero_and_one_round_trip_max():
    rets = pd.DataFrame(0.0, index=SESS, columns=[f"S{i}" for i in range(10)])
    d0, _, _ = world(rets.copy())
    c = dev_cycles(d0)[0]
    assert ft.trade_pair(d0, c, "S0", "S1", 0.01, 5) is None
    days = d0.sessions[(d0.sessions > c.signal) & (d0.sessions <= c.exit)]
    r = rets.copy()
    r.loc[days[1], "S0"] = 0.05                          # diverge
    r.loc[days[3], "S0"] = -0.05 / 1.05                  # converge
    r.loc[days[5], "S0"] = 0.05                          # diverge again
    d, _, _ = world(r)
    t = ft.trade_pair(d, c, "S0", "S1", 0.01, 0)
    assert t["open"] == days[2] and t["close"] == days[4]
    assert t["long"] == "S1"                             # long the lower index
    pairs = [("S0", "S1", 0.01)]
    orig = ft.form_pairs
    try:
        ft.form_pairs = lambda *a, **k: pairs
        cyc, trades = ft.run_pairs(d, [c], spread_bps=0)
    finally:
        ft.form_pairs = orig
    assert len(trades) == 1


def test_development_cannot_read_the_holdout():
    d, _, _ = world()
    c = ft.Cycle(*(pd.Timestamp("2025-09-01"),) * 3, pd.Timestamp("2025-10-01"),
                 pd.Timestamp("2025-10-20"))
    with pytest.raises(ft.HoldoutError):
        ft.run_quantile(d, [c], ft.signal_skew, long_low=True)
    with pytest.raises(ft.HoldoutError):
        ft.run_pairs(d, [c])
    assert all(x.entry < ft.HOLDOUT_START for x in ft.period(
        [c, ft.Cycle(*(pd.Timestamp("2024-01-01"),) * 3, pd.Timestamp("2024-02-01"),
                     pd.Timestamp("2024-02-20"))], holdout=False))


def test_liquidity_rank_uses_only_sessions_ending_s_k(monkeypatch):
    d0, _, _ = world()
    c = dev_cycles(d0)[0]
    syms = [f"S{i}" for i in range(10)]
    v = pd.DataFrame(np.tile(np.arange(10, 0, -1.0), (len(SESS), 1)), index=SESS, columns=syms)
    v.loc[v.index > c.signal] = np.arange(1, 11.0)      # order flips after s_k
    d, _, _ = world(value=v)
    monkeypatch.setattr(ft, "TOP_N", 5)
    assert ft.universe(d, c) == ["S0", "S1", "S2", "S3", "S4"]


def test_parity_signal_dated_after_s_k_does_not_change_the_book():
    d0, _, _ = world()
    c = dev_cycles(d0)[0]
    syms = [f"S{i}" for i in range(10)]
    p = pd.DataFrame(np.tile(np.arange(10) / 1000, (len(SESS), 1)), index=SESS, columns=syms)
    p.loc[p.index > c.signal] = -p.loc[p.index > c.signal] * 100
    d, _, _ = world(parity=p)
    df = ft.run_quantile(d, [c], ft.signal_parity, long_low=False)
    assert df.longs.iloc[0] == "S8 S9"                   # highest parity dev up to s_k


def test_an_expiry_moved_mid_contract_is_one_contract():
    rows = [dict(date=t, symbol="A", expiry=pd.Timestamp("2023-06-29"))
            for t in pd.bdate_range("2023-05-01", "2023-06-27")]
    rows.append(dict(date=pd.Timestamp("2023-06-28"), symbol="A", expiry=pd.Timestamp("2023-06-28")))
    rows += [dict(date=t, symbol="A", expiry=pd.Timestamp("2023-07-27"))
             for t in pd.bdate_range("2023-06-01", "2023-07-27")]
    alias = ft.expiry_aliases(pd.DataFrame(rows))
    assert alias == {pd.Timestamp("2023-06-29"): pd.Timestamp("2023-06-28")}
