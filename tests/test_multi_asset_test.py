"""scripts/multi_asset_test.py on synthetic markets with known answers."""

import numpy as np
import pandas as pd
import pytest

from scripts import multi_asset_test as mt


def market(closes: dict[str, np.ndarray], opens: dict[str, np.ndarray] | None = None,
           start="2010-01-01", liquid=True, vix=None):
    cal = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    prices = {}
    for k, c in closes.items():
        o = opens[k] if opens and k in opens else c
        prices[k] = pd.DataFrame({"open": o, "close": c}, index=cal)
    meta = {k: (liquid, False) for k in closes}
    v = pd.Series(vix if vix is not None else 15.0, index=cal)
    return mt.Market(prices, meta, cal, v)


def months(mk, a, b):
    return pd.Period(a, "M"), pd.Period(b, "M")


def test_splice_rescales_the_old_source_to_meet_the_new():
    cal = pd.bdate_range("2020-01-01", periods=6)
    old = pd.DataFrame({"open": [10, 11, 12, 13, 14, 15.], "close": [10, 11, 12, 13, 14, 15.]}, index=cal)
    new = pd.DataFrame({"open": [130, 140.], "close": [130, 140.]}, index=cal[3:5])
    s = mt.splice([old, new], cal)
    assert s["close"].iloc[:4].tolist() == pytest.approx([100, 110, 120, 130])
    assert s["close"].iloc[4] == 140
    # after the newer source ends its last close carries (holiday rule), never the old one
    assert s["close"].iloc[5] == 140


def test_eligibility_needs_thirteen_month_ends():
    n = 400
    mk = market({"A": np.linspace(100, 200, n)})
    m0 = mk.mclose.index[0]
    assert mk.eligible(["A"], m0 + 11) == []
    assert mk.eligible(["A"], m0 + 12) == ["A"]


def test_gtaa_holds_only_assets_above_their_10_month_sma(monkeypatch):
    n = 420
    up, down = np.linspace(100, 300, n), np.linspace(300, 100, n)
    mk = market({"UP": up, "DOWN": down})
    monkeypatch.setattr(mt, "S1_SET", ["UP", "DOWN"])
    w = mt.s1_gtaa(mk, mk.mclose.index[14])
    assert w == {"UP": 0.5}                          # DOWN's slot goes to cash


def test_dual_momentum_falls_to_cash_without_gilts(monkeypatch):
    n = 420
    mk = market({"A": np.linspace(100, 101, n), "B": np.linspace(100, 300, n), "C": np.linspace(100, 50, n)})
    monkeypatch.setattr(mt, "S2_SET", ["A", "B", "C"])
    w = mt.s2_dual(mk, mk.mclose.index[14], cash=0.06)
    assert w == {"B": 0.5}                           # A beat C but not cash


def test_decision_uses_month_end_close_and_trades_next_open():
    # The asset jumps +50% at the open of the first session of a month. A rule
    # deciding on that month-end's data must hold it from the open, so it does
    # NOT earn the gap; it earns open->close of that day onward.
    n = 300
    c = np.full(n, 100.0)
    o = c.copy()
    mk0 = market({"A": c})
    first_trade = mk0.cal[mk0.cal > mk0.month_end[0]][0]
    i = mk0.cal.get_loc(first_trade)
    c[i:] = 150.0
    o[i:] = 150.0
    mk = market({"A": c}, {"A": o})
    eq, *_ = mt.run(mk, lambda mk, m: {"A": 1.0}, cash=0.0,
                    first=mk.mclose.index[0], last=mk.mclose.index[3])
    assert eq.max() < 1.0                            # only costs; the gap was not earned


def test_holding_earns_gaps_after_entry_and_costs_are_charged():
    n = 300
    c = np.full(n, 100.0)
    mk0 = market({"A": c})
    t1 = mk0.cal[mk0.cal > mk0.month_end[0]][0]
    i = mk0.cal.get_loc(t1)
    c2 = c.copy()
    c2[i + 3:] = 110.0
    mk = market({"A": c2})
    eq, w, turn, _ = mt.run(mk, lambda mk, m: {"A": 1.0}, cash=0.0,
                            first=mk.mclose.index[0], last=mk.mclose.index[0])
    buy = mt.side_cost(mk, "A", True)
    assert eq.iloc[-1] == pytest.approx(1.10 - buy * 1.10, rel=1e-6) or \
        eq.iloc[-1] == pytest.approx((1 - buy) * 1.10, abs=2e-4)
    assert turn.iloc[0] == pytest.approx(1.0)


def test_cash_accrues_when_nothing_is_held():
    n = 300
    mk = market({"A": np.full(n, 100.0)})
    eq, *_ = mt.run(mk, lambda mk, m: {}, cash=0.06,
                    first=mk.mclose.index[0], last=mk.mclose.index[-2])
    days = len(eq)
    assert eq.iloc[-1] == pytest.approx(1.06 ** (days / 252), rel=1e-3)


def test_index_proxies_pay_the_expense_drag():
    n = 300
    mk = market({"A": np.full(n, 100.0)}, liquid=False)
    eq, *_ = mt.run(mk, lambda mk, m: {"A": 1.0}, cash=0.0,
                    first=mk.mclose.index[0], last=mk.mclose.index[0])
    yrs = len(eq) / 252
    lost = 1 - eq.iloc[-1] / eq.iloc[0]
    assert lost == pytest.approx(mt.DRAG * yrs, rel=0.05)


def test_ml_trains_only_on_labels_already_known(monkeypatch):
    # The label of decision month p is the return from p to p+1. At decision m
    # only p <= m-1 may be in training. Plant a target that is only learnable
    # if month m's own (future) label leaks in.
    n = 1300
    rng = np.random.default_rng(0)
    mk = market({k: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for k in "ABCD"})
    monkeypatch.setattr(mt, "ML_SET", list("ABCD"))
    monkeypatch.setattr(mt, "MIN_LABEL_MONTHS", 6)
    rule = mt.MLRule(mk, 0.0)
    seen = []
    orig = rule._row

    def spy(p):
        seen.append(p)
        return orig(p)
    rule._row = spy
    m = mk.mclose.index[30]
    rule(mk, m)
    assert max(seen) == m - 1


def test_verdict_splits_development_and_holdout():
    idx = pd.period_range("2020-01", "2023-12", freq="M")
    s = pd.Series(0.02, index=idx)
    b = pd.Series(0.01, index=idx)
    s[idx > mt.DEV_END + 1] = 0.0
    v = mt.verdict(s + np.random.default_rng(1).normal(0, 1e-4, len(idx)), b, -0.1, -0.2)
    assert v["dev_excess_m"] == pytest.approx(0.01, abs=1e-3)
    assert v["hold_excess_m"] == pytest.approx(-0.01, abs=1e-3)
    assert v["P2"] is False and v["P3"] is True and v["PASS"] is False
    assert v["hold_months"] == (idx > mt.DEV_END + 1).sum()
    # the first holdout month (Feb 2022, from the Jan 2022 decision) belongs to the holdout
    s2 = s.copy()
    s2[pd.Period("2022-02", "M")] = 0.5
    assert mt.verdict(s2, b, -0.1, -0.2)["hold_excess_m"] > 0.005


def test_a_gap_inside_a_series_carries_the_last_close():
    cal = pd.bdate_range("2020-01-01", periods=12)
    c = pd.Series([10.0] * 3 + [np.nan] * 7 + [12.0, 12.0], index=cal)
    s = mt.splice([pd.DataFrame({"open": c, "close": c})], cal)
    assert s["close"].notna().all()
    assert s["close"].iloc[3:10].tolist() == [10.0] * 7
