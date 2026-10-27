"""scripts/fno_features.py on synthetic chains with known answers."""

import numpy as np
import pandas as pd
import pytest

from scripts import fno_features as ff

D = pd.Timestamp("2026-10-01")


def test_implied_vol_round_trip_and_no_arbitrage_band():
    F, T = 100.0, 30 / 365
    K = np.array([80, 90, 100, 110, 120.0])
    for is_call in (True, False):
        sig = np.array([0.15, 0.25, 0.4, 0.8, 1.5])
        px = ff.b76_price(F, K, T, sig, is_call)
        iv = ff.implied_vol(px, np.full(5, F), K, np.full(5, T), np.full(5, is_call))
        assert np.allclose(iv, sig, atol=1e-5)
    # below intrinsic -> no IV
    assert np.isnan(ff.implied_vol([5.0], [100.0], [90.0], [T], [True])[0])


def test_interp_brackets_and_never_extrapolates():
    df = pd.DataFrame({"g": [0, 0, 1, 1], "x": [-1.0, 1.0, 0.5, 1.0], "y": [10.0, 20.0, 1.0, 2.0]})
    r = ff.interp_at(df, "g", "x", "y", 0.0)
    assert r[0] == pytest.approx(15.0)
    assert 1 not in r.index                       # 0 is outside [0.5, 1.0]


def _chain(F, days, smile, symbol="ABC", future=True):
    T = days / 365
    exp = D + pd.Timedelta(days=days)
    rows = []
    for K in np.arange(F * 0.7, F * 1.3, F * 0.025):
        for typ in ("CE", "PE"):
            px = float(ff.b76_price(F, K, T, smile(np.log(K / F)), typ == "CE"))
            rows.append(dict(date=D, symbol=symbol, expiry=exp, strike=K, opt_type=typ,
                             close=px, contracts=10, oi=1000 if typ == "CE" else 1500))
    futs = pd.DataFrame([dict(date=D, symbol=symbol, expiry=exp, close=F)]) if future else \
        pd.DataFrame(columns=["date", "symbol", "expiry", "close"])
    return pd.DataFrame(rows), futs


def test_expiry_iv_recovers_atm_and_skew_sign():
    smile = lambda k: 0.25 - 0.3 * k                # puts (k<0) richer
    o, f = _chain(100.0, 30, smile)
    ex = ff.expiry_iv(o, f).iloc[0]
    assert ex.fwd_src == "future"
    assert ex.atm_iv == pytest.approx(0.25, abs=2e-3)
    assert ex.skew25 > 0.02
    assert ex.oi_pe / ex.oi_ce == pytest.approx(1.5)


def test_parity_forward_when_no_future():
    o, f = _chain(250.0, 7, lambda k: 0.2, future=False)
    ex = ff.expiry_iv(o, f).iloc[0]
    assert ex.fwd_src == "parity"
    assert ex.forward == pytest.approx(250.0, rel=1e-6)
    assert ex.atm_iv == pytest.approx(0.2, abs=1e-3)


def test_iv30_interpolates_total_variance():
    ex = pd.DataFrame({"date": D, "symbol": "ABC", "days": [3, 20, 40],
                       "atm_iv": [0.9, 0.2, 0.3], "skew25": [0.0, 0.01, 0.03],
                       "oi_ce": 1, "oi_pe": 1, "vol_ce": 1, "vol_pe": 1})
    d = ff.iv_daily(ex).iloc[0]
    want = np.sqrt((0.04 * 20 + 0.09 * 40) / 2 / 30)   # the 3-day expiry is excluded
    assert d.iv30 == pytest.approx(want)
    assert d.skew30 == pytest.approx(0.02)
    assert d.iv_front == 0.2 and d.iv_back == 0.3 and d.iv30_interp


def _fut(rows):
    df = pd.DataFrame(rows, columns=["date", "symbol", "expiry", "close", "oi"])
    df["date"], df["expiry"] = pd.to_datetime(df.date), pd.to_datetime(df.expiry)
    df["contracts"], df["value_lakh"] = 1.0, 1.0
    return df


def test_buildup_uses_one_contract_and_total_oi_across_a_roll():
    n, x = "2026-10-29", "2026-11-24"
    f = _fut([
        ("2026-10-26", "ABC", n, 100, 900), ("2026-10-26", "ABC", x, 102, 100),
        # roll: OI moves near -> next, total unchanged; next contract is flat
        ("2026-10-27", "ABC", n, 99, 100), ("2026-10-27", "ABC", x, 102, 900),
        # near still trades but holds little OI: the move is read off next
        ("2026-10-28", "ABC", n, 90, 50), ("2026-10-28", "ABC", x, 103, 1050),
    ])
    d = ff.oi_buildup(f).set_index("date")
    # 10-27: active = near (900 prev OI): -1%; total OI 1000 -> 1000
    assert d.loc["2026-10-27", "ret"] == pytest.approx(-0.01)
    assert d.loc["2026-10-27", "oi_chg_pct"] == 0
    assert d.loc["2026-10-27", "buildup"] == "NEUTRAL"
    # 10-28: active = next (900 prev OI), 102 -> 103; not near's 99 -> 90
    assert d.loc["2026-10-28", "ret"] == pytest.approx(103 / 102 - 1)
    assert d.loc["2026-10-28", "buildup"] == "LONG_BUILDUP"
    assert d.loc["2026-10-26", "near_oi_share"] == pytest.approx(0.9)


def test_corp_action_is_adjusted_not_blanked_and_gap_blanks_change():
    e = "2026-12-29"
    # 1:1 bonus on 10-27: price halves, OI in shares doubles; underneath,
    # price +2% and OI +5%
    f = _fut([("2026-10-26", "ABC", e, 100, 1000), ("2026-10-27", "ABC", e, 51, 2100),
              ("2026-10-26", "XYZ", e, 10, 10), ("2026-10-27", "XYZ", e, 10, 10),
              ("2026-10-28", "XYZ", e, 10, 10), ("2026-10-28", "ABC", e, 50, 2200)])
    f = f[~((f.symbol == "XYZ") & (f.date == "2026-10-27"))]   # XYZ skips a session
    acts = pd.DataFrame({"date": [pd.Timestamp("2026-10-27")], "symbol": ["ABC"],
                         "kind": ["multiply"], "adj_factor": [0.5], "dividend": [0.0]})
    d = ff.oi_buildup(f, actions=acts).set_index(["symbol", "date"])
    r = d.loc[("ABC", "2026-10-27")]
    assert r.corp_action and r.adj_factor == 0.5
    assert r.ret == pytest.approx(0.02) and r.oi_chg_pct == pytest.approx(0.05)
    assert r.buildup == "LONG_BUILDUP"
    assert d.loc[("ABC", "2026-10-28"), "buildup"] == "SHORT_BUILDUP"
    assert d.loc[("ABC", "2026-10-28"), "adj_factor"] == 1.0
    assert pd.isna(d.loc[("XYZ", "2026-10-28"), "oi_chg_pct"])


def _strikes(day, symbol, strikes, expiry="2026-12-29"):
    return pd.DataFrame({"date": pd.Timestamp(day), "symbol": symbol,
                         "expiry": pd.Timestamp(expiry), "strike": strikes,
                         "opt_type": "CE", "oi": 100})


def test_strike_relisting_flags_a_bonus_not_a_crash():
    old = list(range(800, 1210, 20))                   # 21 strikes
    opts = pd.concat([
        _strikes("2026-10-26", "BONUS", old),
        _strikes("2026-10-27", "BONUS", [k / 2 for k in old]),
        _strikes("2026-10-26", "CRASH", old),
        _strikes("2026-10-27", "CRASH", list(range(500, 800, 20)) + old),  # new strikes added
    ])
    a = ff.strike_actions(opts)
    assert list(a.symbol) == ["BONUS"] and a.kind.iloc[0] == "multiply"
    assert a.adj_factor.iloc[0] == pytest.approx(0.5)
    assert a.date.iloc[0] == pd.Timestamp("2026-10-27")


def test_factor_is_exact_when_new_strikes_are_added_on_the_split_day():
    old = list(range(800, 1210, 20))
    new = [k / 10 for k in old] + [60, 62, 64, 66, 68, 70]   # 10:1 plus new low strikes
    a = ff.strike_actions(pd.concat([_strikes("2026-10-26", "S", old),
                                     _strikes("2026-10-27", "S", new)]))
    assert a.adj_factor.iloc[0] == pytest.approx(0.1)


def test_delivery_z_uses_only_prior_sessions():
    days = pd.bdate_range("2026-06-01", periods=25)
    f = _fut([(d, "ABC", "2026-12-29", 100, 1000) for d in days])
    pct = [50.0 + (i % 2) for i in range(24)] + [90.0]
    deliv = pd.DataFrame({"date": days, "symbol": "ABC", "series": "EQ",
                          "deliv_pct": pct, "traded_qty": 1})
    d = ff.oi_buildup(f, deliv)
    prior = pd.Series(pct[4:24])
    assert d.deliv_pct_z20.iloc[-1] == pytest.approx((90 - prior.mean()) / prior.std())


def test_dividend_is_a_subtraction_not_a_grid_ratio():
    old = [float(k) for k in np.arange(300, 400, 2.5)]
    new = [k - 15.25 for k in old] + list(range(250, 300, 10))     # plus round strikes
    a = ff.strike_actions(pd.concat([_strikes("2026-10-26", "D", old),
                                     _strikes("2026-10-27", "D", new)])).iloc[0]
    assert a.kind == "subtract" and a.dividend == pytest.approx(15.25) and a.fit == 1.0


def test_relisted_contracts_are_not_given_a_factor():
    old = list(range(560, 850, 10))
    new = list(range(350, 480, 10))                                 # fresh round strikes
    a = ff.strike_actions(pd.concat([_strikes("2026-10-26", "R", old),
                                     _strikes("2026-10-27", "R", new)])).iloc[0]
    assert a.kind == "relisted" and np.isnan(a.adj_factor)


def test_dividend_and_relisted_days_in_buildup():
    e = "2026-12-29"
    f = _fut([("2026-10-26", "DIV", e, 400, 1000), ("2026-10-27", "DIV", e, 388, 1000),
              ("2026-10-26", "REL", e, 700, 1000), ("2026-10-27", "REL", e, 400, 3000)])
    acts = pd.DataFrame({"date": pd.Timestamp("2026-10-27"), "symbol": ["DIV", "REL"],
                         "kind": ["subtract", "relisted"], "adj_factor": [1.0, np.nan],
                         "dividend": [15.0, np.nan]})
    d = ff.oi_buildup(f, actions=acts).set_index(["symbol", "date"])
    assert d.loc[("DIV", "2026-10-27"), "ret"] == pytest.approx((388 + 15) / 400 - 1)
    assert pd.isna(d.loc[("REL", "2026-10-27"), "ret"])
    assert pd.isna(d.loc[("REL", "2026-10-27"), "buildup"])


def test_an_expired_series_is_not_a_restrike():
    old = list(range(800, 1210, 20))
    opts = pd.concat([_strikes("2026-10-26", "X", old, expiry="2026-10-26"),
                      _strikes("2026-10-26", "X", old[:6], expiry="2026-11-24"),
                      _strikes("2026-10-27", "X", old[:6], expiry="2026-11-24")])
    assert ff.strike_actions(opts).empty
