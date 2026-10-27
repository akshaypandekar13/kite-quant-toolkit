"""
docs/MULTI_ASSET.md round 3, D1: NIFTY calendar effects.

    python -m scripts.calendar_test

D1a TOM          hold NIFTYBEES from the close of a month's second-last session
                 to the close of the next month's third session (days -1..+3).
D1b PRE-HOLIDAY  hold NIFTYBEES over the session before an exchange holiday
                 (close of the session before it -> its close).
Cash earns CASH a year whenever nothing is held. The effect itself is also
measured on NIFTY 50 index daily returns (window days vs all other days).

Writes structural/multi/r3_calendar_{summary,trades}.csv.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from scripts import multi_asset_test as mt

HOLD_START = pd.Timestamp("2019-01-01")     # holdout: trades ENTERED on/after this date
LAST_DAY = pd.Timestamp("2026-09-30")
T_PASS = 2.5
YEARS_FRAC = 0.6
SPREAD = mt.COST_ETF                         # ASSUMED per side, as rounds 1-2
DP_RS = 15.34 * 1.18                         # Zerodha DP charge per sell, incl GST
CAPITAL = 250_000.0


def holidays(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Weekdays inside the calendar's span on which there was no session."""
    wk = pd.bdate_range(cal[0], cal[-1])
    return wk.difference(cal)


def tom_trades(cal: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(entry close, exit close) per month: second-last session of month m ->
    third session of month m+1. Only months with both ends inside cal."""
    per = cal.to_period("M")
    out = []
    months = per.unique()
    for m, nxt in zip(months[:-1], months[1:]):
        cur, after = cal[per == m], cal[per == nxt]
        if nxt != m + 1 or len(cur) < 2 or len(after) < 3:
            continue
        out.append((cur[-2], after[2]))
    return out


def preholiday_trades(cal: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    hol = set(holidays(cal))
    out = []
    for i in range(1, len(cal)):
        d = cal[i]
        nxt = d + pd.offsets.BDay(1)
        if nxt in hol:
            out.append((cal[i - 1], d))
    return out


def trade_cost(buy: bool) -> float:
    c = SPREAD + mt.EXCH * 1.18
    return c + (mt.STAMP if buy else mt.STT_ETF + DP_RS / CAPITAL)


def score_trades(close: pd.Series, trades, cash: float) -> pd.DataFrame:
    rc = (1 + cash) ** (1 / 252) - 1
    rows = []
    pos = {d: i for i, d in enumerate(close.index)}
    for e, x in trades:
        if e not in pos or x not in pos:
            continue
        ce, cx = close.at[e], close.at[x]
        if not (np.isfinite(ce) and np.isfinite(cx)):
            continue
        gross = cx / ce - 1
        net = (1 - trade_cost(True)) * (1 + gross) * (1 - trade_cost(False)) - 1
        days = pos[x] - pos[e]
        rows.append({"entry": e, "exit": x, "days": days, "gross": gross, "net": net,
                     "cash": (1 + rc) ** days - 1})
    t = pd.DataFrame(rows)
    t["excess"] = t["net"] - t["cash"]
    return t


def equity(close: pd.Series, trades: pd.DataFrame, cash: float) -> pd.Series:
    """Daily equity: in the ETF on held sessions (close->close), cash otherwise;
    the trade's two cost sides charged on entry and exit days."""
    rc = (1 + cash) ** (1 / 252) - 1
    r = pd.Series(rc, index=close.index)
    held = pd.Series(False, index=close.index)
    etf = close.pct_change(fill_method=None)
    for _, t in trades.iterrows():
        sl = (close.index > t["entry"]) & (close.index <= t["exit"])
        held[sl] = True
    r[held] = etf[held]
    eq = (1 + r).cumprod()
    for _, t in trades.iterrows():
        eq[eq.index >= t["entry"]] *= 1 - trade_cost(True)
        eq[eq.index >= t["exit"]] *= 1 - trade_cost(False)
    return eq


def effect(idx_close: pd.Series, trades) -> dict:
    """Index daily return on window sessions (entry, exit] vs every other session."""
    r = idx_close.pct_change(fill_method=None).dropna()
    win = pd.Series(False, index=r.index)
    for e, x in trades:
        win[(r.index > e) & (r.index <= x)] = True
    a, b = r[win], r[~win]
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"win_days": len(a), "win_mean_bps": a.mean() * 1e4, "other_mean_bps": b.mean() * 1e4,
            "effect_t": (a.mean() - b.mean()) / se}


def verdict(t: pd.DataFrame) -> dict:
    hold = t[t["entry"] >= HOLD_START]
    yrs = t.groupby(t["entry"].dt.year)["excess"].sum()
    out = {"trades": len(t), "net_bps": t["net"].mean() * 1e4, "excess_bps": t["excess"].mean() * 1e4,
           "t": mt.tstat(t["excess"]), "dev_excess_bps": t.loc[t["entry"] < HOLD_START, "excess"].mean() * 1e4,
           "hold_excess_bps": hold["excess"].mean() * 1e4, "hold_trades": len(hold),
           "years_pos": int((yrs > 0).sum()), "years": len(yrs)}
    out["P1"] = bool(out["t"] >= T_PASS)
    out["P2"] = bool(out["hold_excess_bps"] > 0)
    out["P3"] = bool(out["years_pos"] >= YEARS_FRAC * out["years"])
    out["PASS"] = out["P1"] and out["P2"] and out["P3"]
    return out


def main(argv=None) -> int:
    idx = mt._read("index", "NIFTY_50")
    cal = idx.index[(idx.index.dayofweek < 5) & (idx.index <= LAST_DAY)]
    idx_c = idx["close"].reindex(cal)
    etf = mt._read("etf", "NIFTYBEES")["close"].reindex(cal).ffill()
    rows, alltr = [], []
    bh = etf / etf.iloc[0]
    for name, tr in (("D1a_TOM", tom_trades(cal)), ("D1b_PREHOLIDAY", preholiday_trades(cal))):
        t = score_trades(etf, tr, mt.CASH)
        eq = equity(etf, t, mt.CASH)
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        row = {"strategy": name, **effect(idx_c, tr), **verdict(t),
               "cagr": (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
               "max_dd": mt.max_dd(eq), "time_in_mkt": t["days"].sum() / len(cal),
               "bh_cagr": (bh.iloc[-1]) ** (1 / yrs) - 1, "bh_dd": mt.max_dd(bh)}
        rows.append(row)
        alltr.append(t.assign(strategy=name))
    s = pd.DataFrame(rows)
    mt.OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(mt.OUT / "r3_calendar_summary.csv", index=False)
    pd.concat(alltr).to_csv(mt.OUT / "r3_calendar_trades.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
