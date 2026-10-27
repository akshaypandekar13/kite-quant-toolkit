"""
The docs/MULTI_ASSET.md test: S1 GTAA, S2 DUAL, S3 SECTOR, S5 ML vs B1/B2.

    python -m scripts.multi_asset_test [--cash 0.06]

Reads data/multi (scripts/multi_asset_history.py). Writes
structural/multi/{summary.csv,monthly.csv,weights_<S>.csv}.

Mechanics (frozen in the pre-registration):
- decide at the close of each month's last session, trade at the next open;
- costs per side on traded value; cash accrues CASH a year;
- index/sector proxies carry a 0.40%/yr expense drag.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "multi"
OUT = ROOT / "structural" / "multi"

CASH = 0.06
DRAG = 0.004
COST_ETF, COST_IDX = 0.0010, 0.0020
STAMP, EXCH, STT_ETF = 0.00015, 0.0000297, 0.00001
DEV_END = pd.Period("2021-12", "M")           # last development DECISION month
FIRST_DECISION = pd.Period("2011-10", "M")
LAST_DECISION = pd.Period("2026-09", "M")
T_PASS = 2.5

# name -> (sources oldest first, liquid ETF?, equity-ETF STT applies?)
ASSETS = {
    "LARGE": ([("etf", "NIFTYBEES")], True, True),
    "MID": ([("index", "NIFTY_MIDCAP_100")], False, True),
    "SMALL": ([("index", "NIFTY_SMLCAP_100")], False, True),
    "GOLD": ([("mcx", "GOLD"), ("etf", "GOLDBEES")], True, False),
    "SILVER": ([("mcx", "SILVER"), ("etf", "SILVERBEES")], True, False),
    "US": ([("etf", "MON100")], True, False),
    "GILT": ([("etf", "LTGILTBEES")], True, False),
}
SECTORS = ["NIFTY_BANK", "NIFTY_IT", "NIFTY_PHARMA", "NIFTY_FMCG", "NIFTY_METAL",
           "NIFTY_ENERGY", "NIFTY_AUTO", "NIFTY_REALTY", "NIFTY_INFRA",
           "NIFTY_PSU_BANK", "NIFTY_FIN_SERVICE", "NIFTY_MEDIA"]
S1_SET = list(ASSETS)
S2_SET = [a for a in ASSETS if a != "GILT"]


# ----------------------------------------------------------------- data

def _read(kind: str, sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / kind / f"{sym}.csv", parse_dates=["date"])
    return df.set_index("date")[["open", "close"]].astype(float)


def splice(parts: list[pd.DataFrame], cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Chain-link sources oldest first: each newer source replaces the older
    one from its first date; the older one is rescaled to meet it there."""
    out = parts[0].reindex(cal)
    for p in parts[1:]:
        p = p.reindex(cal)
        first = p["close"].first_valid_index()
        if first is None:
            continue
        old = out.loc[first, "close"]
        if np.isfinite(old) and old > 0:
            out = out * (p.loc[first, "close"] / old)
        out.loc[first:] = p.loc[first:]
    # a missing day inside a series (exchange holiday, or a gap in Kite's
    # history: NIFTY IT 2013-05, NIFTY INFRA 2018-12..2019-02) carries the last
    # close, so a holding earns nothing through the gap and the move on return
    # (amendment A1)
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].fillna(out["close"])
    start = parts[0]["close"].first_valid_index()
    out.loc[out.index < start] = np.nan
    return out


class Market:
    def __init__(self, prices: dict[str, pd.DataFrame], meta: dict[str, tuple[bool, bool]],
                 cal: pd.DatetimeIndex, vix: pd.Series | None = None):
        self.cal, self.meta, self.vix = cal, meta, vix
        self.open = pd.DataFrame({k: v["open"] for k, v in prices.items()}).reindex(cal)
        self.close = pd.DataFrame({k: v["close"] for k, v in prices.items()}).reindex(cal)
        m = pd.Series(cal, index=cal).groupby(cal.to_period("M")).last()
        self.month_end = pd.DatetimeIndex(m.values)          # last session of each month
        self.mclose = self.close.reindex(self.month_end)
        self.mclose.index = self.month_end.to_period("M")

    @classmethod
    def load(cls) -> "Market":
        cal = _read("index", "NIFTY_50").index
        cal = cal[cal.dayofweek < 5]
        prices, meta = {}, {}
        for a, (srcs, liquid, stt) in ASSETS.items():
            prices[a] = splice([_read(k, s) for k, s in srcs], cal)
            meta[a] = (liquid, stt)
        for s in SECTORS + ["NIFTY_50"]:
            prices[s] = splice([_read("index", s)], cal)
            meta[s] = (False, True)
        vix = pd.read_csv(DATA / "index" / "INDIA_VIX.csv", parse_dates=["date"]).set_index("date")["close"]
        return cls(prices, meta, cal, vix.reindex(cal).ffill())

    def eligible(self, names, m: pd.Period) -> list[str]:
        """Names with a close at month-end m and at m-12."""
        names = [n for n in names if n in self.mclose.columns]
        h = self.mclose.loc[:m, names]
        if len(h) < 13:
            return []
        return [n for n in names if np.isfinite(h[n].iloc[-1]) and np.isfinite(h[n].iloc[-13])]

    def ret(self, names, m: pd.Period, months: int) -> pd.Series:
        h = self.mclose.loc[:m, names]
        return h.iloc[-1] / h.iloc[-1 - months] - 1


# ----------------------------------------------------------------- rules

def s1_gtaa(mk: Market, m: pd.Period) -> dict[str, float]:
    el = mk.eligible(S1_SET, m)
    if not el:
        return {}
    h = mk.mclose.loc[:m, el]
    sma = h.iloc[-10:].mean()
    w = 1.0 / len(el)
    return {a: w for a in el if h[a].iloc[-1] > sma[a]}


def s2_dual(mk: Market, m: pd.Period, cash: float) -> dict[str, float]:
    el = mk.eligible(S2_SET, m)
    if not el:
        return {}
    r12 = mk.ret(el, m, 12).sort_values(ascending=False)
    safe = "GILT" if mk.eligible(["GILT"], m) else None
    w: dict[str, float] = {}
    for a in r12.index[:2]:
        tgt = a if r12[a] > cash else safe
        if tgt:
            w[tgt] = w.get(tgt, 0.0) + 0.5
    return w


def s3_sector(mk: Market, m: pd.Period) -> dict[str, float]:
    el = mk.eligible(SECTORS, m)
    if len(el) < 3:
        return {}
    top = mk.ret(el, m, 6).sort_values(ascending=False).index[:3]
    return {s: 1 / 3 for s in top}


def b2_equal(mk: Market, m: pd.Period) -> dict[str, float]:
    el = mk.eligible(S1_SET, m)
    return {a: 1 / len(el) for a in el} if el else {}


def b1_large(mk: Market, m: pd.Period) -> dict[str, float]:
    return {"LARGE": 1.0}


def b_nifty(mk: Market, m: pd.Period) -> dict[str, float]:
    return {"NIFTY_50": 1.0}


# ----------------------------------------------------------------- ML

ML_SET = S1_SET + SECTORS
MIN_LABEL_MONTHS = 36


def features(mk: Market, m: pd.Period) -> pd.DataFrame:
    el = mk.eligible(ML_SET, m)
    if not el:
        return pd.DataFrame()
    h = mk.mclose.loc[:m, el]
    end = mk.month_end[mk.month_end.to_period("M") == m][0]
    d = mk.close.loc[:end, el].iloc[-64:]
    f = pd.DataFrame(index=el)
    for k in (1, 3, 6, 12):
        f[f"r{k}"] = h.iloc[-1] / h.iloc[-1 - k] - 1
    f["vol3"] = d.pct_change().std() * math.sqrt(252)
    f["sma10"] = h.iloc[-1] / h.iloc[-10:].mean() - 1
    y = h.iloc[-13:]
    f["mdd12"] = (y / y.cummax() - 1).min()
    v = mk.vix.loc[:end]
    f["vix"] = v.iloc[-1]
    f["vix_chg"] = v.iloc[-1] / v.iloc[-22] - 1 if len(v) > 22 else np.nan
    return f


def label(mk: Market, m: pd.Period, cash: float) -> pd.Series:
    """Next month's close-to-close return minus cash; known at the close of m+1."""
    nxt = m + 1
    if nxt not in mk.mclose.index:
        return pd.Series(dtype=float)
    r = mk.mclose.loc[nxt] / mk.mclose.loc[m] - 1
    return r - ((1 + cash) ** (1 / 12) - 1)


class MLRule:
    def __init__(self, mk: Market, cash: float):
        self.mk, self.cash = mk, cash
        self.rows: dict[pd.Period, pd.DataFrame] = {}
        self.started: pd.Period | None = None

    def _row(self, m):
        if m not in self.rows:
            f = features(self.mk, m)
            if not f.empty:
                f = f.assign(y=label(self.mk, m, self.cash).reindex(f.index))
            self.rows[m] = f
        return self.rows[m]

    def __call__(self, mk: Market, m: pd.Period) -> dict[str, float] | None:
        from sklearn.ensemble import GradientBoostingRegressor
        # training months: every decision month m' <= m-1 (its label ended at the close of m'+1 <= m)
        past = [p for p in mk.mclose.index if p <= m - 1]
        tr = pd.concat([self._row(p) for p in past if not self._row(p).empty], ignore_index=False) \
            if past else pd.DataFrame()
        tr = tr.dropna() if not tr.empty else tr
        n_months = len({p for p in past if not self._row(p).empty and self._row(p)["y"].notna().any()})
        if tr.empty or n_months < MIN_LABEL_MONTHS:
            return None                                       # not trading yet
        X = tr.drop(columns="y")
        mdl = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.05,
                                        subsample=0.8, random_state=0).fit(X.values, tr["y"].values)
        f = features(mk, m).dropna()
        if f.empty:
            return {}
        if self.started is None:
            self.started = m
        p = pd.Series(mdl.predict(f[X.columns].values), index=f.index).sort_values(ascending=False)
        top = [n for n in p.index[:3] if p[n] > 0]
        return {n: 1 / 3 for n in top}


# ----------------------------------------------------------------- engine

def side_cost(mk: Market, name: str, buy: bool) -> float:
    liquid, stt = mk.meta[name]
    c = (COST_ETF if liquid else COST_IDX) + EXCH * 1.18
    return c + (STAMP if buy else (STT_ETF if stt else 0.0))


def run(mk: Market, rule, cash: float = CASH, first=FIRST_DECISION, last=LAST_DECISION):
    """Daily simulation. Returns (daily equity Series, weights DataFrame by
    decision month, turnover Series by decision month, realised gains by year)."""
    cal = mk.cal
    decisions = [p for p in mk.mclose.index if first <= p <= last]
    trade_day = {}
    for p in decisions:
        after = cal[cal > mk.month_end[mk.month_end.to_period("M") == p][0]]
        if len(after):
            trade_day[after[0]] = p
    rc = (1 + cash) ** (1 / 252) - 1
    drag = {n: (DRAG / 252 if not mk.meta[n][0] else 0.0) for n in mk.meta}
    hold: dict[str, float] = {}           # value per asset
    basis: dict[str, float] = {}          # cost basis per asset (average cost, in value terms)
    cash_v, eq, out, wrec, turn, gains = 1.0, 1.0, {}, {}, {}, {}
    started = False
    prev = None
    for d in cal:
        if prev is not None and started:
            # overnight / intraday: assets move close(prev) -> open(d) -> close(d)
            if d in trade_day:
                for n in hold:
                    hold[n] *= mk.open.at[d, n] / mk.close.at[prev, n]
            else:
                for n in hold:
                    hold[n] *= mk.close.at[d, n] / mk.close.at[prev, n] * (1 - drag[n])
                cash_v *= 1 + rc
        if d in trade_day:
            p = trade_day[d]
            w = rule(mk, p)                    # None: the rule is not trading yet
            if w is not None:
                started = True
                total = cash_v + sum(hold.values())
                tgt = {n: total * x for n, x in w.items()}
                traded = 0.0
                for n in set(hold) | set(tgt):
                    cur, new = hold.get(n, 0.0), tgt.get(n, 0.0)
                    dv = new - cur
                    if abs(dv) < 1e-12:
                        continue
                    traded += abs(dv)
                    fee = abs(dv) * side_cost(mk, n, dv > 0)
                    cash_v -= dv + fee
                    if dv < 0 and cur > 0:                         # realise on the sold fraction
                        frac = -dv / cur
                        g = -dv - basis.get(n, cur) * frac
                        gains[d.year] = gains.get(d.year, 0.0) + g
                        basis[n] = basis.get(n, cur) * (1 - frac)
                    elif dv > 0:
                        basis[n] = basis.get(n, 0.0) + dv
                    hold[n] = new
                hold = {n: v for n, v in hold.items() if v > 1e-12}
                basis = {n: v for n, v in basis.items() if n in hold}
                wrec[p] = {n: v / total for n, v in hold.items()}
                turn[p] = traded / total
            if started:                                            # intraday leg open -> close
                for n in hold:
                    hold[n] *= mk.close.at[d, n] / mk.open.at[d, n] * (1 - drag[n])
                cash_v *= 1 + rc
        if started:
            out[d] = cash_v + sum(hold.values())
        prev = d
    eqs = pd.Series(out)
    return eqs, pd.DataFrame(wrec).T.fillna(0.0), pd.Series(turn), pd.Series(gains)


# ----------------------------------------------------------------- scoring

def monthly(eq: pd.Series) -> pd.Series:
    m = eq.groupby(eq.index.to_period("M")).last()
    return m.pct_change().dropna()


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1).min())


def stats(r: pd.Series, cash: float) -> dict:
    yrs = len(r) / 12
    cagr = (1 + r).prod() ** (1 / yrs) - 1 if yrs > 0 else np.nan
    ex = r - ((1 + cash) ** (1 / 12) - 1)
    sharpe = ex.mean() / ex.std() * math.sqrt(12) if ex.std() > 0 else np.nan
    eq = (1 + r).cumprod()
    return {"months": len(r), "cagr": cagr, "vol": r.std() * math.sqrt(12),
            "sharpe": sharpe, "max_dd": max_dd(pd.concat([pd.Series([1.0]), eq]))}


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def verdict(r: pd.Series, b: pd.Series, eq_r_dd: float, eq_b_dd: float) -> dict:
    j = pd.concat([r.rename("s"), b.rename("b")], axis=1).dropna()
    diff = j["s"] - j["b"]
    hold = diff[diff.index > DEV_END + 1]
    out = {"excess_m": diff.mean(), "t": tstat(diff),
           "dev_excess_m": diff[diff.index <= DEV_END + 1].mean(),
           "hold_excess_m": hold.mean(), "hold_months": len(hold)}
    out["P1"] = bool(out["t"] >= T_PASS)
    out["P2"] = bool(out["hold_excess_m"] > 0)
    out["P3"] = bool(eq_r_dd >= eq_b_dd)
    out["PASS"] = out["P1"] and out["P2"] and out["P3"]
    return out


def after_tax_cagr(eq: pd.Series, gains: pd.Series, buy_hold: bool) -> float:
    """Rough: realised gains taxed 20% in the year (losses offset within the year);
    buy-and-hold taxed 12.5% on the final gain. An estimate, not tax advice."""
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if buy_hold:
        g = eq.iloc[-1] / eq.iloc[0] - 1
        end = 1 + g - 0.125 * max(g, 0.0)
        return end ** (1 / yrs) - 1
    tax = (gains.clip(lower=0) * 0.20)
    # tax paid reduces capital in proportion to equity at each year end
    ye = eq.groupby(eq.index.year).last()
    mult = 1.0
    for y, t in tax.items():
        if y in ye.index and ye[y] > 0:
            mult *= 1 - t / ye[y]
    end = eq.iloc[-1] / eq.iloc[0] * mult
    return end ** (1 / yrs) - 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cash", type=float, default=CASH)
    a = ap.parse_args(argv)
    mk = Market.load()
    cash = a.cash
    rules = {
        "B1_LARGE": (b1_large, None), "B2_EQUAL": (b2_equal, None), "B_NIFTY": (b_nifty, None),
        "S1_GTAA": (s1_gtaa, "B2_EQUAL"), "S2_DUAL": (lambda mk, m: s2_dual(mk, m, cash), "B2_EQUAL"),
        "S3_SECTOR": (s3_sector, "B_NIFTY"), "S5_ML": (MLRule(mk, cash), "B2_EQUAL"),
    }
    res, rows, mon = {}, [], {}
    for name, (rule, _) in rules.items():
        eq, w, turn, gains = run(mk, rule, cash)
        res[name] = (eq, w, turn, gains)
        mon[name] = monthly(eq)
        print(f"ran {name}: {len(eq)} days from {eq.index[0].date()}", file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (rule, bench) in rules.items():
        eq, w, turn, gains = res[name]
        r = mon[name]
        row = {"strategy": name, "start": str(r.index[0]), **stats(r, cash),
               "turnover_yr": turn.sum() / (len(r) / 12),
               "after_tax_cagr": after_tax_cagr(eq, gains, name.startswith("B"))}
        if bench:
            b = mon[bench].reindex(r.index)
            beq = (1 + b.dropna()).cumprod()
            row.update(verdict(r, b, row["max_dd"],
                               max_dd(pd.concat([pd.Series([1.0]), beq]))))
            row["bench"] = bench
        for per, sl in (("dev", r.index <= DEV_END + 1), ("hold", r.index > DEV_END + 1)):
            s = stats(r[sl], cash)
            row[f"{per}_cagr"], row[f"{per}_sharpe"], row[f"{per}_dd"] = s["cagr"], s["sharpe"], s["max_dd"]
        rows.append(row)
        w.to_csv(OUT / f"weights_{name}.csv")
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(mon).to_csv(OUT / "monthly.csv")
    pd.set_option("display.width", 250)
    show = ["strategy", "start", "cagr", "sharpe", "max_dd", "dev_cagr", "hold_cagr", "hold_dd",
            "turnover_yr", "after_tax_cagr", "bench", "excess_m", "t", "hold_excess_m", "P1", "P2", "P3", "PASS"]
    print(summ[[c for c in show if c in summ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
