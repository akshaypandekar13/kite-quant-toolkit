"""Derived F&O features from the tables scripts/fno_history.py builds.

    python -m scripts.fno_features            # all three tables
    python -m scripts.fno_features --only oi  # just OI buildup

Writes to data/fno/features:

expiry_iv.csv.gz   one row per (date, underlying, expiry)
    forward    the same-expiry future's close; put-call parity where no
               future exists (index weeklies). `fwd_src` says which.
    atm_iv     IV at log-moneyness 0, interpolated between OTM options
    iv_p25     IV of the 25-delta put, iv_c25 of the 25-delta call
    skew25     iv_p25 - iv_c25 (positive = puts richer, the usual sign)
    oi/contracts by side, all strikes

iv_daily.csv.gz    one row per (date, underlying)
    iv30, skew30  30-calendar-day constant maturity: ATM total variance and
                  skew interpolated in time between the bracketing expiries
                  (expiries < 5 days out are excluded); `iv30_interp` False
                  means no bracket existed and the expiry nearest 30d was used
    iv_front, iv_back  ATM IV of the first two usable expiries
    pcr_oi, pcr_vol    put/call ratios of OI and contracts, all expiries

oi_buildup.csv.gz  one row per (date, underlying) with futures
    ret         close-to-close return of the contract that held the most OI
                the previous session (same contract both days, so a roll
                never registers as a price move)
    oi_chg_pct  change in OI summed over ALL expiries (rollover-neutral)
    buildup     LONG_BUILDUP (ret>0, OI up) / SHORT_BUILDUP (ret<0, OI up) /
                SHORT_COVERING (ret>0, OI down) / LONG_UNWINDING (ret<0,
                OI down) / NEUTRAL. Sign only: no threshold is fitted here.
    corp_action, kind, adj_factor, dividend  from corp_actions (below). On
                a multiply/subtract day `ret` and `oi_chg_pct` are ADJUSTED;
                on a relisted day they are blank (not recoverable)
    near_oi_share, cal_spread_ann (annualised near->next carry)
    deliv_pct, deliv_pct_z20 from the MTO file; z vs the PREVIOUS 20 sessions

corp_actions.csv.gz  (date, symbol, share, kind, adj_factor, dividend, fit)
    NSE re-strikes every live option on a split, bonus, demerger or dividend,
    so yesterday's strikes vanish; in a crash they all stay listed. An action
    is a session where < 50% of yesterday's call strikes (with OI, >= 5 of
    them, the best-populated expiry) survive. `restrike` says how: multiply
    (factor), subtract (dividend D) or relisted. A price-move rule (|ret| >
    20%) was tried first and rejected: it flags real crashes (YESBANK Mar-2020,
    ZEEL Jan-2019, the COVID days) and misses dividends and demergers.

IV CONVENTIONS. Black-76 on the forward, European exercise (NSE stock options
have been European since 2011), T = calendar days / 365, premiums discounted
at a flat RATE. Only options that traded that day (contracts > 0) are used,
priced at their close: NSE's option settle is often a model price. A premium
below intrinsic has no IV and is dropped.

Prices are RAW (see fno_history), which is consistent here: every input comes
from the same bhavcopy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr

logger = logging.getLogger("fno_features")

FNO = Path("data/fno")
OUT = FNO / "features"
RATE = 0.065
MIN_DAYS_CM = 5          # expiries closer than this are excluded from iv30
CM_DAYS = 30
ACTION_MAX_SHARE = 0.5   # surviving share of yesterday's strikes
ACTION_MIN_STRIKES = 5
ACTION_MIN_FIT = 0.8     # share of old strikes the adjustment must map
KEY = ["date", "symbol", "expiry"]


# ── Black-76 ────────────────────────────────────────────────────────────────

def b76_price(F, K, T, sigma, is_call, r=RATE):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    d2 = d1 - sd
    df = np.exp(-r * T)
    call = df * (F * ndtr(d1) - K * ndtr(d2))
    put = df * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def b76_delta(F, K, T, sigma, is_call, r=RATE):
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd ** 2) / sd
    df = np.exp(-r * T)
    return np.where(is_call, df * ndtr(d1), -df * ndtr(-d1))


def implied_vol(price, F, K, T, is_call, r=RATE, lo=1e-3, hi=5.0, iters=60):
    """Vectorised bisection. NaN where the price is outside the no-arbitrage
    band (below discounted intrinsic, or above the discounted forward/strike)."""
    price, F, K, T = (np.asarray(x, float) for x in (price, F, K, T))
    is_call = np.asarray(is_call, bool)
    df = np.exp(-r * T)
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = df * np.where(is_call, F, K)
    ok = (price > intrinsic) & (price < upper) & (T > 0) & (F > 0) & (K > 0)
    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        too_high = b76_price(F, K, T, m, is_call, r) > price
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    iv = 0.5 * (a + b)
    iv[~ok | (iv <= lo * 1.01) | (iv >= hi * 0.99)] = np.nan
    return iv


# ── interpolation within groups ─────────────────────────────────────────────

def interp_at(df: pd.DataFrame, g: str, x: str, y: str, target) -> pd.Series:
    """Per group `g`, linearly interpolate `y` at x == target between the
    nearest point at-or-below and the nearest at-or-above. NaN when the target
    is not bracketed: no extrapolation."""
    d = df[[g, x, y]].dropna().copy()
    d["_t"] = float(target)
    d["_dx"] = d[x] - d["_t"]
    lo = d[d["_dx"] <= 0].sort_values("_dx").groupby(g).tail(1).set_index(g)
    hi = d[d["_dx"] >= 0].sort_values("_dx").groupby(g).head(1).set_index(g)
    j = lo[[x, y, "_t"]].join(hi[[x, y]], rsuffix="_hi", how="inner")
    span = j[x + "_hi"] - j[x]
    w = np.where(span > 0, (j["_t"] - j[x]) / span.replace(0, np.nan), 0.0)
    return pd.Series(j[y] + w * (j[y + "_hi"] - j[y]), index=j.index)


# ── IV tables ───────────────────────────────────────────────────────────────

def forwards(opts: pd.DataFrame, futs: pd.DataFrame) -> pd.DataFrame:
    """Forward per (date, symbol, expiry): the future's close, else parity at
    the strike where |C - P| is smallest among strikes where both traded."""
    fut = futs[KEY + ["close"]].rename(columns={"close": "fut_close"})
    t = opts[opts["contracts"] > 0]
    c = t[t.opt_type == "CE"].set_index(KEY + ["strike"])["close"]
    p = t[t.opt_type == "PE"].set_index(KEY + ["strike"])["close"]
    pair = pd.concat({"c": c, "p": p}, axis=1, join="inner").reset_index()
    T = (pair["expiry"] - pair["date"]).dt.days / 365
    pair["parity"] = pair["strike"] + (pair["c"] - pair["p"]) * np.exp(RATE * T)
    pair["_gap"] = (pair["c"] - pair["p"]).abs()
    par = pair.sort_values("_gap").groupby(KEY).head(1)[KEY + ["parity"]]
    f = par.merge(fut, on=KEY, how="outer")
    f["fut_close"] = f["fut_close"].astype(float)
    f["forward"] = f["fut_close"].fillna(f["parity"]).astype(float)
    f["fwd_src"] = np.where(f["fut_close"].notna(), "future", "parity")
    return f


def expiry_iv(opts: pd.DataFrame, futs: pd.DataFrame) -> pd.DataFrame:
    fw = forwards(opts, futs)
    side = opts.assign(oi_ce=np.where(opts.opt_type == "CE", opts.oi, 0),
                       oi_pe=np.where(opts.opt_type == "PE", opts.oi, 0),
                       vol_ce=np.where(opts.opt_type == "CE", opts.contracts, 0),
                       vol_pe=np.where(opts.opt_type == "PE", opts.contracts, 0))
    agg = side.groupby(KEY)[["oi_ce", "oi_pe", "vol_ce", "vol_pe"]].sum()

    t = opts[(opts["contracts"] > 0) & (opts["close"] > 0)].merge(
        fw[KEY + ["forward"]], on=KEY, how="inner")
    t["days"] = (t["expiry"] - t["date"]).dt.days
    t = t[t["days"] >= 1]
    T = t["days"] / 365
    is_call = (t["opt_type"] == "CE").to_numpy()
    t["iv"] = implied_vol(t["close"], t["forward"], t["strike"], T, is_call)
    t["k"] = np.log(t["strike"] / t["forward"])
    t["delta"] = b76_delta(t["forward"], t["strike"], T, t["iv"], is_call)
    otm = t[((t.opt_type == "PE") & (t.k < 0)) | ((t.opt_type == "CE") & (t.k >= 0))]
    otm = otm.dropna(subset=["iv"]).copy()
    otm["g"] = otm.groupby(KEY, sort=False).ngroup()
    keys = otm.groupby("g")[KEY].first()

    out = keys.copy()
    out["atm_iv"] = interp_at(otm, "g", "k", "iv", 0.0)
    out["iv_p25"] = interp_at(otm[otm.opt_type == "PE"], "g", "delta", "iv", -0.25)
    out["iv_c25"] = interp_at(otm[otm.opt_type == "CE"], "g", "delta", "iv", 0.25)
    out["n_otm"] = otm.groupby("g").size()
    out = out.merge(fw[KEY + ["forward", "fwd_src", "parity", "fut_close"]], on=KEY, how="left")
    out = out.merge(agg.reset_index(), on=KEY, how="left")
    out["days"] = (out["expiry"] - out["date"]).dt.days
    out["skew25"] = out["iv_p25"] - out["iv_c25"]
    return out.sort_values(KEY).reset_index(drop=True)


def iv_daily(ex: pd.DataFrame) -> pd.DataFrame:
    dk = ["date", "symbol"]
    pcr = ex.groupby(dk)[["oi_ce", "oi_pe", "vol_ce", "vol_pe"]].sum()
    pcr["pcr_oi"] = pcr.oi_pe / pcr.oi_ce.replace(0, np.nan)
    pcr["pcr_vol"] = pcr.vol_pe / pcr.vol_ce.replace(0, np.nan)

    u = ex[(ex.days >= MIN_DAYS_CM) & ex.atm_iv.notna()].copy()
    u["T"] = u.days / 365
    u["w"] = u.atm_iv ** 2 * u["T"]
    u["g"] = u.groupby(dk, sort=False).ngroup()
    keys = u.groupby("g")[dk].first()

    w30 = interp_at(u, "g", "days", "w", CM_DAYS)
    s30 = interp_at(u.dropna(subset=["skew25"]), "g", "days", "skew25", CM_DAYS)
    near = u.assign(_d=(u.days - CM_DAYS).abs()).sort_values("_d").groupby("g").head(1).set_index("g")
    out = keys.copy()
    out["iv30_interp"] = w30.reindex(out.index).notna()
    out["iv30"] = np.sqrt(w30 / (CM_DAYS / 365)).reindex(out.index).fillna(near.atm_iv)
    out["skew30"] = s30.reindex(out.index).fillna(near.skew25)
    by_exp = u.sort_values("days").groupby("g")
    out["iv_front"] = by_exp.nth(0).set_index("g").atm_iv
    out["iv_back"] = by_exp.nth(1).set_index("g").atm_iv
    out = out.merge(pcr[["pcr_oi", "pcr_vol"]].reset_index(), on=dk, how="left")
    return out.sort_values(dk).reset_index(drop=True)


# ── corporate actions ───────────────────────────────────────────────────────

def _best_map(old: np.ndarray, new: np.ndarray, cand: np.ndarray, apply, tol) -> tuple:
    """Of the candidate parameters, the one that lands the most old strikes
    on a listed new strike. Returns (parameter, share of old strikes hit)."""
    new = np.sort(new)
    f = apply(old[None, :], cand[:, None])                   # candidates x old
    j = np.clip(np.searchsorted(new, f), 1, len(new) - 1)
    nearest = np.where(np.abs(new[j - 1] - f) <= np.abs(new[j] - f), new[j - 1], new[j])
    hit = np.abs(nearest - f) <= tol(f)
    i = int(hit.sum(axis=1).argmax())
    return i, nearest[i], hit[i]


def restrike(old: np.ndarray, new: np.ndarray) -> tuple:
    """How NSE re-struck an underlying's options: (kind, factor, dividend, fit).

    NSE adjusts every live strike in one of two ways:
      multiply  K -> K x f   splits, bonuses, rights (f = 0.5 for a 1:1 bonus)
      subtract  K -> K - D   dividends (COALINDIA Nov-2023: all 54 strikes
                             moved down by exactly Rs15.25, its interim dividend)
    Every new/old ratio and every old-new difference is a candidate; the true
    one maps nearly every old strike, while a strike-grid coincidence (0.8 =
    800/1000, or Rs10.25 when the true D is Rs15.25 on a Rs2.5 grid) maps only
    some. `fit` is the share of old strikes the winner maps. Below ACTION_MIN_FIT
    the contracts were RELISTED rather than adjusted (TATAMOTORS' Oct-2025
    demerger: strikes 560-840 replaced by fresh round 350-470), and the day's
    economic return cannot be recovered from F&O data.

    Two simpler estimates were tried and rejected: the median-strike ratio is
    biased by strikes listed the same day (0.0962 for TATASTEEL's 10:1), and a
    ratio-only search forced dividends onto factors like 1.008."""
    ratios = np.unique(np.round(np.log(new[None, :] / old[:, None]).ravel(), 3))
    i, near, hit = _best_map(old, new, np.exp(ratios), lambda o, c: o * c, lambda f: 1e-3 * f)
    m_fit = hit.mean()
    m_f = float(np.median(near[hit] / old[hit])) if hit.any() else np.nan
    diffs = np.unique(np.round((old[:, None] - new[None, :]).ravel(), 2))
    diffs = diffs[diffs > 0]
    a_fit, a_d = 0.0, np.nan
    if len(diffs):
        i, near, hit = _best_map(old, new, diffs, lambda o, c: o - c, lambda f: 0.011)
        a_fit, a_d = hit.mean(), float(np.median(old[hit] - near[hit])) if hit.any() else np.nan
    if max(m_fit, a_fit) < ACTION_MIN_FIT:
        return "relisted", np.nan, np.nan, float(max(m_fit, a_fit))
    if a_fit > m_fit:
        return "subtract", 1.0, a_d, float(a_fit)
    return "multiply", m_f, 0.0, float(m_fit)


def strike_actions(opts: pd.DataFrame) -> pd.DataFrame:
    """Sessions where an underlying's option strikes were re-struck. `opts`
    must include the session BEFORE the first one to be scored."""
    c = opts[(opts.opt_type == "CE") & (opts.oi > 0)][KEY + ["strike"]].drop_duplicates()
    s = np.sort(c.date.unique())
    prev = c.assign(date=c.date.map(pd.Series(s[1:], index=s[:-1]))).dropna(subset=["date"])
    m = prev.merge(c.assign(hit=1), on=KEY + ["strike"], how="left")
    g = m.groupby(KEY).agg(n_prev=("strike", "size"), matched=("hit", "sum"))
    # only expiries still listed today: one that expired yesterday has every
    # strike "vanish" and would read as a re-strike
    g = g[g.index.isin(pd.MultiIndex.from_frame(c[KEY].drop_duplicates()))]
    g = g.reset_index().sort_values("n_prev").groupby(["date", "symbol"]).tail(1)
    g["share"] = g.matched / g.n_prev
    g = g[(g.n_prev >= ACTION_MIN_STRIKES) & (g.share < ACTION_MAX_SHARE)].copy()
    cur = c.set_index(KEY).sort_index()
    old = prev.set_index(KEY).sort_index()
    fits = [restrike(old.loc[[k], "strike"].to_numpy(), cur.loc[[k], "strike"].to_numpy())
            for k in zip(g.date, g.symbol, g.expiry)]
    cols = ["kind", "adj_factor", "dividend", "fit"]
    g[cols] = pd.DataFrame(fits, columns=cols, index=g.index) if fits else None
    return g[["date", "symbol", "share"] + cols].sort_values(
        ["date", "symbol"]).reset_index(drop=True)


# ── OI buildup ──────────────────────────────────────────────────────────────

def classify(ret: pd.Series, oi_chg: pd.Series) -> pd.Series:
    cls = np.select(
        [(ret > 0) & (oi_chg > 0), (ret < 0) & (oi_chg > 0),
         (ret > 0) & (oi_chg < 0), (ret < 0) & (oi_chg < 0)],
        ["LONG_BUILDUP", "SHORT_BUILDUP", "SHORT_COVERING", "LONG_UNWINDING"],
        default="NEUTRAL")
    return pd.Series(cls, index=ret.index).where(ret.notna() & oi_chg.notna())


def oi_buildup(futs: pd.DataFrame, delivery: Optional[pd.DataFrame] = None,
               actions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """On an action day a contract's price is scaled by f and its OI in
    shares by 1/f, so the economic return is r/f - 1 and the OI change
    oi * f / prev_oi - 1."""
    f = futs.sort_values(["symbol", "expiry", "date"]).copy()
    sessions = np.sort(f["date"].unique())
    prev_session = pd.Series(sessions[:-1], index=sessions[1:])
    f["prev_session"] = f["date"].map(prev_session)
    g = f.groupby(["symbol", "expiry"])
    f["prev_date"], f["prev_close"], f["prev_oi"] = g.date.shift(), g.close.shift(), g.oi.shift()
    contiguous = f["prev_date"] == f["prev_session"]
    f["ret_c"] = (f["close"] / f["prev_close"] - 1).where(contiguous)
    f["prev_oi"] = f["prev_oi"].where(contiguous)

    dk = ["date", "symbol"]
    active = f.dropna(subset=["prev_oi"]).sort_values("prev_oi").groupby(dk).tail(1)
    d = f.groupby(dk).agg(total_oi=("oi", "sum"), contracts=("contracts", "sum"),
                          value_lakh=("value_lakh", "sum"))
    byexp = f.sort_values("expiry").groupby(dk)
    near, nxt = byexp.nth(0).set_index(dk), byexp.nth(1).set_index(dk)
    d["near_close"], d["near_oi"] = near.close, near.oi
    d["near_oi_share"] = d.near_oi / d.total_oi.replace(0, np.nan)
    yrs = (nxt.expiry - near.expiry.reindex(nxt.index)).dt.days / 365
    d["cal_spread_ann"] = np.log(nxt.close / near.close.reindex(nxt.index)) / yrs
    d["ret"] = active.set_index(dk)["ret_c"]
    d["_prev_px"] = active.set_index(dk)["prev_close"]
    d = d.reset_index().sort_values(["symbol", "date"])

    d["prev_session"] = d["date"].map(prev_session)
    gs = d.groupby("symbol")
    prev_tot, prev_date = gs.total_oi.shift(), gs.date.shift()
    d["oi_chg_pct"] = (d.total_oi / prev_tot - 1).where(prev_date == d.prev_session)
    acts = actions if actions is not None and len(actions) else pd.DataFrame(
        columns=dk + ["kind", "adj_factor", "dividend"])
    d = d.merge(acts[dk + ["kind", "adj_factor", "dividend"]], on=dk, how="left")
    d["corp_action"] = d.kind.notna()
    d["adj_factor"] = d.adj_factor.astype(float).fillna(1.0)
    d["dividend"] = d.dividend.astype(float).fillna(0.0)
    # multiply: price x f, OI(shares) x 1/f. subtract: the contract's base
    # price is cut by D, so a holder's P&L is close - (prev - D).
    d["ret"] = (1 + d["ret"]) / d.adj_factor + d.dividend / d._prev_px - 1
    d["oi_chg_pct"] = (1 + d["oi_chg_pct"]) * d.adj_factor - 1
    relisted = d.kind == "relisted"
    d.loc[relisted, ["ret", "oi_chg_pct"]] = np.nan
    d["buildup"] = classify(d["ret"], d["oi_chg_pct"])

    if delivery is not None:
        eq = delivery[delivery.series == "EQ"][["date", "symbol", "deliv_pct", "traded_qty"]]
        d = d.merge(eq, on=dk, how="left").sort_values(["symbol", "date"])
        prior = d.groupby("symbol").deliv_pct.shift().groupby(d.symbol)
        mu = prior.transform(lambda s: s.rolling(20, min_periods=15).mean())
        sd = prior.transform(lambda s: s.rolling(20, min_periods=15).std())
        d["deliv_pct_z20"] = (d.deliv_pct - mu) / sd.replace(0, np.nan)
    return d.drop(columns=["prev_session", "_prev_px"]).reset_index(drop=True)


# ── driver ──────────────────────────────────────────────────────────────────

def _read(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[c for c in ("date", "expiry")
                                          if c in pd.read_csv(path, nrows=0).columns], **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fno", default=str(FNO))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", choices=["iv", "oi"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    fno, out = Path(a.fno), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    futs = _read(fno / "futures.csv.gz")

    # One pass over the option years: IV, and corporate actions (which need
    # the previous year's last session to score each year's first).
    parts, acts, tail = [], [], None
    for p in sorted((fno / "options").glob("options_*.csv.gz")):
        opts = _read(p)
        acts.append(strike_actions(opts if tail is None else pd.concat([tail, opts])))
        tail = opts[opts.date == opts.date.max()]
        if a.only != "oi":
            ex = expiry_iv(opts, futs)
            parts.append(ex)
            logger.info(f"{p.name}: {len(ex):,} expiry rows, "
                        f"atm_iv filled {ex.atm_iv.notna().mean():.0%}")
    actions = pd.concat(acts, ignore_index=True)
    actions.to_csv(out / "corp_actions.csv.gz", index=False)
    logger.info(f"corp_actions: {len(actions):,}")

    if a.only != "iv":
        deliv = _read(fno / "delivery.csv.gz")
        oi = oi_buildup(futs, deliv, actions)
        oi.to_csv(out / "oi_buildup.csv.gz", index=False)
        logger.info(f"oi_buildup: {len(oi):,} rows")

    if a.only != "oi":
        ex = pd.concat(parts, ignore_index=True)
        ex.to_csv(out / "expiry_iv.csv.gz", index=False)
        daily = iv_daily(ex)
        daily.to_csv(out / "iv_daily.csv.gz", index=False)
        logger.info(f"expiry_iv {len(ex):,} rows, iv_daily {len(daily):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
