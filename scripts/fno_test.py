"""F&O long-short: the test frozen in docs/FNO_PREREG.md. Paper only.

    python -m scripts.fno_test                  # development run (once)
    python -m scripts.fno_test --holdout H1     # ONLY for a hypothesis that passed

Every rule here is a section of the pre-registration; the section number is
cited beside the code that implements it. Changing a rule means a new
pre-registration, not an edit here (FNO_PREREG §11).

Writes to structural/fno: one CSV of cycle returns per hypothesis, the
sensitivity grid, and summary.txt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._vehicles import _FUT  # noqa: E402

logger = logging.getLogger("fno_test")

FNO = Path("data/fno")
OUT = Path("structural/fno")

# §1
DEV_START = pd.Timestamp("2019-02-01")
HOLDOUT_START = pd.Timestamp("2025-10-01")
EXIT_SESSIONS_BEFORE_EXPIRY = 5
# §2
TOP_N = 100
VALUE_WINDOW = 20
MIN_HISTORY = 250
SIGNAL_WINDOW = 5
SIGNAL_MIN_OBS = 3
# §3
SPREAD_BPS_SIDE = 5.0
PRE_LOT_NOTIONAL = 620_000.0
# §4-5
N_QUANTILES = 5
# §6
FORMATION = 250
N_PAIRS = 20
OPEN_SIGMA = 2.0
# §8
T_PASS = 2.4
YEARS = list(range(2019, 2026))
MIN_POSITIVE_YEARS = 5


class HoldoutError(RuntimeError):
    pass


# ── data ────────────────────────────────────────────────────────────────────

@dataclass
class Data:
    sessions: pd.DatetimeIndex
    expiries: List[pd.Timestamp]
    closes: Dict[Tuple[str, pd.Timestamp], pd.Series]     # (symbol, expiry) -> close by date
    lots: Dict[Tuple[str, pd.Timestamp], float]           # lot at the contract's first row
    value: pd.DataFrame                                    # date x symbol, FUTSTK value_lakh
    ret: pd.DataFrame                                      # date x symbol, adjusted daily ret
    actions: Dict[Tuple[pd.Timestamp, str], Tuple[str, float, float]]
    skew: pd.DataFrame                                     # date x symbol, skew30
    parity: pd.Series                                      # (date, symbol, expiry) -> dev
    nifty: pd.Series                                       # date -> NIFTY adjusted ret

    @classmethod
    def build(cls, futs: pd.DataFrame, actions: pd.DataFrame, oi: pd.DataFrame,
              ivd: pd.DataFrame, exiv: pd.DataFrame) -> "Data":
        alias = expiry_aliases(futs[futs.instrument == "FUTSTK"])
        futs = futs.assign(expiry=futs.expiry.replace(alias))
        exiv = exiv.assign(expiry=exiv.expiry.replace(alias))
        stk = futs[futs.instrument == "FUTSTK"]
        sessions = pd.DatetimeIndex(np.sort(futs.date.unique()))
        expiries = sorted(e for e in stk.expiry.unique() if e <= sessions[-1])
        closes, lots = {}, {}
        for (sym, exp), g in stk.sort_values("date").groupby(["symbol", "expiry"]):
            closes[(sym, exp)] = g.set_index("date")["close"]
            lot = g["lot"].dropna()
            lots[(sym, exp)] = float(lot.iloc[0]) if len(lot) else np.nan
        value = stk.groupby(["date", "symbol"]).value_lakh.sum().unstack()
        stocks = set(stk.symbol)
        o = oi[oi.symbol.isin(stocks)]
        ret = o.pivot(index="date", columns="symbol", values="ret").reindex(sessions)
        acts = {(r.date, r.symbol): (r.kind, r.adj_factor, r.dividend)
                for r in actions.itertuples(index=False)}
        skew = ivd.pivot(index="date", columns="symbol", values="skew30")
        e = exiv.dropna(subset=["parity", "fut_close"])
        parity = (e.parity / e.fut_close - 1).set_axis(
            pd.MultiIndex.from_frame(e[["date", "symbol", "expiry"]]))
        nifty = oi[oi.symbol == "NIFTY"].set_index("date")["ret"].reindex(sessions)
        return cls(sessions, expiries, closes, lots, value, ret, acts, skew, parity, nifty)


# ── calendar (§1) ───────────────────────────────────────────────────────────

def expiry_aliases(stk: pd.DataFrame, max_gap_days: int = 7) -> Dict:
    """NSE sometimes moves an expiry mid-contract for a holiday, and the
    bhavcopy then carries ONE contract under two dates: the June-2023 series
    is labelled 2023-06-29 until 06-27 and 2023-06-28 on its last day. Two
    expiry dates this close are one contract; the one that actually traded on
    the final day is the real expiry, and the other is mapped onto it."""
    last = stk.groupby("expiry").date.max()
    e = sorted(last.index)
    out = {}
    for a, b in zip(e[:-1], e[1:]):
        if (b - a).days <= max_gap_days:
            real, other = (a, b) if last[a] >= last[b] else (b, a)
            out[other] = real
    return out


@dataclass(frozen=True)
class Cycle:
    expiry: pd.Timestamp        # E_k
    next_expiry: pd.Timestamp   # E_{k+1}: the traded contract
    signal: pd.Timestamp        # s_k
    entry: pd.Timestamp         # s_k + 1 session
    exit: pd.Timestamp          # 5th session before E_{k+1}


def cycles(sessions: pd.DatetimeIndex, expiries: List[pd.Timestamp]) -> List[Cycle]:
    out = []
    for e0, e1 in zip(expiries[:-1], expiries[1:]):
        i = sessions.searchsorted(e0, side="right")          # first session > E_k
        j = sessions.searchsorted(e1) - EXIT_SESSIONS_BEFORE_EXPIRY
        if i + 1 >= len(sessions) or e1 not in sessions or j <= i + 1:
            continue
        out.append(Cycle(e0, e1, sessions[i], sessions[i + 1], sessions[j]))
    return out


def period(cs: List[Cycle], holdout: bool) -> List[Cycle]:
    if holdout:
        return [c for c in cs if c.entry >= HOLDOUT_START]
    return [c for c in cs if DEV_START <= c.entry < HOLDOUT_START]


def guard(c: Cycle, allow_holdout: bool) -> None:
    if c.entry >= HOLDOUT_START and not allow_holdout:
        raise HoldoutError(f"cycle entering {c.entry.date()} is in the holdout")


# ── universe (§2) ───────────────────────────────────────────────────────────

def _window(sessions: pd.DatetimeIndex, end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    i = sessions.searchsorted(end, side="right")
    return sessions[max(0, i - n):i]


def universe(d: Data, c: Cycle) -> List[str]:
    win = _window(d.sessions, c.signal, VALUE_WINDOW)
    med = d.value.reindex(win).median()
    top = set(med.dropna().nlargest(TOP_N).index)
    hist = d.ret.loc[d.ret.index < c.signal].notna().sum()
    out = []
    for sym in sorted(top):
        s = d.closes.get((sym, c.next_expiry))
        if s is None or c.signal not in s.index or c.entry not in s.index:
            continue
        if hist.get(sym, 0) < MIN_HISTORY:
            continue
        out.append(sym)
    return out


# ── returns and costs (§3) ──────────────────────────────────────────────────

def leg_return(d: Data, sym: str, expiry: pd.Timestamp,
               start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Compounded close-to-close return of one contract, adjusted for
    corporate actions; a relisted day closes the leg at the previous close."""
    s = d.closes[(sym, expiry)]
    s = s[(s.index >= start) & (s.index <= end)]
    growth = 1.0
    for (t0, p0), (t1, p1) in zip(s.items(), list(s.items())[1:]):
        act = d.actions.get((t1, sym))
        if act is None:
            growth *= p1 / p0
        elif act[0] == "relisted":
            break
        elif act[0] == "multiply":
            growth *= (p1 / p0) / act[1]
        else:                                                # subtract
            growth *= p1 / p0 + act[2] / p0
    return growth - 1


def cost_frac(d: Data, sym: str, expiry: pd.Timestamp, entry: pd.Timestamp,
              side: str, spread_bps: float) -> float:
    """Round trip for one leg as a fraction of its notional: charges at the
    contract's real notional, plus the spread on both sides."""
    px = d.closes[(sym, expiry)].get(entry)
    lot = d.lots.get((sym, expiry), np.nan)
    notional = lot * px if np.isfinite(lot) and px else PRE_LOT_NOTIONAL
    charges = _FUT.round_trip_cost(side, 1.0, notional, notional) / notional
    return charges + 2 * spread_bps / 1e4


# ── H1 / H2: quantile long-shorts (§4, §5) ──────────────────────────────────

def signal_skew(d: Data, c: Cycle, syms: List[str], days: int = SIGNAL_WINDOW) -> pd.Series:
    win = _window(d.sessions, c.signal, days)
    x = d.skew.reindex(index=win, columns=syms)
    need = SIGNAL_MIN_OBS if days == SIGNAL_WINDOW else 1
    return x.mean()[x.notna().sum() >= need]


def signal_parity(d: Data, c: Cycle, syms: List[str], days: int = SIGNAL_WINDOW) -> pd.Series:
    win = _window(d.sessions, c.signal, days)
    idx = pd.MultiIndex.from_product([win, syms, [c.next_expiry]])
    x = d.parity.reindex(idx).droplevel(2).unstack()
    need = SIGNAL_MIN_OBS if days == SIGNAL_WINDOW else 1
    return x.mean()[x.notna().sum() >= need]


def quantile_book(sig: pd.Series, n_q: int, long_low: bool) -> Tuple[List[str], List[str]]:
    """Quantiles of the signal, ties broken by symbol so a run is reproducible."""
    sig = sig.dropna()
    if len(sig) < n_q * 2:
        return [], []
    order = pd.DataFrame({"sym": sig.index, "v": sig.to_numpy()}).sort_values(["v", "sym"])
    q = pd.qcut(np.arange(len(order)), n_q, labels=False)
    low = list(order.sym[q == 0])
    high = list(order.sym[q == n_q - 1])
    return (low, high) if long_low else (high, low)


def run_quantile(d: Data, cs: List[Cycle], signal: Callable, long_low: bool,
                 n_q: int = N_QUANTILES, spread_bps: float = SPREAD_BPS_SIDE,
                 days: int = SIGNAL_WINDOW, allow_holdout: bool = False) -> pd.DataFrame:
    rows = []
    for c in cs:
        guard(c, allow_holdout)
        syms = universe(d, c)
        sig = signal(d, c, syms, days)
        longs, shorts = quantile_book(sig, n_q, long_low)
        if not longs:
            continue
        r = {s: leg_return(d, s, c.next_expiry, c.entry, c.exit) for s in syms}
        L = np.mean([r[s] for s in longs])
        S = np.mean([r[s] for s in shorts])
        U = np.mean([r[s] for s in sig.index])
        cost = (np.mean([cost_frac(d, s, c.next_expiry, c.entry, "BUY", spread_bps) for s in longs])
                + np.mean([cost_frac(d, s, c.next_expiry, c.entry, "SELL", spread_bps) for s in shorts]))
        rows.append(dict(entry=c.entry, exit=c.exit, n_universe=len(sig), n_leg=len(longs),
                         long=L, short=S, universe=U, gross=L - S, cost=cost, net=L - S - cost,
                         nifty=_compound(d.nifty, c.entry, c.exit),
                         longs=" ".join(longs), shorts=" ".join(shorts)))
    return pd.DataFrame(rows)


def _compound(s: pd.Series, start, end) -> float:
    x = s[(s.index > start) & (s.index <= end)].fillna(0)
    return float(np.prod(1 + x) - 1)


# ── H3: distance pairs (§6) ─────────────────────────────────────────────────

def form_pairs(d: Data, c: Cycle, syms: List[str], n_pairs: int = N_PAIRS) -> List[Tuple[str, str, float]]:
    win = _window(d.sessions, c.signal, FORMATION)
    r = d.ret.reindex(index=win, columns=syms).fillna(0.0)
    x = np.vstack([np.ones(len(syms)), np.cumprod(1 + r.to_numpy(), axis=0)])
    sq = (x ** 2).sum(axis=0)
    ssd = sq[:, None] + sq[None, :] - 2 * x.T @ x
    iu = np.triu_indices(len(syms), 1)
    order = np.argsort(ssd[iu], kind="stable")
    used, pairs = set(), []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        if i in used or j in used:
            continue
        used |= {i, j}
        pairs.append((syms[i], syms[j], float(np.std(x[:, i] - x[:, j]))))
        if len(pairs) == n_pairs:
            break
    return pairs


def trade_pair(d: Data, c: Cycle, a: str, b: str, sigma: float,
               spread_bps: float) -> Optional[dict]:
    """One cycle of one pair: open the session after |spread| > 2 sigma at a
    close, close the session after it crosses zero, or at the cycle exit."""
    days = d.sessions[(d.sessions > c.signal) & (d.sessions <= c.exit)]
    r = d.ret.reindex(index=days, columns=[a, b]).fillna(0.0)
    spread = (np.cumprod(1 + r[a]) - np.cumprod(1 + r[b])).to_numpy()
    for t in range(len(days) - 1):
        if abs(spread[t]) > OPEN_SIGMA * sigma:
            open_i, sign = t + 1, np.sign(spread[t])
            break
    else:
        return None
    close_i = len(days) - 1
    for u in range(open_i, len(days) - 1):
        if np.sign(spread[u]) != sign:
            close_i = u + 1
            break
    if close_i <= open_i:
        return None
    t_open, t_close = days[open_i], days[close_i]
    long_, short_ = (b, a) if sign > 0 else (a, b)        # long the lower index
    for s in (long_, short_):                             # relisted -> close early
        for t in days[(days > t_open) & (days <= t_close)]:
            if d.actions.get((t, s), ("",))[0] == "relisted":
                t_close = d.sessions[d.sessions.searchsorted(t) - 1]
                break
    lr = leg_return(d, long_, c.next_expiry, t_open, t_close)
    sr = leg_return(d, short_, c.next_expiry, t_open, t_close)
    cost = (cost_frac(d, long_, c.next_expiry, t_open, "BUY", spread_bps)
            + cost_frac(d, short_, c.next_expiry, t_open, "SELL", spread_bps))
    return dict(long=long_, short=short_, open=t_open, close=t_close,
                long_ret=lr, short_ret=sr, gross=lr - sr, cost=cost, net=lr - sr - cost)


def run_pairs(d: Data, cs: List[Cycle], n_pairs: int = N_PAIRS,
              spread_bps: float = SPREAD_BPS_SIDE, allow_holdout: bool = False
              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, trades = [], []
    for c in cs:
        guard(c, allow_holdout)
        syms = universe(d, c)
        pairs = form_pairs(d, c, syms, n_pairs)
        if not pairs:
            continue
        u_daily = d.ret.reindex(columns=syms).mean(axis=1)
        nets, longs, shorts = [], [], []
        for a, b, sig in pairs:
            t = trade_pair(d, c, a, b, sig, spread_bps)
            if t is None:
                nets.append(0.0)
                continue
            u = _compound(u_daily, t["open"], t["close"])
            longs.append(t["long_ret"] - u)
            shorts.append(t["short_ret"] - u)
            nets.append(t["net"])
            trades.append(dict(entry=c.entry, **t))
        rows.append(dict(entry=c.entry, exit=c.exit, n_universe=len(syms), n_pairs=len(pairs),
                         n_open=len(longs), net=float(np.mean(nets)),
                         long_vs_univ=float(np.mean(longs)) if longs else np.nan,
                         short_vs_univ=float(np.mean(shorts)) if shorts else np.nan,
                         nifty=_compound(d.nifty, c.entry, c.exit)))
    return pd.DataFrame(rows), pd.DataFrame(trades)


# ── C0 (§7) ─────────────────────────────────────────────────────────────────

def run_c0(d: Data, oi: pd.DataFrame, cs: List[Cycle]) -> pd.DataFrame:
    carry = oi.pivot(index="date", columns="symbol", values="cal_spread_ann")
    fut_cost = _FUT.round_trip_cost("SELL", 1.0, PRE_LOT_NOTIONAL, PRE_LOT_NOTIONAL) / PRE_LOT_NOTIONAL
    rows = []
    for c in cs:
        syms = universe(d, c)
        cr = carry.reindex(index=[c.signal], columns=syms).iloc[0].median()
        days = (c.next_expiry - c.entry).days
        rows.append(dict(entry=c.entry, carry_ann=cr,
                         net=cr * days / 365 - 0.00222 - fut_cost))
    return pd.DataFrame(rows)


# ── verdict (§8) ────────────────────────────────────────────────────────────

def verdict(df: pd.DataFrame, legs: Tuple[float, float]) -> dict:
    x = df.net.to_numpy()
    n = len(x)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    yr = df.groupby(df.entry.dt.year).net.mean()
    yrs = yr.reindex(YEARS)
    best = yr.idxmax()
    drop_best = df[df.entry.dt.year != best].net.mean()
    p = dict(P1=bool(x.mean() > 0 and t >= T_PASS),
             P2=bool((yrs > 0).sum() >= MIN_POSITIVE_YEARS),
             P3=bool(drop_best > 0),
             P4=bool(legs[0] > 0 and legs[1] > 0))
    cum = np.cumsum(x)
    return dict(n=n, mean=x.mean(), sd=x.std(ddof=1), t=t,
                years_pos=int((yrs > 0).sum()), by_year=yrs.round(4).to_dict(),
                best_year=int(best), mean_drop_best=drop_best,
                worst=x.min(), max_dd=float((cum - np.maximum.accumulate(cum)).min()),
                beta=float(np.cov(x, df.nifty)[0, 1] / df.nifty.var()) if n > 2 else np.nan,
                **p, PASS=all(p.values()))


def quantile_legs(df: pd.DataFrame) -> Tuple[float, float]:
    return float((df.long - df.universe).mean()), float((df.universe - df.short).mean())


def pair_legs(df: pd.DataFrame) -> Tuple[float, float]:
    # §8 P4 (H3): the long-leg minus short-leg difference must be > 0; both
    # legs vs the universe are reported. Expressed as (difference, difference)
    # so the same P4 test applies.
    diff = float((df.long_vs_univ - df.short_vs_univ).mean())
    return diff, diff


# ── driver ──────────────────────────────────────────────────────────────────

def load(fno: Path = FNO) -> Tuple[Data, pd.DataFrame]:
    rd = lambda p, **k: pd.read_csv(fno / p, low_memory=False, **k)
    futs = rd("futures.csv.gz", parse_dates=["date", "expiry"])
    acts = rd("features/corp_actions.csv.gz", parse_dates=["date"])
    oi = rd("features/oi_buildup.csv.gz", parse_dates=["date"])
    ivd = rd("features/iv_daily.csv.gz", parse_dates=["date"])
    exiv = rd("features/expiry_iv.csv.gz", parse_dates=["date", "expiry"],
              usecols=["date", "symbol", "expiry", "parity", "fut_close"])
    return Data.build(futs, acts, oi, ivd, exiv), oi


def _fmt(name: str, v: dict) -> str:
    return (f"{name:4s} n={v['n']:3d} mean={v['mean']*1e4:+8.1f} bps  t={v['t']:+.2f}  "
            f"years+={v['years_pos']}/7  drop-best={v['mean_drop_best']*1e4:+.1f}  "
            f"P1={v['P1']} P2={v['P2']} P3={v['P3']} P4={v['P4']}  -> "
            f"{'PASS' if v['PASS'] else 'FAIL'}\n"
            f"     by year (bps): { {k: round(x*1e4, 1) for k, x in v['by_year'].items()} }\n"
            f"     worst {v['worst']*1e4:+.0f} bps, max DD {v['max_dd']*1e4:+.0f} bps, "
            f"beta to NIFTY {v['beta']:+.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--holdout", choices=["H1", "H2", "H3"],
                    help="open the holdout for ONE hypothesis that passed development")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d, oi = load()
    allc = cycles(d.sessions, d.expiries)
    if a.holdout:
        return _holdout(d, allc, a.holdout, out)

    cs = period(allc, holdout=False)
    logger.info(f"{len(cs)} development cycles, {cs[0].entry.date()} .. {cs[-1].entry.date()}")
    lines, res = [], {}
    h1 = run_quantile(d, cs, signal_skew, long_low=True)
    h2 = run_quantile(d, cs, signal_parity, long_low=False)
    h3, h3t = run_pairs(d, cs)
    for name, df, legs in (("H1", h1, quantile_legs(h1)), ("H2", h2, quantile_legs(h2)),
                           ("H3", h3, pair_legs(h3))):
        df.to_csv(out / f"{name}_cycles.csv", index=False)
        res[name] = verdict(df, legs)
        lines.append(_fmt(name, res[name]))
        if name != "H3":
            lines.append(f"     legs: long-univ {legs[0]*1e4:+.1f} bps, univ-short {legs[1]*1e4:+.1f}"
                         f" bps; mean cost {df.cost.mean()*1e4:.1f} bps; universe {df.n_universe.median():.0f}")
        else:
            lines.append(f"     legs vs universe: long {h3.long_vs_univ.mean()*1e4:+.1f}, short "
                         f"{h3.short_vs_univ.mean()*1e4:+.1f} bps; pairs opened/cycle "
                         f"{h3.n_open.mean():.1f} of {h3.n_pairs.median():.0f}")
    h3t.to_csv(out / "H3_trades.csv", index=False)

    c0 = run_c0(d, oi, cs)
    c0.to_csv(out / "C0_cycles.csv", index=False)
    lines.append(f"C0   cash-and-carry: median carry {c0.carry_ann.median():.2%}/yr, "
                 f"net per cycle {c0.net.mean()*1e4:+.1f} bps")

    # §8 reported, never used to pass
    sens = []
    for name, sig, low in (("H1", signal_skew, True), ("H2", signal_parity, False)):
        for label, kw in (("spread0", dict(spread_bps=0.0)), ("spread10", dict(spread_bps=10.0)),
                          ("terciles", dict(n_q=3)), ("1day", dict(days=1))):
            v = run_quantile(d, cs, sig, long_low=low, **kw)
            sens.append(dict(h=name, variant=label, mean_bps=v.net.mean() * 1e4,
                             t=verdict(v, quantile_legs(v))["t"]))
    for label, kw in (("spread0", dict(spread_bps=0.0)), ("spread10", dict(spread_bps=10.0)),
                      ("10pairs", dict(n_pairs=10)), ("30pairs", dict(n_pairs=30))):
        v, _ = run_pairs(d, cs, **kw)
        sens.append(dict(h="H3", variant=label, mean_bps=v.net.mean() * 1e4,
                         t=verdict(v, pair_legs(v))["t"]))
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)
    lines.append("\nSENSITIVITY (reported only, never used to pass):\n" + sens.round(2).to_string(index=False))
    passed = [k for k, v in res.items() if v["PASS"]]
    lines.append(f"\nDEVELOPMENT VERDICT: {len(passed)} of 3 pass" +
                 (f" ({', '.join(passed)}); holdout may be opened once for each." if passed
                  else ". Per FNO_PREREG §8 nothing is re-run with other settings."))
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


def _holdout(d: Data, allc: List[Cycle], name: str, out: Path) -> int:
    dev = pd.read_csv(out / f"{name}_cycles.csv", parse_dates=["entry"])
    hc = period(allc, holdout=True)
    if name == "H3":
        df, _ = run_pairs(d, hc, allow_holdout=True)
    else:
        sig, low = (signal_skew, True) if name == "H1" else (signal_parity, False)
        df = run_quantile(d, hc, sig, long_low=low, allow_holdout=True)
    df.to_csv(out / f"{name}_holdout.csv", index=False)
    ok = df.net.mean() > 0 and df.net.mean() >= 0.5 * dev.net.mean()
    print(f"{name} holdout: n={len(df)} mean {df.net.mean()*1e4:+.1f} bps vs development "
          f"{dev.net.mean()*1e4:+.1f} -> {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
