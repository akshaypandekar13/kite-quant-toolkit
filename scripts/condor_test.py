"""NIFTY monthly iron condor: the test frozen in docs/CONDOR_PREREG.md. Paper only.

    python -m scripts.condor_test               # development run (once)
    python -m scripts.condor_test --holdout     # ONLY after a development pass

Section numbers in comments refer to CONDOR_PREREG. Writes to
structural/condor: cycles.csv, sensitivity.csv, summary.txt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fno_features import b76_delta, implied_vol  # noqa: E402
from scripts.fno_test import (Cycle, HoldoutError, cycles, expiry_aliases,  # noqa: E402
                              guard, period)

logger = logging.getLogger("condor_test")

FNO = Path("data/fno")
OUT = Path("structural/condor")

# §2
SHORT_DELTA = 0.15
WING_PCT = 0.04
# §3
SPREAD_PTS = 0.5
BROKERAGE_ORDER = 20.0
STT_SELL = 0.001
EXCHANGE = 0.0003503
SEBI = 10 / 1e7
STAMP_BUY = 0.00003
GST = 0.18
# §4
T_PASS = 2.0
YEARS = list(range(2019, 2026))
MIN_POSITIVE_YEARS = 5
MAX_DD = 0.30


@dataclass
class Market:
    sessions: pd.DatetimeIndex
    expiries: List[pd.Timestamp]
    fut: pd.Series                              # (date, expiry) -> NIFTY future close
    chains: Dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]   # (date, expiry) -> options
    carry: pd.Series                            # date -> NIFTY cal_spread_ann
    index_close: pd.Series                      # date -> NIFTY 50 close (expiry settlement)
    lot: float

    @classmethod
    def build(cls, futs: pd.DataFrame, opts: pd.DataFrame, carry: pd.Series,
              index_close: pd.Series) -> "Market":
        n = futs[(futs.symbol == "NIFTY") & (futs.instrument == "FUTIDX")]
        alias = expiry_aliases(n)
        n = n.assign(expiry=n.expiry.replace(alias))
        opts = opts.assign(expiry=opts.expiry.replace(alias))
        sessions = pd.DatetimeIndex(np.sort(futs.date.unique()))
        expiries = sorted(e for e in n.expiry.unique() if e <= sessions[-1])
        opts = opts[opts.expiry.isin(set(n.expiry))]
        chains = {k: g.set_index(["strike", "opt_type"]).sort_index()
                  for k, g in opts.groupby(["date", "expiry"])}
        fut = n.groupby(["date", "expiry"]).close.last()
        lots = n.dropna(subset=["lot"]) if "lot" in n else n.iloc[0:0]
        lot = float(lots.sort_values("date").lot.iloc[-1]) if len(lots) else 75.0
        return cls(sessions, expiries, fut, chains, carry, index_close, lot)


# ── §2: choose the condor on s_k ────────────────────────────────────────────

@dataclass
class Condor:
    short_put: float
    short_call: float
    long_put: float
    long_call: float

    def legs(self):
        # (strike, type, +1 long / -1 short)
        return [(self.short_put, "PE", -1), (self.short_call, "CE", -1),
                (self.long_put, "PE", +1), (self.long_call, "CE", +1)]


def choose(m: Market, c: Cycle, delta: float = SHORT_DELTA,
           wing: float = WING_PCT) -> Optional[Condor]:
    ch = m.chains.get((c.signal, c.next_expiry))
    F = m.fut.get((c.signal, c.next_expiry))
    if ch is None or F is None or not np.isfinite(F):
        return None
    T = (c.next_expiry - c.signal).days / 365
    ch = ch.reset_index()
    tr = ch[(ch.contracts > 0) & (ch.close > 0)].copy()
    is_call = (tr.opt_type == "CE").to_numpy()
    tr["iv"] = implied_vol(tr.close, np.full(len(tr), F), tr.strike, np.full(len(tr), T), is_call)
    tr = tr.dropna(subset=["iv"])
    tr["delta"] = b76_delta(F, tr.strike, T, tr.iv, (tr.opt_type == "CE").to_numpy())

    def pick(side, target, further_is_higher):
        x = tr[tr.opt_type == side].assign(err=lambda d: (d.delta - target).abs())
        if x.empty:
            return None
        x = x.sort_values(["err", "strike"], ascending=[True, not further_is_higher])
        return float(x.strike.iloc[0])

    sp, sc = pick("PE", -delta, False), pick("CE", delta, True)
    if sp is None or sc is None:
        return None
    W = wing * F
    puts = ch[(ch.opt_type == "PE") & (ch.strike < sp)].strike
    calls = ch[(ch.opt_type == "CE") & (ch.strike > sc)].strike
    if puts.empty or calls.empty:
        return None
    # nearest to the target; ties go further out
    lp = float(puts.iloc[np.lexsort((puts.to_numpy(), (puts - (sp - W)).abs().to_numpy()))[0]])
    lc = float(calls.iloc[np.lexsort((-calls.to_numpy(), (calls - (sc + W)).abs().to_numpy()))[0]])
    return Condor(sp, sc, lp, lc)


# ── §3: prices, costs ───────────────────────────────────────────────────────

def leg_price(m: Market, day, expiry, strike, typ, counts: dict) -> Optional[float]:
    """Close if it traded, else NSE's settle (counted). None if absent."""
    ch = m.chains.get((day, expiry))
    if ch is None or (strike, typ) not in ch.index:
        return None
    row = ch.loc[(strike, typ)]
    if row.contracts > 0:
        return float(row.close)
    counts["settle"] = counts.get("settle", 0) + 1
    return float(row.settle)


def order_cost(premium: float, sell: bool, lot: float) -> float:
    """Charges for one order, per unit of underlying, in index points."""
    brk = BROKERAGE_ORDER / lot
    exch = EXCHANGE * premium
    sebi = SEBI * premium
    return (brk + exch + sebi + GST * (brk + exch + sebi)
            + (STT_SELL * premium if sell else STAMP_BUY * premium))


def intrinsic(strike, typ, F) -> float:
    return max(F - strike, 0.0) if typ == "CE" else max(strike - F, 0.0)


@dataclass
class Result:
    entry: pd.Timestamp
    credit: float
    debit: float
    costs: float
    net: float
    risk: float
    r: float
    c0: float
    e: float
    put_side: float
    call_side: float
    fallbacks: int
    missing_exit: int
    strikes: str


def run_cycle(m: Market, c: Cycle, delta=SHORT_DELTA, wing=WING_PCT, spread=SPREAD_PTS,
              to_expiry=False, allow_holdout=False) -> Optional[Result]:
    guard(c, allow_holdout)
    k = choose(m, c, delta, wing)
    if k is None:
        return None
    counts: dict = {}
    entry = {}
    for s, t, sign in k.legs():
        p = leg_price(m, c.entry, c.next_expiry, s, t, counts)
        if p is None:                                   # A1: untradeable at entry
            return None
        entry[(s, t)] = p
    exit_px, missing = {}, 0
    if to_expiry:
        S = m.index_close.get(c.next_expiry)
        if S is None or not np.isfinite(S):
            return None
        exit_px = {(s, t): intrinsic(s, t, S) for s, t, _ in k.legs()}
    else:
        F = m.fut.get((c.exit, c.next_expiry))
        for s, t, sign in k.legs():
            p = leg_price(m, c.exit, c.next_expiry, s, t, counts)
            if p is None:                               # A1: absent at exit -> intrinsic
                missing += 1
                p = intrinsic(s, t, F)
            exit_px[(s, t)] = p
    credit = sum(-sign * entry[(s, t)] for s, t, sign in k.legs())
    debit = sum(-sign * exit_px[(s, t)] for s, t, sign in k.legs())
    costs = sum(order_cost(entry[(s, t)], sell=sign < 0, lot=m.lot) for s, t, sign in k.legs())
    if to_expiry:
        # cash settlement: no exit orders; STT 0.125% on exercised (ITM) longs
        costs += sum(0.00125 * exit_px[(s, t)] for s, t, sign in k.legs() if sign > 0)
        costs += 4 * spread
    else:
        costs += sum(order_cost(exit_px[(s, t)], sell=sign > 0, lot=m.lot) for s, t, sign in k.legs())
        costs += 8 * spread
    width = max(k.long_call - k.short_call, k.short_put - k.long_put)
    risk = width - credit
    net = credit - debit - costs
    r = net / risk if risk > 0 else np.nan
    days = (c.exit - c.entry).days if not to_expiry else (c.next_expiry - c.entry).days
    c0 = float(m.carry.get(c.signal, np.nan)) * days / 365
    side = lambda typ: sum(-sign * (entry[(s, t)] - exit_px[(s, t)])
                           for s, t, sign in k.legs() if t == typ)
    return Result(c.entry, credit, debit, costs, net, risk, r, c0, r - c0,
                  side("PE"), side("CE"), counts.get("settle", 0), missing,
                  f"{k.long_put:.0f}/{k.short_put:.0f}/{k.short_call:.0f}/{k.long_call:.0f}")


def run(m: Market, cs: List[Cycle], **kw) -> pd.DataFrame:
    rows = [run_cycle(m, c, **kw) for c in cs]
    return pd.DataFrame([r.__dict__ for r in rows if r is not None])


# ── §4: verdict ─────────────────────────────────────────────────────────────

def drawdown(r: np.ndarray) -> float:
    eq = 1 + 0.5 * np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(((peak - eq) / peak).max()) if len(eq) else np.nan


def verdict(df: pd.DataFrame) -> dict:
    e = df.e.to_numpy()
    n = len(e)
    t = e.mean() / (e.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan
    yr = df.groupby(df.entry.dt.year).e.mean()
    yrs = yr.reindex(YEARS)
    best = yr.idxmax()
    drop = df[df.entry.dt.year != best].e.mean()
    dd = drawdown(df.r.to_numpy())
    p = dict(P1=bool(e.mean() > 0 and t >= T_PASS),
             P2=bool((yrs > 0).sum() >= MIN_POSITIVE_YEARS),
             P3=bool(drop > 0), P4=bool(dd <= MAX_DD))
    return dict(n=n, mean_e=e.mean(), t=t, years_pos=int((yrs > 0).sum()),
                by_year=yrs.round(4).to_dict(), best=int(best), drop_best=drop, max_dd=dd,
                mean_r=df.r.mean(), win_rate=float((df.net > 0).mean()),
                **p, PASS=all(p.values()))


# ── driver ──────────────────────────────────────────────────────────────────

def load(fno: Path = FNO) -> Market:
    futs = pd.read_csv(fno / "futures.csv.gz", parse_dates=["date", "expiry"], low_memory=False)
    parts = []
    for p in sorted((fno / "options").glob("options_*.csv.gz")):
        o = pd.read_csv(p, usecols=["date", "symbol", "expiry", "strike", "opt_type",
                                    "close", "settle", "contracts"],
                        parse_dates=["date", "expiry"])
        parts.append(o[o.symbol == "NIFTY"].drop(columns="symbol"))
    opts = pd.concat(parts, ignore_index=True)
    oi = pd.read_csv(fno / "features/oi_buildup.csv.gz", parse_dates=["date"], low_memory=False)
    carry = oi[oi.symbol == "NIFTY"].set_index("date").cal_spread_ann
    k = pd.read_csv(fno / "kite/day/NIFTY_50.csv", parse_dates=["Datetime"])
    idx = k.set_index("Datetime").Close
    return Market.build(futs, opts, carry, idx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = load()
    allc = cycles(m.sessions, m.expiries)
    if a.holdout:
        dev = pd.read_csv(out / "cycles.csv")
        h = run(m, period(allc, holdout=True), allow_holdout=True)
        h.to_csv(out / "holdout.csv", index=False)
        ok = h.e.mean() > 0 and h.e.mean() >= 0.5 * dev.e.mean()
        print(f"holdout n={len(h)} mean e {h.e.mean():+.4f} vs dev {dev.e.mean():+.4f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return 0

    cs = period(allc, holdout=False)
    df = run(m, cs)
    df.to_csv(out / "cycles.csv", index=False)
    v = verdict(df)
    w = df.sort_values("net").head(3)
    lines = [
        f"{len(cs)} development cycles, {len(df)} traded "
        f"({len(cs) - len(df)} skipped); lot {m.lot:.0f}",
        f"mean e {v['mean_e']:+.4f}  t {v['t']:+.2f}  years+ {v['years_pos']}/7  "
        f"drop-best({v['best']}) {v['drop_best']:+.4f}  maxDD {v['max_dd']:.1%}  "
        f"P1={v['P1']} P2={v['P2']} P3={v['P3']} P4={v['P4']} -> "
        f"{'PASS' if v['PASS'] else 'FAIL'}",
        f"by year (mean e): {v['by_year']}",
        f"mean r {v['mean_r']:+.4f}, mean C0 {df.c0.mean():+.4f}, win rate {v['win_rate']:.0%} "
        f"(NOT a criterion)",
        f"mean credit {df.credit.mean():.1f} pts, mean risk R {df.risk.mean():.1f} pts, "
        f"mean costs {df.costs.mean():.1f} pts",
        f"put side {df.put_side.mean():+.1f} pts/cycle, call side {df.call_side.mean():+.1f}",
        f"settle fallbacks {int(df.fallbacks.sum())} legs in {int((df.fallbacks > 0).sum())} "
        f"cycles; absent at exit (intrinsic) {int(df.missing_exit.sum())}",
        f"mean P&L per lot Rs{df.net.mean() * m.lot:+,.0f}; worst cycles: "
        + "; ".join(f"{r.entry.date()} {r.net:+.0f} pts (r {r.r:+.2f})" for r in w.itertuples()),
    ]
    sens = []
    for label, kw in (("delta0.10", dict(delta=0.10)), ("delta0.20", dict(delta=0.20)),
                      ("wing3%", dict(wing=0.03)), ("wing6%", dict(wing=0.06)),
                      ("to_expiry", dict(to_expiry=True)), ("spread0.25", dict(spread=0.25)),
                      ("spread1.0", dict(spread=1.0)), ("spread2.0", dict(spread=2.0))):
        s = run(m, cs, **kw)
        sv = verdict(s)
        sens.append(dict(variant=label, n=sv["n"], mean_e=sv["mean_e"], t=sv["t"],
                         max_dd=sv["max_dd"], win=sv["win_rate"]))
    nf = df[(df.fallbacks == 0) & (df.missing_exit == 0)]
    if len(nf) > 2:
        sv = verdict(nf)
        sens.append(dict(variant="no_fallback", n=sv["n"], mean_e=sv["mean_e"], t=sv["t"],
                         max_dd=sv["max_dd"], win=sv["win_rate"]))
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity.csv", index=False)
    lines.append("\nSENSITIVITY (reported only, never used to pass):\n"
                 + sens.round(4).to_string(index=False))
    lines.append(f"\nDEVELOPMENT VERDICT: {'PASS - holdout may be opened once' if v['PASS'] else 'FAIL. Per CONDOR_PREREG §4 nothing is re-run with other settings.'}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
