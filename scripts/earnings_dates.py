"""
Point-in-time quarterly result dates from NSE, for docs/MULTI_ASSET.md round 3 (D2).

    python -m scripts.earnings_dates --symbols-file <file> [--sleep 1.0]

Two NSE endpoints, each giving the exchange's own dissemination timestamp:
- corporates-financial-results (legacy format, 2007 -> Jan 2025)
- integrated-filing-results, type "Integrated Filing- Financials" (2025 ->)

Raw JSON is cached per symbol in data/earnings/raw/. events.csv holds one row
per (symbol, quarter end): the EARLIEST broadcast over both endpoints and over
standalone/consolidated, since that is when the market first saw the numbers.
Nothing here computes a return.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "earnings"
RAW = OUT / "raw"
API = "https://www.nseindia.com/api/"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
LEGACY = "corporates-financial-results?index=equities&period=Quarterly&symbol={s}"
INTEGRATED = "integrated-filing-results?index=equities&type=Integrated%20Filing-%20Financials&symbol={s}"


def _get(sess: requests.Session, url: str, tries: int = 3):
    for k in range(tries):
        try:
            r = sess.get(API + url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3 * (k + 1))
    return None


def fetch(symbols, sleep: float) -> list[str]:
    RAW.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Referer": "https://www.nseindia.com/",
                         "Accept": "application/json"})
    failed = []
    for i, s in enumerate(symbols):
        f = RAW / f"{s}.json"
        if f.exists():
            continue
        q = requests.utils.quote(s, safe="")
        leg = _get(sess, LEGACY.format(s=q))
        time.sleep(sleep)
        integ = _get(sess, INTEGRATED.format(s=q))
        time.sleep(sleep)
        if leg is None or integ is None:
            failed.append(s)
            continue
        f.write_text(json.dumps({"legacy": leg, "integrated": integ}))
        if i % 50 == 0:
            print(f"{i}/{len(symbols)} {s}", flush=True)
    return failed


def parse_one(symbol: str, raw: dict) -> pd.DataFrame:
    rows = []
    for r in raw.get("legacy") or []:
        rows.append((r.get("broadCastDate"), r.get("toDate"), "%d-%b-%Y"))
    integ = raw.get("integrated") or {}
    for r in (integ.get("data") if isinstance(integ, dict) else integ) or []:
        if (r.get("type_Sub") or "Original") != "Original":
            continue                        # a revision is not a first sight of the numbers
        rows.append((r.get("broadcast_Date"), r.get("qe_Date"), "%d-%b-%Y"))
    if not rows:
        return pd.DataFrame(columns=["symbol", "quarter_end", "broadcast"])
    d = pd.DataFrame(rows, columns=["b", "q", "fmt"])
    d["broadcast"] = pd.to_datetime(d["b"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    d["quarter_end"] = pd.to_datetime(d["q"].str.title(), format="%d-%b-%Y", errors="coerce")
    d = d.dropna(subset=["broadcast", "quarter_end"])
    # a broadcast before its quarter has ended, or more than 200 days after, is a data error
    lag = (d["broadcast"] - d["quarter_end"]).dt.days
    d = d[(lag >= 0) & (lag <= 200)]
    d = d.groupby("quarter_end", as_index=False)["broadcast"].min()
    d.insert(0, "symbol", symbol)
    return d


def build() -> pd.DataFrame:
    parts = [parse_one(f.stem, json.loads(f.read_text())) for f in sorted(RAW.glob("*.json"))]
    ev = pd.concat([p for p in parts if len(p)], ignore_index=True).sort_values(["broadcast", "symbol"])
    ev.to_csv(OUT / "events.csv", index=False)
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args(argv)
    syms = [s.strip() for s in Path(a.symbols_file).read_text().split() if s.strip()]
    failed = fetch(syms, a.sleep)
    ev = build()
    print(f"events {len(ev)} symbols {ev['symbol'].nunique()} failed {len(failed)}: {failed[:20]}")
    print(ev.groupby(ev["broadcast"].dt.year).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
