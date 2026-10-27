"""
Daily history for the multi-asset research: NSE ETFs, NSE indices, MCX and
currency futures (Kite continuous series).

    python -m scripts.multi_asset_history          # needs ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN

Writes data/multi/{etf,index,mcx,cds}/<SYMBOL>.csv (date, open, high, low,
close, volume[, oi]).

**Continuous futures** (Kite `continuous=True`) splice the near contract
expiry to expiry with NO back-adjustment, so the close series JUMPS at every
roll by the calendar spread. Returns across a roll are wrong; research code
must mark roll days (where the near contract changed) and drop or adjust
them. Kite does not say where the rolls are, so `research` detects them from
the expiry calendar in the instrument dump (see `roll_dates`).
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "multi"

ETFS = ["NIFTYBEES", "JUNIORBEES", "BANKBEES", "MID150BEES", "HDFCSML250",
        "GOLDBEES", "SILVERBEES", "MON100", "MAFANG", "MASPTOP50", "HNGSNGBEES",
        "LIQUIDBEES", "LTGILTBEES", "GILT5YBEES", "SETF10GILT", "CPSEETF",
        "PSUBNKBEES", "ITBEES", "PHARMABEES", "INFRABEES", "CONSUMBEES",
        "MOM30IETF", "LOWVOLIETF", "NV20IETF", "ALPHA", "AUTOBEES", "METALIETF",
        "FMCGIETF", "EBBETF0430"]
INDICES = ["NIFTY 50", "NIFTY NEXT 50", "NIFTY MIDCAP 100", "NIFTY SMLCAP 100",
           "NIFTY BANK", "NIFTY IT", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY METAL",
           "NIFTY ENERGY", "NIFTY AUTO", "NIFTY REALTY", "NIFTY INFRA",
           "NIFTY PSU BANK", "NIFTY FIN SERVICE", "NIFTY MEDIA", "INDIA VIX",
           "NIFTY200 MOMENTM30", "NIFTY100 LOWVOL30", "NIFTY ALPHA 50"]
MCX = ["GOLD", "GOLDM", "SILVER", "SILVERM", "CRUDEOIL", "NATURALGAS", "COPPER",
       "ZINC", "ALUMINIUM", "LEAD", "NICKEL"]
CDS = ["USDINR", "EURINR", "GBPINR", "JPYINR"]

START = dt.date(2008, 1, 1)
CHUNK = 1900          # Kite day candles: at most 2000 days a request


def kite():
    from kiteconnect import KiteConnect
    k = KiteConnect(api_key=os.environ["ZERODHA_API_KEY"])
    k.set_access_token(os.environ["ZERODHA_ACCESS_TOKEN"])
    return k


def fetch(k, token: int, continuous: bool = False, oi: bool = False) -> pd.DataFrame:
    rows, a, end = [], START, dt.date.today()
    while a <= end:
        b = min(a + dt.timedelta(days=CHUNK), end)
        for attempt in range(3):
            try:
                rows += k.historical_data(token, a, b, "day", continuous=continuous, oi=oi)
                break
            except Exception as e:                      # noqa: BLE001
                if "No data" in str(e) or attempt == 2:
                    break
                time.sleep(1.5)
        time.sleep(0.35)
        a = b + dt.timedelta(days=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def near_future(inst: pd.DataFrame, name: str) -> int | None:
    f = inst[(inst["name"] == name) & (inst["instrument_type"] == "FUT")]
    if f.empty:
        return None
    return int(f.sort_values("expiry").iloc[0]["instrument_token"])


def main() -> int:
    k = kite()
    nse = pd.DataFrame(k.instruments("NSE"))
    jobs = []
    eq = nse[nse["segment"] == "NSE"].set_index("tradingsymbol")["instrument_token"]
    ix = nse[nse["segment"] == "INDICES"].set_index("tradingsymbol")["instrument_token"]
    jobs += [("etf", s, int(eq[s]), False) for s in ETFS if s in eq]
    jobs += [("index", s, int(ix[s]), False) for s in INDICES if s in ix]
    for ex, names in (("MCX", MCX), ("CDS", CDS)):
        inst = pd.DataFrame(k.instruments(ex))
        inst.to_csv(OUT / f"instruments_{ex}.csv", index=False) if OUT.exists() else None
        for n in names:
            t = near_future(inst, n)
            if t:
                jobs.append((ex.lower(), n, t, True))
    for kind, sym, tok, cont in jobs:
        path = OUT / kind / f"{sym.replace(' ', '_')}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch(k, tok, continuous=cont, oi=cont)
        if df.empty:
            print(f"{kind:5s} {sym:20s} NO DATA")
            continue
        df.to_csv(path, index=False)
        print(f"{kind:5s} {sym:20s} {len(df):5d} rows {df.date.iloc[0].date()} -> {df.date.iloc[-1].date()}")
    return 0


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sys.exit(main())
