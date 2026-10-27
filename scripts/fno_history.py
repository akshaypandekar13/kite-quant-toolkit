"""Download and normalise NSE F&O history for the long-short research.

Four daily files from NSE's public archive, one per trading session:

    fo     F&O bhavcopy: every future and option contract's OHLC, settle,
           open interest and change in OI. Legacy format before 2024-07-08,
           UDiFF from then on (which adds the underlying price and lot size).
    mto    security-wise delivery position: traded qty, delivered qty and
           delivery % for every cash-market symbol.
    poi    participant-wise open interest (Client / DII / FII / Pro).
    pvol   participant-wise trading volume, same layout.

    python -m scripts.fno_history download --start 2019-01-01
    python -m scripts.fno_history build

`download` caches the raw files under data/fno_raw (re-runs only fetch what is
missing; holidays are remembered). `build` writes normalised tables to
data/fno:

    futures.csv.gz              one row per futures contract per day
    options/options_YYYY.csv.gz option contracts with OI or volume > 0
    delivery.csv.gz             MTO delivery %, all series
    participant_oi.csv.gz / participant_vol.csv.gz

UNITS, checked across the 2024-07-05 -> 07-08 switch on RELIANCE/SBIN/NIFTY:
`oi` and `chg_oi` are in SHARES in both formats (continuous across the
switch). `contracts` is the number of contracts traded in both: UDiFF's
`TtlTradgVol` is contracts too, not shares (39,175 x lot 250 x Rs3,208 =
Rs313,500 lakh = its `value_lakh`). `lot` exists only in UDiFF and stays NaN
before it rather than being guessed. `value_lakh` is comparable across both.

PRICES ARE RAW, NOT ADJUSTED for splits/bonuses, unlike Kite's daily bars:
RELIANCE near futures sit ~+110% over Kite's cash close before its Oct-2024
1:1 bonus and +24 bps after it (UDiFF's own fut-vs-underlying: +23.8 bps).
Compare F&O prices only with raw cash prices (the CM bhavcopy), or adjust.

NOT AVAILABLE from any archive used here: order flow / delta (needs tick
data), historical IV (derive it from option settles and the underlying),
and news or transcripts.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.universe import NseHttpSource  # noqa: E402

logger = logging.getLogger("fno_history")

ARCH = "https://nsearchives.nseindia.com"
UDIFF_FROM = date(2024, 7, 8)
FO_UDIFF_URL = ARCH + "/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
FO_LEGACY_URL = ARCH + "/content/historical/DERIVATIVES/{y}/{mon}/fo{d}{mon}{y}bhav.csv.zip"
MTO_URL = ARCH + "/archives/equities/mto/MTO_{dmy}.DAT"
POI_URL = ARCH + "/content/nsccl/fao_participant_oi_{dmy}.csv"
PVOL_URL = ARCH + "/content/nsccl/fao_participant_vol_{dmy}.csv"

KINDS = ("fo", "mto", "poi", "pvol")
RAW = Path("data/fno_raw")
OUT = Path("data/fno")

COLS = ["date", "instrument", "symbol", "expiry", "strike", "opt_type",
        "open", "high", "low", "close", "settle", "contracts",
        "value_lakh", "oi", "chg_oi", "underlying", "lot"]


# ── fetching ────────────────────────────────────────────────────────────────

def raw_name(kind: str, day: date) -> str:
    ext = {"fo": "csv.zip", "mto": "DAT", "poi": "csv", "pvol": "csv"}[kind]
    return f"{kind}_{day:%Y%m%d}.{ext}"


def url_for(kind: str, day: date) -> str:
    dmy = f"{day:%d%m%Y}"
    if kind == "fo":
        if day >= UDIFF_FROM:
            return FO_UDIFF_URL.format(ymd=f"{day:%Y%m%d}")
        mon = f"{day:%b}".upper()
        return FO_LEGACY_URL.format(y=day.year, mon=mon, d=f"{day:%d}")
    return {"mto": MTO_URL, "poi": POI_URL, "pvol": PVOL_URL}[kind].format(dmy=dmy)


def weekdays(start: date, end: date) -> List[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


#: Kite's NIFTY 50 daily bars: the trading calendar, special weekend sessions
#: (Muhurat, Budget day, mock-trading Saturdays) included.
CALENDAR = Path("data/fno/kite/day/NIFTY_50.csv")


def load_calendar(path: Path = CALENDAR) -> Optional[List[date]]:
    if not path.exists():
        return None
    return sorted(date.fromisoformat(s[:10]) for s in pd.read_csv(path)["Datetime"])


def session_days(start: date, end: date,
                 calendar: Optional[List[date]]) -> List[tuple]:
    """(day, known_session). Inside the calendar's span only its sessions are
    used and every one is known to be a session; past its end, weekdays, with
    a missing `fo` file read as a holiday."""
    if not calendar:
        return [(d, False) for d in weekdays(start, end)]
    last = calendar[-1]
    days = [(d, True) for d in calendar if start <= d <= min(end, last)]
    return days + [(d, False) for d in weekdays(max(start, last + timedelta(days=1)), end)]


def download(start: date, end: date, kinds=KINDS, raw: Path = RAW,
             source: Optional[NseHttpSource] = None,
             calendar: Optional[List[date]] = None) -> Dict[str, Dict[str, int]]:
    """Fetch every missing raw file. On a day not known to be a session, an
    absent `fo` file means a holiday and the other kinds are not requested.
    On a known session they are: NSE's archive lacks a few F&O files (e.g.
    2021-03-30) whose delivery file exists."""
    src = source or NseHttpSource(cache_dir=str(raw))
    stats = {k: {"have": 0, "got": 0, "none": 0} for k in kinds}
    days = session_days(start, end, calendar)
    for i, (day, known) in enumerate(days, 1):
        for kind in kinds:
            name = raw_name(kind, day)
            if (Path(src.cache) / name).exists():
                stats[kind]["have"] += 1
                continue
            blob = src._fetch(url_for(kind, day), name, day)
            stats[kind]["got" if blob is not None else "none"] += 1
            if kind == "fo" and blob is None and not known:
                break                       # holiday: skip the other kinds
        if i % 50 == 0:
            logger.info(f"{i}/{len(days)} {day}  " + "  ".join(
                f"{k}:{v['have'] + v['got']}" for k, v in stats.items()))
    return stats


# ── parsing ─────────────────────────────────────────────────────────────────

def _read_zip_csv(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return pd.read_csv(zf.open(inner))


def parse_fo_legacy(raw: pd.DataFrame, day: date) -> pd.DataFrame:
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    df = pd.DataFrame({
        "date": pd.Timestamp(day),
        "instrument": raw["INSTRUMENT"].str.strip(),
        "symbol": raw["SYMBOL"].str.strip(),
        "expiry": pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": raw["STRIKE_PR"].astype(float),
        "opt_type": raw["OPTION_TYP"].str.strip().replace("XX", ""),
        "open": raw["OPEN"], "high": raw["HIGH"], "low": raw["LOW"],
        "close": raw["CLOSE"], "settle": raw["SETTLE_PR"],
        "contracts": raw["CONTRACTS"].astype(float),
        "value_lakh": raw["VAL_INLAKH"].astype(float),
        "oi": raw["OPEN_INT"].astype(float), "chg_oi": raw["CHG_IN_OI"].astype(float),
        "underlying": float("nan"), "lot": float("nan"),
    })
    return df[COLS]


#: UDiFF instrument type -> legacy name, so one column means one thing.
_UDIFF_INSTR = {"IDF": "FUTIDX", "STF": "FUTSTK", "IDO": "OPTIDX", "STO": "OPTSTK"}


def parse_fo_udiff(raw: pd.DataFrame, day: date) -> pd.DataFrame:
    df = pd.DataFrame({
        "date": pd.Timestamp(day),
        "instrument": raw["FinInstrmTp"].map(_UDIFF_INSTR).fillna(raw["FinInstrmTp"]),
        "symbol": raw["TckrSymb"].str.strip(),
        "expiry": pd.to_datetime(raw["XpryDt"]),
        "strike": raw["StrkPric"].fillna(0).astype(float),
        "opt_type": raw["OptnTp"].fillna("").astype(str).str.strip(),
        "open": raw["OpnPric"], "high": raw["HghPric"], "low": raw["LwPric"],
        "close": raw["ClsPric"], "settle": raw["SttlmPric"],
        "contracts": raw["TtlTradgVol"].astype(float),
        "value_lakh": raw["TtlTrfVal"].astype(float) / 1e5,
        "oi": raw["OpnIntrst"].astype(float), "chg_oi": raw["ChngInOpnIntrst"].astype(float),
        "underlying": raw["UndrlygPric"].astype(float),
        "lot": raw["NewBrdLotQty"].astype(float),
    })
    return df[COLS]


def parse_fo(blob: bytes, day: date) -> pd.DataFrame:
    raw = _read_zip_csv(blob)
    return parse_fo_udiff(raw, day) if "TckrSymb" in raw.columns else parse_fo_legacy(raw, day)


def parse_mto(blob: bytes, day: date) -> pd.DataFrame:
    """Record type 20 rows: sr, symbol, series, traded qty, delivered qty, %."""
    rows = []
    for line in blob.decode("latin-1").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and p[0] == "20":
            rows.append((p[2], p[3], p[4], p[5], p[6]))
    df = pd.DataFrame(rows, columns=["symbol", "series", "traded_qty",
                                     "deliv_qty", "deliv_pct"])
    for c in ("traded_qty", "deliv_qty", "deliv_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "date", pd.Timestamp(day))
    return df


def parse_participant(blob: bytes, day: date) -> pd.DataFrame:
    """Line 1 is a title; line 2 the header; then Client/DII/FII/Pro/TOTAL."""
    df = pd.read_csv(io.BytesIO(blob), skiprows=1)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["client_type"] = df["client_type"].astype(str).str.strip()
    df = df[df["client_type"].isin(["Client", "DII", "FII", "Pro", "TOTAL"])]
    num = [c for c in df.columns if c != "client_type"]
    df[num] = df[num].apply(lambda s: pd.to_numeric(
        s.astype(str).str.replace(",", "").str.strip(), errors="coerce"))
    df.insert(0, "date", pd.Timestamp(day))
    return df.reset_index(drop=True)


def _day_of(path: Path) -> date:
    s = path.name.split("_")[1].split(".")[0]
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def build(raw: Path = RAW, out: Path = OUT) -> Dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "options").mkdir(exist_ok=True)
    counts: Dict[str, int] = {}

    def flush(year, parts):
        if parts:
            o = pd.concat(parts, ignore_index=True)
            o.to_csv(out / "options" / f"options_{year}.csv.gz", index=False)
            counts[f"options_{year}"] = len(o)
            logger.info(f"options_{year}: {len(o):,} rows")

    # Files sort by date, so one year's options are held at a time.
    futs, year, opts = [], None, []
    for f in sorted(raw.glob("fo_*.csv.zip")):
        day = _day_of(f)
        try:
            df = parse_fo(f.read_bytes(), day)
        except Exception as e:
            logger.warning(f"skip {f.name}: {e}")
            continue
        if day.year != year:
            flush(year, opts)
            year, opts = day.year, []
        is_fut = df["instrument"].str.startswith("FUT")
        futs.append(df[is_fut])
        o = df[~is_fut]
        opts.append(o[(o["oi"] > 0) | (o["value_lakh"] > 0)])
    flush(year, opts)
    if futs:
        fut = pd.concat(futs, ignore_index=True)
        fut.drop(columns=["strike", "opt_type"]).to_csv(out / "futures.csv.gz", index=False)
        counts["futures"] = len(fut)

    for kind, parser, name in (("mto", parse_mto, "delivery"),
                               ("poi", parse_participant, "participant_oi"),
                               ("pvol", parse_participant, "participant_vol")):
        parts = []
        for f in sorted(raw.glob(f"{kind}_*")):
            try:
                parts.append(parser(f.read_bytes(), _day_of(f)))
            except Exception as e:
                logger.warning(f"skip {f.name}: {e}")
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_csv(out / f"{name}.csv.gz", index=False)
            counts[name] = len(df)
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", default="2019-01-01")
    d.add_argument("--end", default=str(date.today()))
    d.add_argument("--kinds", default=",".join(KINDS))
    d.add_argument("--raw", default=str(RAW))
    b = sub.add_parser("build")
    b.add_argument("--raw", default=str(RAW))
    b.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if a.cmd == "download":
        stats = download(date.fromisoformat(a.start), date.fromisoformat(a.end),
                         kinds=tuple(a.kinds.split(",")), raw=Path(a.raw),
                         calendar=load_calendar())
        for k, v in stats.items():
            print(f"{k:5s} cached {v['have']:5d}  fetched {v['got']:5d}  missing {v['none']:5d}")
    else:
        for k, v in build(Path(a.raw), Path(a.out)).items():
            print(f"{k:18s} {v:>10,d} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
