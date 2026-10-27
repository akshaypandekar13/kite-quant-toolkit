"""Record live ticks with 5-level market depth from Kite. Records only;
never places an order.

Route A (retail microstructure) needs order-book history that no archive
gives us: Kite serves candles and live ticks, never past depth. So this
records it, one market day per process:

    python -m scripts.depth_recorder --freeze            # once: fix the universe
    python -m scripts.depth_recorder                     # each market day, ~09:05 IST

Instruments, chosen fresh each morning from Kite's instrument dump:
  - the frozen top-N F&O stocks (`--freeze` ranks them by futures turnover in
    data/fno): the cash share AND the near-month future
  - NIFTY and BANKNIFTY near-month futures, and the two index levels
  - NIFTY options, nearest expiry and nearest monthly, strikes within
    +/- OPT_BAND of spot at the open
about 330 instruments, under Kite's 3,000-per-connection limit.

Output, data/depth/YYYY-MM-DD/:
  instruments.csv           token -> symbol, segment, expiry, strike, type, lot, tick
  ticks_HH.csv.gz           one row per tick (hourly files; gzip members, so a
                            crash loses at most the unflushed tail)
  status.log                per-minute tick counts and connection events

Each row carries `recv_ns` (this machine's clock when the tick arrived) AND
`exch_ts` (the exchange's timestamp). The gap between them IS the latency a
retail strategy would have traded with, and the research needs it.

Timestamps are stored as epoch seconds: kiteconnect builds them with
`datetime.fromtimestamp`, i.e. in this machine's local zone, and `.timestamp()`
undoes that exactly whatever the zone (a VPS in UTC included).

The access token expires daily; refresh it first (python -m scripts.kite_token).
The process exits at STOP_AT (15:31 IST); one fresh process per day also
sidesteps the wedged-Twisted-reactor failure the live bot hit in June.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
import queue
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("depth_recorder")

OUT = Path("data/depth")
UNIVERSE_FILE = Path("depth_universe.txt")
TOP_N = 100
OPT_BAND = 0.03
STOP_AT = (15, 31)
LEVELS = 5
INDEX_TS = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}

COLS = (["recv_ns", "exch_ts", "token", "ltp", "ltq", "ltt", "volume", "atp",
         "tbq", "tsq", "oi"]
        + [f"{s}{i}{f}" for s in ("b", "a") for i in range(1, LEVELS + 1)
           for f in ("p", "q", "o")])


def _epoch(ts) -> Optional[float]:
    try:
        return ts.timestamp() if ts is not None else None
    except Exception:
        return None


def tick_row(t: dict, recv_ns: int) -> list:
    """One CSV row from a kiteconnect tick dict (full, quote or index mode).
    Missing fields are blank, never zero: a zero bid is a real state."""
    row = [recv_ns, _epoch(t.get("exchange_timestamp")), t.get("instrument_token"),
           t.get("last_price"), t.get("last_traded_quantity"),
           _epoch(t.get("last_trade_time")), t.get("volume_traded"),
           t.get("average_traded_price"), t.get("total_buy_quantity"),
           t.get("total_sell_quantity"), t.get("oi")]
    depth = t.get("depth") or {}
    for side in ("buy", "sell"):
        lv = depth.get(side) or []
        for i in range(LEVELS):
            if i < len(lv):
                row += [lv[i].get("price"), lv[i].get("quantity"), lv[i].get("orders")]
            else:
                row += [None, None, None]
    return row


# ── universe ────────────────────────────────────────────────────────────────

def freeze_universe(futures_csv: Path, top: int, out: Path, window: int = 20) -> List[str]:
    """Top-N stock underlyings by median daily futures turnover over the last
    `window` sessions in the F&O history; written once, read every day."""
    f = pd.read_csv(futures_csv, usecols=["date", "instrument", "symbol", "value_lakh"],
                    parse_dates=["date"])
    f = f[f.instrument == "FUTSTK"]
    last = sorted(f.date.unique())[-window:]
    v = f[f.date.isin(last)].groupby(["date", "symbol"]).value_lakh.sum().unstack().median()
    syms = sorted(v.nlargest(top).index)
    out.write_text("\n".join(syms) + "\n")
    return syms


def select(nse: pd.DataFrame, nfo: pd.DataFrame, stocks: List[str], spot: float,
           today: date, band: float = OPT_BAND) -> pd.DataFrame:
    """The day's instruments from Kite instrument dumps (as DataFrames).
    Columns kept: token, symbol, kind, tradingsymbol, expiry, strike, type,
    lot, tick."""
    rows = []

    def add(r, symbol, kind):
        rows.append(dict(token=int(r.instrument_token), symbol=symbol, kind=kind,
                         tradingsymbol=r.tradingsymbol, segment=r.segment,
                         expiry=r.expiry if pd.notna(r.expiry) else None,
                         strike=r.strike, type=r.instrument_type,
                         lot=r.lot_size, tick=r.tick_size))

    nfo = nfo[pd.to_datetime(nfo.expiry).dt.date >= today]
    futs = nfo[nfo.instrument_type == "FUT"]
    eq = nse[nse.segment == "NSE"]
    for s in stocks:
        cash = eq[(eq.tradingsymbol == s) & (eq.instrument_type == "EQ")]
        fut = futs[futs.name == s].sort_values("expiry")
        if cash.empty or fut.empty:
            logger.warning(f"{s}: no cash or future today, skipped")
            continue
        add(cash.iloc[0], s, "cash")
        add(fut.iloc[0], s, "fut")
    for name, idx in INDEX_TS.items():
        f = futs[futs.name == name].sort_values("expiry")
        if not f.empty:
            add(f.iloc[0], name, "fut")
        i = nse[nse.tradingsymbol == idx]
        if not i.empty:
            add(i.iloc[0], name, "index")
    opts = nfo[(nfo.name == "NIFTY") & nfo.instrument_type.isin(["CE", "PE"])]
    if len(opts) and spot:
        exps = sorted(pd.to_datetime(opts.expiry).dt.date.unique())
        monthly = [e for e in exps if not any(
            x.year == e.year and x.month == e.month and x > e for x in exps)]
        keep = {exps[0], monthly[0]} if monthly else {exps[0]}
        o = opts[pd.to_datetime(opts.expiry).dt.date.isin(keep)
                 & (opts.strike >= spot * (1 - band)) & (opts.strike <= spot * (1 + band))]
        for r in o.itertuples(index=False):
            add(r, "NIFTY", "opt")
    return pd.DataFrame(rows).drop_duplicates("token")


# ── writer ──────────────────────────────────────────────────────────────────

class TickWriter:
    """Consumes rows from a queue on its own thread; hourly gzip files.
    The WebSocket callback only enqueues, so a slow disk never stalls ticks."""

    def __init__(self, day_dir: Path, clock=time.time):
        self.dir = day_dir
        self.q: "queue.Queue[Optional[list]]" = queue.Queue()
        self.clock = clock
        self._fh = None
        self._hour = None
        self._csv = None
        self.rows = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def put(self, row: list):
        self.q.put(row)

    def _open(self, hour: int):
        if self._fh:
            self._fh.close()
        path = self.dir / f"ticks_{hour:02d}.csv.gz"
        new = not path.exists()
        self._fh = gzip.open(path, "at", newline="")
        self._csv = csv.writer(self._fh)
        if new:
            self._csv.writerow(COLS)
        self._hour = hour

    def _run(self):
        last_flush = self.clock()
        while True:
            row = self.q.get()
            if row is None:
                break
            hour = ist_hour(row[0])
            if hour != self._hour:
                self._open(hour)
            self._csv.writerow(row)
            self.rows += 1
            if self.clock() - last_flush > 5:
                self._fh.flush()
                last_flush = self.clock()
        if self._fh:
            self._fh.close()

    def stop(self):
        self.q.put(None)
        self.thread.join(timeout=60)


def ist_hour(recv_ns: int) -> int:
    """IST hour of an epoch-ns stamp, whatever this machine's time zone."""
    return int((recv_ns / 1e9 + 19800) % 86400 // 3600)


def stop_time(now: datetime) -> datetime:
    return now.replace(hour=STOP_AT[0], minute=STOP_AT[1], second=0, microsecond=0)


# ── live ────────────────────────────────────────────────────────────────────

def record(api_key: str, access_token: str, stocks: List[str], out: Path) -> int:
    from kiteconnect import KiteConnect, KiteTicker
    from src.kite_env import patch_twisted_signals
    patch_twisted_signals()
    from src.clock import ist_now

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    now = ist_now().replace(tzinfo=None)
    day_dir = out / f"{now:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    nse = pd.DataFrame(kite.instruments("NSE"))
    nfo = pd.DataFrame(kite.instruments("NFO"))
    spot = kite.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]
    inst = select(nse, nfo, stocks, spot, now.date())
    inst.to_csv(day_dir / "instruments.csv", index=False)
    tokens = inst.token.tolist()

    status = open(day_dir / "status.log", "a")

    def log(msg):
        status.write(f"{datetime.now():%H:%M:%S} {msg}\n")
        status.flush()
        logger.info(msg)

    log(f"{len(tokens)} instruments ({inst.kind.value_counts().to_dict()}), NIFTY spot {spot}")
    w = TickWriter(day_dir)
    w.start()
    counter = {"n": 0}

    ws = KiteTicker(api_key, access_token, reconnect=True,
                    reconnect_max_tries=300, reconnect_max_delay=60)

    def on_ticks(_ws, ticks):
        ns = time.time_ns()
        for t in ticks:
            w.put(tick_row(t, ns))
        counter["n"] += len(ticks)

    def on_connect(_ws, _resp):
        for i in range(0, len(tokens), 100):
            batch = tokens[i:i + 100]
            _ws.subscribe(batch)
            _ws.set_mode(_ws.MODE_FULL, batch)
        log(f"connected; subscribed {len(tokens)} in FULL mode")

    ws.on_ticks = on_ticks
    ws.on_connect = on_connect
    ws.on_close = lambda _ws, code, reason: log(f"closed {code} {reason}")
    ws.on_error = lambda _ws, code, reason: log(f"error {code} {reason}")
    ws.on_reconnect = lambda _ws, n: log(f"reconnecting, attempt {n}")
    ws.connect(threaded=True)

    end = stop_time(now)
    try:
        while ist_now().replace(tzinfo=None) < end:
            time.sleep(60)
            log(f"ticks last minute {counter['n']}, rows written {w.rows}, queue {w.q.qsize()}")
            counter["n"] = 0
    except KeyboardInterrupt:
        log("interrupted")
    finally:
        try:
            ws.close()
        except Exception:
            pass
        w.stop()
        log(f"stopped; {w.rows} rows")
        status.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", action="store_true",
                    help="write the stock universe from data/fno/futures.csv.gz and exit")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--universe", default=str(UNIVERSE_FILE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.freeze:
        syms = freeze_universe(Path("data/fno/futures.csv.gz"), a.top, Path(a.universe))
        print(f"froze {len(syms)} symbols -> {a.universe}")
        return 0
    from src.kite_env import load_dotenv as _load_dotenv
    _load_dotenv()
    key, tok = os.getenv("ZERODHA_API_KEY"), os.getenv("ZERODHA_ACCESS_TOKEN")
    if not key or not tok:
        raise SystemExit("Set ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN "
                         "(python -m scripts.kite_token refreshes the token).")
    stocks = [s.strip() for s in Path(a.universe).read_text().split() if s.strip()]
    return record(key, tok, stocks, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
