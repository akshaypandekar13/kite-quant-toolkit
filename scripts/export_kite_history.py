"""
Kite historical export
======================
Pulls the OHLCV history the offline replay harness needs and writes it as a
local store that `src/replay.py` can read without touching the network.

    python -m scripts.export_kite_history --symbols-from universe.csv --top 100 \
        --start 2024-09-01 --end 2026-09-01 --out data/history

What it writes, under `--out`:

    5minute/<SYMBOL>.parquet   intraday bars, the replay's primary input
    day/<SYMBOL>.parquet       daily bars for ATR / ADR / the excursion model
    5minute/NIFTY_50.parquet   index bars for the RS filter and the regime detector
    day/NIFTY_50.parquet
    manifest.json              what was pulled, when, and every gap

Columns match what `DataFetcherBase._validate_dataframe` expects downstream:
a tz-naive IST DatetimeIndex plus Open/High/Low/Close/Volume.

WHY A LOCAL STORE AND NOT A LIVE FETCHER
----------------------------------------
Kite's historical endpoint is rate-limited to ~3 requests/second and caps a
5-minute request at 100 days. A single replay pass over 100 symbols x 2 years
is ~700 requests; a parameter sweep is hundreds of passes. Pulling once and
replaying from disk turns a 4-minute-per-pass network bill into a
seconds-per-pass local read, which is the difference between testing a
hypothesis and guessing at it.

RESUMABILITY
------------
Each (symbol, timeframe) chunk is written as it arrives and skipped on a
re-run unless --force. A token expiry or a dropped connection costs you the
chunk in flight, not the export.

WHAT THIS CANNOT GIVE YOU
-------------------------
Candles are not fills. There is no spread, no depth, and no queue position in
this data, so any backtest built on it assumes you traded at the printed
price. That bias is one-directional (optimistic) and it is not small at the
stop distances this system uses: 5bps of slippage on a 0.5% stop is 0.10R,
comparable to the entire measured friction. Treat replay output as an upper
bound.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._io import atomic_path, write_csv, write_text  # noqa: E402

from src.symbols import canonical_symbol  # noqa: E402

logger = logging.getLogger("export_kite")

#: Kite caps a historical request by timeframe. These are the documented
#: per-request day limits; exceeding one returns an empty body rather than an
#: error, which is why the chunker enforces them instead of retrying.
CHUNK_DAYS = {
    "minute": 60,
    "3minute": 90,
    "5minute": 90,
    "10minute": 90,
    "15minute": 180,
    "30minute": 180,
    "60minute": 365,
    "day": 2000,
}

#: Kite's historical endpoint permits ~3 requests/second. Sleeping 0.34s
#: between calls keeps a long export inside that without a token bucket.
REQUEST_INTERVAL_SEC = 0.34

NIFTY_50_TOKEN = 256265
NIFTY_50_NAME = "NIFTY_50"

#: Indices exported alongside the stocks. NIFTY 50 is the default and has been
#: since the store was first built, so it stays the default here.
#:
#: WHY MORE THAN ONE INDEX MATTERS. The 98-name basket was ranked by liquidity
#: *as of the export date*, so anything that collapsed inside the window is
#: simply absent and the basket's measured drift is biased upward. An index is
#: the only survivorship-free control available: its published level is what a
#: holder actually earned, reconstitutions included. But NIFTY 50 is large-cap,
#: and `structural_edge.py`'s size split found the basket's overnight leg in
#: ALL THREE size terciles — so a large-cap control cannot speak to the mid and
#: small tiers where most of the effect sits. A midcap/smallcap index can, and
#: it is the one measurement that separates "a real premium in that size
#: segment" from "an artefact of picking today's survivors".
DEFAULT_INDICES = ("NIFTY 50",)


def index_store_name(tradingsymbol: str) -> str:
    """Kite's index tradingsymbol -> the name this store files it under.

    Kite spells indices with spaces ("NIFTY MIDCAP 100"); the store is a
    directory of `<symbol>.parquet`, and a space in a filename is a
    quoting hazard in every downstream `--symbols` invocation. Underscores
    throughout, which is already how NIFTY 50 is filed.
    """
    return "_".join(str(tradingsymbol).upper().split())


# ── symbol selection ────────────────────────────────────────────────────────


def symbols_from_universe(path: str, top: int,
                          min_atr_pct: float = 0.0,
                          sort_by: str = "median_turnover_20d") -> List[str]:
    """Pick the top-N symbols out of a Tier-0 universe.csv.

    Sorted by liquidity, not by anything predictive: the export is a data
    pull, and choosing names on a feature we later want to test would bake a
    selection effect into every result drawn from the store.
    """
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} has no `symbol` column")
    if min_atr_pct > 0 and "atr_pct" in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df["atr_pct"], errors="coerce") >= min_atr_pct]
        logger.info("ATR%% floor %.2f: %d -> %d symbols", min_atr_pct, before, len(df))
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    else:
        logger.warning("universe has no `%s` column — keeping file order", sort_by)
    return [canonical_symbol(s) for s in df["symbol"].head(top).tolist()]


def symbols_from_file(path: str) -> List[str]:
    """One bare symbol per line; blanks and `#` comments ignored."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(canonical_symbol(line))
    return out


# ── the export itself ───────────────────────────────────────────────────────


class KiteHistoryExporter:
    """Chunked, resumable pull of Kite historical candles to a local store."""

    def __init__(self, kite, out_dir: str, exchange: str = "NSE",
                 force: bool = False, request_interval: float = REQUEST_INTERVAL_SEC):
        self.kite = kite
        self.out = Path(out_dir)
        self.exchange = exchange
        self.force = force
        self.request_interval = request_interval
        self._tokens: Dict[str, int] = {}
        self._index_tokens: Dict[str, int] = {}
        self._last_request = 0.0
        self.stats = {"requests": 0, "rows": 0, "skipped": 0, "failed": 0}

    # -- plumbing --------------------------------------------------------

    def _pace(self) -> None:
        wait = self.request_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def load_instruments(self) -> None:
        """Build the tradingsymbol -> instrument_token maps once.

        Equities and indices are kept in SEPARATE maps. They share a namespace
        in the dump and an index is not an equity — it has no volume, so a
        VWAP built from it is NaN, which is the exact defect that made the
        regime detector's STRONG state mathematically unreachable for 220
        logged rows (`2b837a2`). Keeping them apart means a typo in an index
        name fails loudly instead of resolving to some equity.
        """
        logger.info("Downloading %s instrument master…", self.exchange)
        for row in self.kite.instruments(self.exchange):
            name = str(row.get("tradingsymbol", "")).upper()
            if not name:
                continue
            if row.get("segment") == "INDICES":
                self._index_tokens[name] = int(row["instrument_token"])
            elif (row.get("segment") == f"{self.exchange}-EQ"
                  or row.get("instrument_type") == "EQ"):
                self._tokens[name] = int(row["instrument_token"])
        logger.info("  %d equity instruments, %d indices mapped",
                    len(self._tokens), len(self._index_tokens))

    def available_indices(self) -> List[str]:
        """Index tradingsymbols the dump offers, for `--list-indices`."""
        return sorted(self._index_tokens)

    def token_for(self, symbol: str) -> Optional[int]:
        """Resolve a store name to an instrument token.

        Indices are looked up under their SPACED tradingsymbol, since the
        store files them with underscores. NIFTY_50 keeps its hardcoded token
        as a fallback so a dump that omits the INDICES segment — or an offline
        re-run — exports exactly what it always did.
        """
        upper = str(symbol).upper()
        spaced = upper.replace("_", " ")
        if spaced in self._index_tokens:
            return self._index_tokens[spaced]
        if upper == NIFTY_50_NAME:
            return NIFTY_50_TOKEN
        return self._tokens.get(canonical_symbol(symbol).upper())

    # -- fetch -----------------------------------------------------------

    def _fetch_chunk(self, token: int, timeframe: str,
                     start: date, end: date) -> Optional[pd.DataFrame]:
        self._pace()
        self.stats["requests"] += 1
        try:
            raw = self.kite.historical_data(
                instrument_token=token,
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
                interval=timeframe,
                continuous=False,
                oi=False,
            )
        except Exception as e:
            logger.warning("  chunk %s..%s failed: %s", start, end, e)
            self.stats["failed"] += 1
            return None
        if not raw:
            return None
        df = pd.DataFrame(raw)
        df = df.rename(columns={"date": "Datetime", "open": "Open", "high": "High",
                                "low": "Low", "close": "Close", "volume": "Volume"})
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        # Store tz-NAIVE IST. Kite returns tz-aware +05:30; the live path
        # normalises to naive, and mixing the two is the exact defect that
        # produced an object-dtype index and silently zeroed every day-sliced
        # field (see analyzer.py's corrupted-index tripwire).
        if getattr(df["Datetime"].dt, "tz", None) is not None:
            df["Datetime"] = df["Datetime"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        return df.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    def export_symbol(self, symbol: str, timeframe: str,
                      start: date, end: date) -> Optional[pd.DataFrame]:
        """Fetch one symbol/timeframe across as many chunks as the API needs."""
        path = self.out / timeframe / f"{symbol}.parquet"
        if path.exists() and not self.force:
            self.stats["skipped"] += 1
            logger.debug("  %s %s already present — skipping", symbol, timeframe)
            return None
        token = self.token_for(symbol)
        if token is None:
            logger.warning("  %s: no instrument token in %s — skipped", symbol, self.exchange)
            self.stats["failed"] += 1
            return None

        span = CHUNK_DAYS.get(timeframe, 90)
        frames, cursor = [], start
        while cursor <= end:
            chunk_end = min(cursor + timedelta(days=span - 1), end)
            part = self._fetch_chunk(token, timeframe, cursor, chunk_end)
            if part is not None and len(part):
                frames.append(part)
            cursor = chunk_end + timedelta(days=1)

        if not frames:
            logger.warning("  %s %s: no data returned", symbol, timeframe)
            self.stats["failed"] += 1
            return None

        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(df, path)
        self.stats["rows"] += len(df)
        logger.info("  %-14s %-8s %6d bars  %s .. %s", symbol, timeframe, len(df),
                    df.index[0].date(), df.index[-1].date())
        return df

    def run(self, symbols: Sequence[str], start: date, end: date,
            timeframes: Sequence[str] = ("5minute", "day"),
            daily_lead_days: int = 400,
            indices: Sequence[str] = DEFAULT_INDICES) -> Dict:
        """Export every symbol, then the indices, then write the manifest.

        `daily_lead_days` pulls daily bars further back than the intraday
        window. The excursion model reads 60 sessions plus an ATR warm-up
        before it will price the first replayed bar, so a daily series that
        starts on the same day as the intraday one leaves the first three
        months of the replay running on a fallback target.

        `indices` are filed under their underscored names and are NOT part of
        the stock basket — `structural_edge.py` reports them separately,
        because an index is the only reading in this store that no survivorship
        filter has touched.
        """
        self.load_instruments()
        coverage = {}
        index_names = [index_store_name(i) for i in indices]
        unknown = [i for i, n in zip(indices, index_names)
                   if self.token_for(n) is None]
        if unknown:
            # Loud, and before a multi-hour pull rather than after it. A
            # mistyped index that silently exported nothing would look
            # identical to "the effect is not there in that segment" — the
            # one conclusion this export exists to make trustworthy.
            raise SystemExit(
                f"unknown index name(s): {unknown}. Run --list-indices to see "
                f"what {self.exchange} actually offers.")
        todo = list(symbols) + index_names
        for i, sym in enumerate(todo, 1):
            logger.info("[%d/%d] %s", i, len(todo), sym)
            for tf in timeframes:
                tf_start = start - timedelta(days=daily_lead_days) if tf == "day" else start
                df = self.export_symbol(sym, tf, tf_start, end)
                if df is not None:
                    coverage.setdefault(sym, {})[tf] = {
                        "bars": len(df),
                        "first": str(df.index[0]),
                        "last": str(df.index[-1]),
                        "sessions": int(pd.Series(df.index.date).nunique()),
                    }
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "exchange": self.exchange,
            "start": str(start), "end": str(end),
            "timeframes": list(timeframes),
            "symbols_requested": len(todo),
            "stats": self.stats,
            "coverage": coverage,
        }
        self.out.mkdir(parents=True, exist_ok=True)
        write_text(self.out / "manifest.json", json.dumps(manifest, indent=2))
        return manifest


def _write(df: pd.DataFrame, path: Path) -> None:
    """Parquet when an engine is installed, CSV otherwise.

    The store is read back by `replay.HistoryStore`, which accepts either, so
    a missing pyarrow degrades the export to a larger file rather than
    failing it after an hour of API calls.
    """
    try:
        with atomic_path(path) as tmp:
            df.to_parquet(tmp)
    except ImportError as e:
        csv_path = path.with_suffix(".csv")
        logger.debug("parquet unavailable (%s) — writing %s", e, csv_path.name)
        write_csv(df, csv_path)


# ── entry point ─────────────────────────────────────────────────────────────


def build_kite_client(api_key: Optional[str] = None,
                      access_token: Optional[str] = None):
    """Authenticated KiteConnect from args or the standard env vars."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise SystemExit(
            "kiteconnect is not installed. Run:  pip install 'kiteconnect>=5.0.0'\n"
            "(or `pip install -e '.[zerodha]'` from trading_scanner/)")
    api_key = api_key or os.getenv("ZERODHA_API_KEY")
    access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN")
    if not api_key or not access_token:
        raise SystemExit(
            "Set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN (or pass --api-key / "
            "--access-token). Generate a fresh token with:  "
            "python -m scripts.kite_token")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--symbols-from", help="universe.csv to take the top-N from")
    src.add_argument("--symbols-file", help="text file, one bare symbol per line")
    src.add_argument("--symbols", help="comma-separated bare symbols")
    ap.add_argument("--top", type=int, default=100, help="with --symbols-from (default 100)")
    ap.add_argument("--min-atr-pct", type=float, default=0.0,
                    help="with --symbols-from: drop names below this daily ATR%%")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD, intraday window start")
    ap.add_argument("--end", default=str(date.today()), help="YYYY-MM-DD (default today)")
    ap.add_argument("--out", default="data/history", help="output store directory")
    ap.add_argument("--timeframes", default="5minute,day")
    ap.add_argument("--daily-lead-days", type=int, default=400,
                    help="extra daily history before --start (excursion warm-up)")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--indices", default=",".join(DEFAULT_INDICES),
                    help="comma-separated index tradingsymbols to export "
                         "alongside the stocks (default: %(default)s). Kite "
                         "spells them with spaces; they are filed with "
                         "underscores. An index is the only survivorship-free "
                         "control in the store, so a midcap/smallcap one is "
                         "what tests whether the basket's overnight drift is "
                         "a real premium or an artefact of picking today's "
                         "survivors.")
    ap.add_argument("--list-indices", action="store_true",
                    help="print the index names this exchange offers and exit")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--api-key"), ap.add_argument("--access-token")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(message)s")

    if a.list_indices:
        ex = KiteHistoryExporter(build_kite_client(a.api_key, a.access_token),
                                 a.out, exchange=a.exchange)
        ex.load_instruments()
        for name in ex.available_indices():
            logger.info("  %-28s -> %s", name, index_store_name(name))
        return 0

    if a.symbols_from:
        symbols = symbols_from_universe(a.symbols_from, a.top, a.min_atr_pct)
    elif a.symbols_file:
        symbols = symbols_from_file(a.symbols_file)
    else:
        symbols = [canonical_symbol(s) for s in a.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("no symbols selected")

    start = datetime.strptime(a.start, "%Y-%m-%d").date()
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
    if start >= end:
        raise SystemExit(f"--start {start} must be before --end {end}")

    indices = [i.strip() for i in a.indices.split(",") if i.strip()]
    logger.info("Exporting %d symbols + %d indices (%s), %s .. %s -> %s",
                len(symbols), len(indices), ", ".join(indices) or "none",
                start, end, a.out)
    exporter = KiteHistoryExporter(build_kite_client(a.api_key, a.access_token),
                                  a.out, exchange=a.exchange, force=a.force)
    m = exporter.run(symbols, start, end,
                     timeframes=tuple(t.strip() for t in a.timeframes.split(",")),
                     daily_lead_days=a.daily_lead_days,
                     indices=indices)

    s = m["stats"]
    logger.info("\nDone. %d requests, %d rows, %d skipped (already present), %d failed",
                s["requests"], s["rows"], s["skipped"], s["failed"])
    logger.info("Manifest: %s", Path(a.out) / "manifest.json")
    if s["failed"]:
        logger.warning("%d symbol/timeframe pulls failed — re-run to retry just those.",
                       s["failed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
