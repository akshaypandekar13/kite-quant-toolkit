"""
Tier 0 — nightly tradeable-universe build
=========================================

Builds the daily list of instruments the bot is *allowed* to trade, plus a
wide daily/weekly feature table over them. Runs after close; consumes no Kite
quota (NSE publishes all of this free over plain HTTP).

WHY THIS EXISTS
---------------
The bot has always traded a hand-curated watchlist of 53-65 large caps. That
list never needed structural filtering because every name on it was liquid,
in the 20% price band, and MIS-eligible by construction. The moment the
universe widens past a curated list, that stops being true and the filtering
becomes load-bearing: a stock in a 2% circuit band physically cannot deliver
a 2R intraday move, and an ASM/GSM name cannot be traded intraday at all.

It also fixes a sample-size problem. 65 symbols and 3 slots produced 162
trades in 23 sessions; at that rate a feature with a genuine +0.15R edge
needs ~350 trades before it clears significance. A ~900-name universe raises
candidate flow by an order of magnitude, which is the actual argument for
widening it — statistical power, not more filters.

WHAT IT DOES NOT DO
-------------------
Nothing here places, blocks, or scores a trade. It writes two files and
exits. Tier 1 (the intraday sweep) is what eventually reads them.

OUTPUT FORMAT
-------------
CSV, not parquet. The repo has no pyarrow/fastparquet dependency and the
table is ~900 rows x ~60 columns — small enough that the format buys nothing
and consistent with every other artefact the bot writes.

  universe.csv         one row per ELIGIBLE symbol, with its feature vector
  universe_report.json the exclusion funnel: what was dropped and by which
                       rule. Without this the eligible list is unfalsifiable.

DATA SOURCES (all free, all plain HTTP, no auth)
------------------------------------------------
  UDiFF CM bhavcopy   OHLCV + series, per session. Replaced the legacy
                      bhavcopy format on 2024-07-08.
                      /content/cm/BhavCopy_NSE_CM_0_0_0_{YYYYMMDD}_F_0000.csv.zip
  sec_bhavdata_full   adds DELIV_QTY / DELIV_PER — delivery percentage, which
                      separates genuine accumulation from intraday churn and
                      is not available from any broker API.
                      /products/content/sec_bhavdata_full_{DDMMYYYY}.csv
  EQUITY_L.csv        the listed-equity master (symbol, series, ISIN).
                      /content/equities/EQUITY_L.csv

⚠️  THE LIVE ENDPOINTS ARE UNVERIFIED FROM THE DEV CONTAINER — outbound
    access to nseindia.com is blocked there, so `NseHttpSource` is written
    from NSE's published formats and has NOT been exercised against the real
    host. Every parsing and feature path IS tested, against fixtures, via
    `LocalFileSource`. Run `scripts/build_universe.py --dry-run` once on a
    machine with real network access before trusting a nightly schedule.

SURVEILLANCE LISTS ARE OPT-IN, AND THEIR ABSENCE IS RECORDED
------------------------------------------------------------
ASM / GSM / F&O-ban / broker-MIS lists sit behind session-cookie'd endpoints
that are not reliably scriptable. Rather than pretend, this module accepts
them as optional local files. When one is absent the corresponding rule does
NOT run and the report says so explicitly — `surveillance_applied: false` —
so a downstream consumer can never mistake "not filtered" for "filtered and
clean". That distinction is the whole point of the report file.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── NSE endpoints ────────────────────────────────────────────────────────────

NSE_ARCHIVES = "https://nsearchives.nseindia.com"
BHAVCOPY_URL = NSE_ARCHIVES + "/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
DELIVERY_URL = NSE_ARCHIVES + "/products/content/sec_bhavdata_full_{dmy}.csv"
EQUITY_LIST_URL = NSE_ARCHIVES + "/content/equities/EQUITY_L.csv"

#: A 404 on a day younger than this MANY CALENDAR DAYS is treated as "not
#: published yet" rather than "does not exist".
#:
#: Sized for the calendar, not for the publication delay. The binding case is
#: a weekend: Friday's file first requested on Monday is 3 calendar days old,
#: and a Monday holiday pushes that to 4. A window of 3 excluded exactly the
#: Friday-on-Monday case — which is the most common way a session goes
#: missing, since nobody runs a build over the weekend.
#:
#: The cost of erring wide is a handful of repeat HTTP requests for a genuine
#: holiday. The cost of erring narrow is a trading session permanently absent
#: from the history, which cannot be regenerated. Not a close call.
_MISSING_MIN_AGE_DAYS = 6

#: Daily F&O ban list — a plain CSV on the same archive host as the bhavcopy.
FNO_BAN_URL = NSE_ARCHIVES + "/content/fo/fo_secban.csv"
#: Surveillance lists. These live on the WWW host behind the cookie, not the
#: archive host, and are JSON rather than CSV.
ASM_URL = "https://www.nseindia.com/api/reportASM"
GSM_URL = "https://www.nseindia.com/api/reportGSM"

# NSE rejects requests without a browser-shaped User-Agent.
_HTTP_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "text/csv,application/zip,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

#: UDiFF column -> our canonical name. UDiFF replaced the legacy bhavcopy on
#: 2024-07-08; the legacy names are kept as aliases so an archived file from
#: before the switch still parses.
_UDIFF_MAP = {
    "TckrSymb": "symbol", "SctySrs": "series", "TradDt": "date",
    "OpnPric": "open", "HghPric": "high", "LwPric": "low",
    "ClsPric": "close", "PrvsClsgPric": "prev_close",
    "TtlTradgVol": "volume", "TtlTrfVal": "turnover",
    "TtlNbOfTxsExctd": "trades", "FinInstrmTp": "instrument_type",
}
_LEGACY_MAP = {
    "SYMBOL": "symbol", "SERIES": "series", "TIMESTAMP": "date",
    "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
    "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume", "TOTTRDVAL": "turnover",
    "TOTALTRADES": "trades",
}

#: The equity series that settle normally and permit intraday. Everything
#: else is excluded — BE and BZ are Trade-to-Trade (compulsory delivery, no
#: intraday square-off at all), SM/ST are SME, and the rest are debt or
#: rights instruments that do not behave like the tape this bot models.
TRADEABLE_SERIES = frozenset({"EQ"})

#: NSE equity tick size, in rupees.
NSE_TICK_SIZE = 0.05


# ── Exclusion funnel ─────────────────────────────────────────────────────────

@dataclass
class ExclusionResult:
    """One rule's effect on the universe, for the audit report."""
    name: str
    applied: bool
    removed: int
    remaining: int
    detail: str = ""

    def to_dict(self) -> Dict:
        return {"rule": self.name, "applied": self.applied,
                "removed": self.removed, "remaining": self.remaining,
                "detail": self.detail}


@dataclass
class UniverseConfig:
    """Thresholds for the hard exclusions.

    These are structural facts about tradeability, not tuned parameters —
    each describes an instrument that cannot deliver the trade the engine
    places. They are configurable because exchange conventions change, not
    because they are meant to be swept.
    """
    min_price: float = 50.0
    #: 20-day median turnover, in rupees. At Rs 225k per position you are a
    #: meaningful share of a thin book and exit slippage swamps a 0.38R
    #: median excursion.
    min_turnover_rupees: float = 25_00_00_000.0      # Rs 25 crore
    #: Minimum share of recent sessions whose implied VWAP (turnover/volume)
    #: lands inside that day's high-low range. Below this the published
    #: liquidity figures for that symbol do not describe its own price, so
    #: neither the turnover ranking nor the RVOL denominator can be trusted.
    #: 0.0 disables the check (the rule is then recorded as skipped).
    min_vwap_consistency: float = 0.8
    #: Average Daily Range as % of close. A stock whose normal day is
    #: narrower than the stop plus costs has no room to pay.
    min_adr_pct: float = 1.5
    #: One tick as a fraction of price. The lesson the watchlist already
    #: learned from IDEA (Rs 8 stock: one tick = 0.6% of price, so the spread
    #: alone is a third of the stop) generalised into a rule.
    max_tick_pct: float = 0.05
    #: Circuit band floor. A 2% band cannot contain a 2R move when the median
    #: stop is ~1.4% of price. Only applied when a band file is supplied.
    min_circuit_band_pct: float = 10.0
    #: Sessions of history to build features from. 90 calendar days is ~60
    #: sessions — enough for SMA50, ADR14, and a 55-day Donchian.
    history_days: int = 120
    #: Minimum sessions a symbol must appear in to be featurised at all.
    min_sessions: int = 30


# ── Sources ──────────────────────────────────────────────────────────────────

class UniverseSource(ABC):
    """Where the raw exchange files come from.

    Split out so every parsing, merging and feature path can be tested
    without network access — `LocalFileSource` is not a mock, it is the same
    code path reading files from disk.
    """

    @abstractmethod
    def bhavcopy(self, day: date) -> Optional[pd.DataFrame]:
        """One session's OHLCV. None when that day has no file (holiday)."""

    @abstractmethod
    def delivery(self, day: date) -> Optional[pd.DataFrame]:
        """One session's delivery data. None when unavailable."""

    @abstractmethod
    def equity_list(self) -> Optional[pd.DataFrame]:
        """The listed-equity master. None when unavailable."""


def _normalise_bhavcopy(raw: pd.DataFrame, day: date) -> pd.DataFrame:
    """Map a UDiFF or legacy bhavcopy frame onto our canonical columns.

    Defensive on purpose: NSE has changed this format once already
    (2024-07-08) and the column set carries reserved fields that have been
    repurposed before. Anything we do not recognise is dropped rather than
    guessed at.
    """
    raw = raw.copy()
    # sec_bhavdata_full ships column names with LEADING SPACES; the CM
    # bhavcopy has been seen with trailing ones. Strip before matching or
    # every lookup silently misses.
    raw.columns = [str(c).strip() for c in raw.columns]

    mapping = _UDIFF_MAP if "TckrSymb" in raw.columns else _LEGACY_MAP
    present = {src: dst for src, dst in mapping.items() if src in raw.columns}
    missing_core = {"symbol", "series", "close"} - set(present.values())
    if missing_core:
        raise ValueError(
            f"bhavcopy for {day} is missing core columns {sorted(missing_core)}; "
            f"got {sorted(raw.columns)[:12]}..."
        )

    df = raw[list(present)].rename(columns=present)

    # UDiFF carries futures and options in the same segment file; keep cash.
    if "instrument_type" in df.columns:
        df = df[df["instrument_type"].astype(str).str.upper().isin({"STK", "EQ", ""})]

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["series"] = df["series"].astype(str).str.strip().str.upper()
    for col in ("open", "high", "low", "close", "prev_close", "volume",
                "turnover", "trades"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df["date"] = pd.Timestamp(day)
    return df.dropna(subset=["close"])


def _normalise_delivery(raw: pd.DataFrame, day: date) -> pd.DataFrame:
    """Map sec_bhavdata_full onto symbol/series/deliv_qty/deliv_pct."""
    raw = raw.copy()
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    if "SYMBOL" not in raw.columns:
        raise ValueError(f"delivery file for {day} has no SYMBOL column")

    out = pd.DataFrame({
        "symbol": raw["SYMBOL"].astype(str).str.strip().str.upper(),
        "series": raw.get("SERIES", pd.Series("", index=raw.index))
                     .astype(str).str.strip().str.upper(),
    })
    # DELIV_QTY / DELIV_PER are literally '-' on the days a symbol had no
    # deliverable position (and for non-EQ series). Coerce, do not fail.
    out["deliv_qty"] = pd.to_numeric(raw.get("DELIV_QTY"), errors="coerce")
    out["deliv_pct"] = pd.to_numeric(raw.get("DELIV_PER"), errors="coerce")
    out["date"] = pd.Timestamp(day)
    return out


class NseHttpSource(UniverseSource):
    """Fetches from NSE's public archive, with an on-disk cache.

    The cache is what makes a nightly run cheap: history is built once and
    each subsequent night downloads a single new session. It also means the
    whole builder can be re-run offline against whatever has been collected.

    NOT EXERCISED against the live host from the dev container (egress to
    nseindia.com is blocked there). See the module docstring.
    """

    def __init__(self, cache_dir: str = "data/nse_cache", timeout: float = 30.0):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._session = None
        #: Days known to have no file (holidays/weekends). Cached so a repeat
        #: run does not re-request 20 known-missing days every night.
        #: {"ASM": n, "GSM": n} from the last surveillance() call. A zero
        #: means that source contributed nothing, which makes the rule only
        #: PARTIALLY applied — see `missing_sources`.
        self.last_surveillance_sources: Dict[str, int] = {}
        self._missing_marker = self.cache / "_missing_days.json"
        self._missing: set = set(self._load_missing())

    # -- plumbing ----------------------------------------------------------

    def _load_missing(self) -> List[str]:
        try:
            return json.loads(self._missing_marker.read_text())
        except Exception:
            return []

    def forget_missing(self) -> int:
        """Clear the known-missing marker so every day is re-requested once.

        The marker is a pure cache: it exists to stop a nightly run
        re-requesting 20 known holidays. Clearing it costs a handful of
        extra HTTP requests and nothing else, so it is the right recovery
        whenever a day was recorded missing for the wrong reason — a build
        run before NSE published, or a window that was too narrow.
        """
        n = len(self._missing)
        self._missing = set()
        try:
            if self._missing_marker.exists():
                self._missing_marker.unlink()
        except Exception as e:
            logger.warning(f"could not clear the missing-day marker: {e}")
        logger.info(f"cleared {n} known-missing day(s) — all will be retried")
        return n

    def _mark_missing(self, key: str) -> None:
        self._missing.add(key)
        try:
            self._missing_marker.write_text(json.dumps(sorted(self._missing)))
        except Exception as e:
            logger.debug(f"could not persist missing-day marker: {e}")

    def _get_session(self):
        import requests
        if self._session is None:
            s = requests.Session()
            s.headers.update(_HTTP_HEADERS)
            # NSE hands out a cookie on the landing page and rejects archive
            # requests made without one. Best-effort: the archive host has
            # served direct requests too, so a failure here is not fatal.
            try:
                s.get("https://www.nseindia.com", timeout=self.timeout)
            except Exception as e:
                logger.debug(f"NSE cookie warm-up failed (continuing): {e}")
            self._session = s
        return self._session

    def _fetch(self, url: str, cache_name: str,
               day: Optional[date] = None) -> Optional[bytes]:
        cached = self.cache / cache_name
        if cached.exists():
            return cached.read_bytes()
        if cache_name in self._missing:
            return None
        try:
            resp = self._get_session().get(url, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"fetch failed for {url}: {e}")
            return None
        if resp.status_code == 404:
            # A 404 means two completely different things depending on age.
            #
            # On an OLD day it is permanent: a trading holiday, or a date
            # before the file series existed. Caching that saves re-requesting
            # 20 known-missing days every night, which is why the marker
            # exists at all.
            #
            # On a RECENT day it is transient: NSE publishes the bhavcopy
            # after the close (~18:00-19:00 IST) and sec_bhavdata_full later
            # still. A build run before publication would otherwise mark the
            # session permanently missing — and because the marker persists,
            # EVERY later run would skip it too. One early run would silently
            # cost that session's history forever, and the delivery file has a
            # wider window than the bhavcopy, so a build timed between the two
            # would lose delivery data for a day it otherwise fetched fine.
            if day is not None and (date.today() - day).days < _MISSING_MIN_AGE_DAYS:
                logger.info(
                    f"no file yet for {day} (404) — not marking it missing; "
                    f"NSE publishes after the close, so re-run later.")
                return None
            logger.debug(f"no file at {url} (404)")
            self._mark_missing(cache_name)
            return None
        if resp.status_code != 200:
            logger.warning(f"fetch for {url} returned HTTP {resp.status_code}")
            return None
        try:
            cached.write_bytes(resp.content)
        except Exception as e:
            logger.debug(f"could not cache {cache_name}: {e}")
        return resp.content

    # -- surveillance / ban lists ------------------------------------------
    #
    # THE CONTRACT FOR EVERYTHING BELOW: return None when the list could not
    # be obtained, and a list (possibly empty) only when it genuinely was.
    #
    # None and [] mean OPPOSITE things downstream. [] says "nothing is banned
    # today" and the rule runs and removes nothing. None says "we do not know
    # what is banned" and the rule is recorded as SKIPPED with a warning. A
    # fetcher that swallowed an error into [] would turn an unfiltered
    # universe into one that reports itself as filtered — the single worst
    # outcome this module can produce, and worse than not fetching at all.
    #
    # So every failure path here returns None, and a wrong URL is a clean
    # no-op that leaves today's behaviour exactly as it is.

    @staticmethod
    def _symbols_from_csv(blob: bytes) -> Optional[List[str]]:
        """Symbols from a CSV whose shape NSE has changed before.

        Tolerant by design: takes a column named like a symbol if there is
        one, else the last column of a single/two-column file. Returns None
        rather than guessing when nothing looks like a symbol column.
        """
        try:
            df = pd.read_csv(io.BytesIO(blob))
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(blob), skiprows=1)
            except Exception as e:
                logger.warning(f"could not parse list CSV: {e}")
                return None
        if df.empty:
            # A real empty list is meaningful — no bans today — but an empty
            # PARSE is not, and the two are indistinguishable here.
            logger.warning("list CSV parsed to zero rows; treating as unknown")
            return None
        cols = {str(c).strip().lower(): c for c in df.columns}
        col = next((cols[k] for k in cols if "symbol" in k), None)
        if col is None and len(df.columns) <= 2:
            col = df.columns[-1]
        if col is None:
            logger.warning(f"no symbol column in list CSV (cols={list(df.columns)})")
            return None
        out = [str(v).strip().upper() for v in df[col].dropna()
               if str(v).strip() and str(v).strip().lower() != "symbol"]
        return out or None

    @staticmethod
    def _symbols_from_json(blob: bytes) -> Optional[List[str]]:
        """Every symbol anywhere in the payload, whatever shape it arrives in.

        The three live endpoints already disagree with each other, which is
        why this walks the structure instead of naming a path:

            GSM   a bare list of row dicts
            F&O   (CSV, handled elsewhere)
            ASM   {"longterm": {"data": [...]}, "shortterm": {"data": [...]}}

        An earlier version read `payload["data"]`, which is none of these for
        ASM — so 201 surveillance names (132 long-term + 69 short-term)
        parsed as None and the rule stayed skipped. It failed SAFE, which is
        the design working, but it failed: ASM names carry 100% margin and
        are MIS-blocked, so they are the ones that cost money.

        Recursing also means a future reshape by NSE is a no-op here rather
        than a silent regression to None.
        """
        try:
            payload = json.loads(blob.decode("utf-8", "replace"))
        except Exception as e:
            logger.warning(f"could not parse list JSON: {e}")
            return None

        out: List[str] = []
        seen = set()

        def walk(node, depth: int = 0) -> None:
            if depth > 8:                        # malformed / cyclic guard
                return
            if isinstance(node, dict):
                for key in ("symbol", "Symbol", "SYMBOL"):
                    v = node.get(key)
                    if isinstance(v, str) and v.strip():
                        sym = v.strip().upper()
                        if sym not in seen:
                            seen.add(sym)
                            out.append(sym)
                        return                   # a row yields one symbol
                for v in node.values():
                    walk(v, depth + 1)
            elif isinstance(node, list):
                for v in node:
                    walk(v, depth + 1)

        walk(payload)
        return out or None

    def _daily_cache_name(self, stem: str, day: date) -> str:
        """Date-stamped: these lists change every session, unlike a bhavcopy
        for a fixed day, so yesterday's copy must not be served as today's."""
        return f"{stem}_{day:%Y%m%d}.raw"

    def fno_ban(self, day: Optional[date] = None) -> Optional[List[str]]:
        """Symbols in the F&O ban period today, or None if unobtainable."""
        day = day or date.today()
        blob = self._fetch(FNO_BAN_URL, self._daily_cache_name("fnoban", day))
        if blob is None:
            return None
        syms = self._symbols_from_csv(blob)
        logger.info(f"F&O ban list: {len(syms)} symbols" if syms
                    else "F&O ban list unavailable — rule will be SKIPPED")
        return syms

    def surveillance(self, day: Optional[date] = None) -> Optional[List[str]]:
        """Union of the ASM and GSM lists, or None if NEITHER could be read.

        A PARTIAL result is still returned when only one source responds, but
        `last_surveillance_sources` records which ones contributed and callers
        MUST surface that — see the note on `missing_sources` below.
        """
        day = day or date.today()
        found: set = set()
        self.last_surveillance_sources = {}
        for label, url, stem in (("ASM", ASM_URL, "asm"), ("GSM", GSM_URL, "gsm")):
            blob = self._fetch(url, self._daily_cache_name(stem, day))
            syms = self._symbols_from_json(blob) if blob is not None else None
            self.last_surveillance_sources[label] = len(syms) if syms else 0
            if syms:
                found.update(syms)
            else:
                logger.warning(f"{label} list unavailable")
        if not found:
            logger.warning("surveillance lists unavailable — rule will be SKIPPED")
            return None
        counts = ", ".join(f"{k}:{v}" for k, v in
                           self.last_surveillance_sources.items())
        logger.info(f"surveillance list: {len(found)} symbols ({counts})")
        return sorted(found)

    @property
    def missing_sources(self) -> List[str]:
        """Surveillance sources that returned nothing on the last fetch.

        THIS IS NOT COSMETIC. The module's guarantee is that a rule which did
        not run is recorded applied=false. That guarantee was built for the
        binary case and `surveillance()` merges TWO sources, so a half-fetch
        produced a rule reported as APPLIED that had only checked half the
        names — strictly worse than skipping, because skipping at least
        printed "universe is NOT surveillance-filtered" while this printed
        nothing at all.

        Observed live on 2026-08-20: ASM failed to parse, GSM succeeded, the
        funnel recorded `surveillance_asm_gsm applied=true removed=0`, and 33
        genuinely ASM-listed symbols — MIS-blocked at the broker — sat in the
        eligible universe with no warning anywhere.
        """
        return [k for k, v in getattr(self, "last_surveillance_sources", {}).items()
                if not v]

    # -- UniverseSource ----------------------------------------------------

    def bhavcopy(self, day: date) -> Optional[pd.DataFrame]:
        name = f"bhav_{day:%Y%m%d}.csv.zip"
        blob = self._fetch(BHAVCOPY_URL.format(ymd=f"{day:%Y%m%d}"), name, day)
        if blob is None:
            return None
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                inner = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                raw = pd.read_csv(zf.open(inner))
        except Exception as e:
            logger.warning(f"could not parse bhavcopy for {day}: {e}")
            return None
        return _normalise_bhavcopy(raw, day)

    def delivery(self, day: date) -> Optional[pd.DataFrame]:
        name = f"deliv_{day:%Y%m%d}.csv"
        blob = self._fetch(DELIVERY_URL.format(dmy=f"{day:%d%m%Y}"), name, day)
        if blob is None:
            return None
        try:
            return _normalise_delivery(pd.read_csv(io.BytesIO(blob)), day)
        except Exception as e:
            logger.warning(f"could not parse delivery file for {day}: {e}")
            return None

    def equity_list(self) -> Optional[pd.DataFrame]:
        # Deliberately NOT cached by date — this one is a living document and
        # a stale copy would keep delisted symbols in the universe.
        try:
            resp = self._get_session().get(EQUITY_LIST_URL, timeout=self.timeout)
            resp.raise_for_status()
            raw = pd.read_csv(io.BytesIO(resp.content))
        except Exception as e:
            logger.warning(f"equity list fetch failed: {e}")
            return None
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        return pd.DataFrame({
            "symbol": raw["SYMBOL"].astype(str).str.strip().str.upper(),
            "series": raw["SERIES"].astype(str).str.strip().str.upper(),
            "isin": raw.get("ISIN NUMBER", pd.Series("", index=raw.index)),
        })


class LocalFileSource(UniverseSource):
    """Reads pre-downloaded files from a directory.

    Same parsing path as NseHttpSource — this is how every feature and
    exclusion is tested without touching the network, and how the builder can
    be re-run against an archived snapshot.

    Expected layout:
        <dir>/bhav_YYYYMMDD.csv[.zip]
        <dir>/deliv_YYYYMMDD.csv
        <dir>/equity_list.csv
    """

    def __init__(self, directory: str):
        self.dir = Path(directory)

    def bhavcopy(self, day: date) -> Optional[pd.DataFrame]:
        zipped = self.dir / f"bhav_{day:%Y%m%d}.csv.zip"
        plain = self.dir / f"bhav_{day:%Y%m%d}.csv"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                inner = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                return _normalise_bhavcopy(pd.read_csv(zf.open(inner)), day)
        if plain.exists():
            return _normalise_bhavcopy(pd.read_csv(plain), day)
        return None

    def delivery(self, day: date) -> Optional[pd.DataFrame]:
        path = self.dir / f"deliv_{day:%Y%m%d}.csv"
        return _normalise_delivery(pd.read_csv(path), day) if path.exists() else None

    def equity_list(self) -> Optional[pd.DataFrame]:
        path = self.dir / "equity_list.csv"
        if not path.exists():
            return None
        raw = pd.read_csv(path)
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        return pd.DataFrame({
            "symbol": raw["SYMBOL"].astype(str).str.strip().str.upper(),
            "series": raw["SERIES"].astype(str).str.strip().str.upper(),
            "isin": raw.get("ISIN NUMBER", pd.Series("", index=raw.index)),
        })


# ── Feature computation ──────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> float:
    """Wilder RSI, last value. NaN when there is not enough history."""
    if len(series) < period + 1:
        return np.nan
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = gain.iloc[-1] / last_loss
    return float(100 - 100 / (1 + rs))


def _macd_hist(series: pd.Series) -> tuple:
    """(histogram, histogram slope) for MACD(12,26,9) on the last bar.

    The SLOPE is the informative part on this timeframe — the signal-line
    crossover is a lagging restatement of what the histogram already showed.
    """
    if len(series) < 35:
        return np.nan, np.nan
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(hist.iloc[-1]), float(hist.iloc[-1] - hist.iloc[-2])


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR as a % of the last close."""
    if len(df) < period + 1:
        return np.nan
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    close = df["close"].iloc[-1]
    return float(atr / close * 100) if close > 0 else np.nan


def compute_symbol_features(df: pd.DataFrame, cfg: UniverseConfig) -> Optional[Dict]:
    """Daily/weekly feature vector for one symbol.

    `df` is that symbol's history, ascending by date, canonical columns.
    Returns None when there is too little history to characterise it — a
    symbol we cannot describe is one we should not trade.

    Everything here is a FEATURE, not a gate. Only the handful of fields the
    exclusion rules read (close, turnover, adr_pct) gate anything; the rest
    exists to be ranked on and, later, to be researched against realised R.
    """
    if len(df) < cfg.min_sessions:
        return None

    df = df.sort_values("date")
    close = df["close"]
    last = float(close.iloc[-1])
    if last <= 0:
        return None

    # -- range / volatility -------------------------------------------------
    rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100
    adr_pct = float(rng.tail(14).mean())
    atr_pct = _atr_pct(df)

    # -- liquidity ----------------------------------------------------------
    turnover = df["turnover"]
    median_turnover = float(turnover.tail(20).median())
    volume = df["volume"]
    median_volume = float(volume.tail(20).median())

    # INTEGRITY: turnover / volume is the day's VWAP, which MUST sit inside
    # that day's high-low range. This is arithmetic, not a market opinion —
    # no trading condition can violate it. A violation means the row is
    # corrupt: misaligned columns, a units change on NSE's side, or price
    # adjusted for a split/bonus while turnover was not.
    #
    # This matters more than it looks. median_turnover_20d is the DEFAULT
    # watchlist ranking key, so a symbol with inflated turnover does not just
    # slip through — it sorts to the TOP and takes a scarce watchlist slot.
    # And median_volume_20d is the denominator of every Tier 1 RVOL reading,
    # so a bad volume silently mis-scores that symbol all day.
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_vwap = df["turnover"] / df["volume"].replace(0, np.nan)
    #: Share of the last 20 sessions whose implied VWAP lands inside [low, high].
    #: 1.0 = fully consistent. Tolerance is 1% to absorb rounding in the
    #: published files rather than to forgive genuine misalignment.
    _in_range = ((implied_vwap >= df["low"] * 0.99)
                 & (implied_vwap <= df["high"] * 1.01))
    _checkable = implied_vwap.notna() & (df["volume"] > 0)
    _recent = _checkable.tail(20)
    vwap_consistency = (float(_in_range.tail(20)[_recent].mean())
                        if int(_recent.sum()) > 0 else np.nan)

    # -- structure ----------------------------------------------------------
    def donchian(n: int) -> tuple:
        """Channel levels, and today's close measured against the PRIOR bars.

        The level and the distance need DIFFERENT windows, which is the whole
        subtlety here.

        The published level includes today, because its consumer is Tier 1 on
        the NEXT session: tomorrow's live price against the high of the 20
        sessions up to and including today is exactly a breakout test.

        The distance must EXCLUDE today. Measuring today's close against a
        window containing today's own high is not a comparison — the high is
        greater than or equal to the close by construction, so the number is
        mathematically incapable of being positive and can never report a
        breakout. It read as an informative feature and was one: verified
        against 300 random tapes the maximum value was -0.03, and a stock
        closing 8% ABOVE its 20-day high reported -0.20, indistinguishable
        from an ordinary pullback. The low side had the mirror defect and
        could never report a breakdown.
        """
        window = df.tail(n)
        hi, lo = float(window["high"].max()), float(window["low"].min())
        prior = df.iloc[:-1].tail(n)
        if prior.empty:
            return hi, lo, np.nan, np.nan
        p_hi, p_lo = float(prior["high"].max()), float(prior["low"].min())
        # Signed, and now genuinely two-sided: positive = closed above the
        # prior high (a breakout, and by how much), negative = still below it.
        return hi, lo, (last - p_hi) / last * 100, (last - p_lo) / last * 100

    d5_hi, d5_lo, d5_hi_dist, d5_lo_dist = donchian(5)
    d20_hi, d20_lo, d20_hi_dist, d20_lo_dist = donchian(20)
    d55_hi, d55_lo, d55_hi_dist, d55_lo_dist = donchian(55)

    # NOT 52 weeks. This is the extreme over whatever history was loaded,
    # which at the default --history-days 120 is ~82 sessions — a third of a
    # year, not a year. Named for what it is; `history_sessions` below says
    # how much it actually covers. Same prior-bar exclusion as above.
    hi_hist = float(df["high"].max())
    lo_hist = float(df["low"].min())
    _prior_all = df.iloc[:-1]
    hi_hist_prior = float(_prior_all["high"].max()) if len(_prior_all) else hi_hist
    lo_hist_prior = float(_prior_all["low"].min()) if len(_prior_all) else lo_hist

    # -- moving averages ----------------------------------------------------
    def sma_dist(n: int) -> float:
        if len(close) < n:
            return np.nan
        return float((last - close.tail(n).mean()) / last * 100)

    # -- coil / contraction -------------------------------------------------
    # Bollinger width as a percentile of its own recent distribution:
    # contraction precedes expansion, and a breakout out of a genuine base is
    # a different animal from one out of chop.
    bb_width_pct = np.nan
    if len(close) >= 20:
        roll = close.rolling(20)
        width = (roll.std() * 4) / roll.mean() * 100
        recent = width.dropna()
        if len(recent) >= 20:
            bb_width_pct = float((recent <= recent.iloc[-1]).mean() * 100)

    # NR7: today's range is the narrowest of the last 7.
    nr7 = False
    if len(df) >= 7:
        last7 = (df["high"] - df["low"]).tail(7)
        nr7 = bool(last7.iloc[-1] <= last7.min())

    inside_bar = False
    if len(df) >= 2:
        inside_bar = bool(df["high"].iloc[-1] <= df["high"].iloc[-2]
                          and df["low"].iloc[-1] >= df["low"].iloc[-2])

    # -- returns ------------------------------------------------------------
    def ret(n: int) -> float:
        if len(close) <= n:
            return np.nan
        base = float(close.iloc[-(n + 1)])
        return float((last - base) / base * 100) if base > 0 else np.nan

    # -- delivery -----------------------------------------------------------
    deliv_pct = deliv_pct_20d = deliv_ratio = np.nan
    if "deliv_pct" in df.columns:
        dp = pd.to_numeric(df["deliv_pct"], errors="coerce").dropna()
        if len(dp) > 0:
            deliv_pct = float(dp.iloc[-1])
            if len(dp) >= 5:
                deliv_pct_20d = float(dp.tail(20).mean())
                if deliv_pct_20d > 0:
                    deliv_ratio = float(deliv_pct / deliv_pct_20d)

    macd_h, macd_slope = _macd_hist(close)

    return {
        "symbol": df["symbol"].iloc[-1],
        "date": df["date"].iloc[-1].date().isoformat(),
        "sessions": len(df),
        # price / liquidity
        "close": last,
        "prev_close": float(df["prev_close"].iloc[-1]) if pd.notna(df["prev_close"].iloc[-1]) else np.nan,
        "turnover": float(turnover.iloc[-1]),
        "median_turnover_20d": median_turnover,
        "volume": float(volume.iloc[-1]),
        "median_volume_20d": median_volume,
        "vwap_consistency": vwap_consistency,
        "tick_pct": NSE_TICK_SIZE / last * 100,
        # volatility / feasibility
        "adr_pct": adr_pct,
        "atr_pct": atr_pct,
        "bb_width_pctile": bb_width_pct,
        "nr7": nr7,
        "inside_bar": inside_bar,
        # structure
        "donchian_5_high": d5_hi, "donchian_5_low": d5_lo,
        "donchian_20_high": d20_hi, "donchian_20_low": d20_lo,
        "donchian_55_high": d55_hi, "donchian_55_low": d55_lo,
        "dist_5d_high_pct": d5_hi_dist, "dist_5d_low_pct": d5_lo_dist,
        "dist_20d_high_pct": d20_hi_dist, "dist_20d_low_pct": d20_lo_dist,
        "dist_55d_high_pct": d55_hi_dist, "dist_55d_low_pct": d55_lo_dist,
        # Renamed from *_52w, which it never was: the window is whatever
        # --history-days loaded (~82 sessions at the default 120), so calling
        # it a 52-week extreme overstated the lookback by roughly 3x.
        "history_sessions": len(df),
        "high_hist": hi_hist, "low_hist": lo_hist,
        "dist_hist_high_pct": (last - hi_hist_prior) / last * 100,
        "dist_hist_low_pct": (last - lo_hist_prior) / last * 100,
        # trend
        "dist_sma20_pct": sma_dist(20),
        "dist_sma50_pct": sma_dist(50),
        "dist_sma200_pct": sma_dist(200),
        "rsi_14d": _rsi(close),
        "macd_hist": macd_h,
        "macd_hist_slope": macd_slope,
        # momentum
        "ret_1d": ret(1), "ret_5d": ret(5), "ret_20d": ret(20),
        # participation
        "deliv_pct": deliv_pct,
        "deliv_pct_20d": deliv_pct_20d,
        "deliv_ratio": deliv_ratio,
    }


def add_cross_sectional_ranks(features: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank the fields that are only meaningful in comparison.

    This is the dividend of a wide universe. "RVOL > 2" is an absolute
    threshold that means different things for a Rs 300cr name and a Rs 20cr
    one; "top 5% of today's universe on volume anomaly" is the same statement
    everywhere. Ranks also make the composite score's sub-scores
    commensurable without hand-tuned normalisation constants.

    Relative strength here is measured against the UNIVERSE MEDIAN, not an
    index — no extra data source, and for a stock-picking screen the peer
    group is the more relevant benchmark anyway. Sector-relative RS is the
    better version and needs a sector map; deferred, not forgotten.
    """
    out = features.copy()
    rankable = ["ret_1d", "ret_5d", "ret_20d", "median_turnover_20d",
                "adr_pct", "atr_pct", "deliv_ratio", "macd_hist_slope",
                "dist_hist_high_pct", "dist_20d_high_pct"]
    for col in rankable:
        if col in out.columns:
            out[f"{col}_rank"] = out[col].rank(pct=True) * 100

    for horizon in ("ret_1d", "ret_5d", "ret_20d"):
        if horizon in out.columns:
            median = out[horizon].median()
            out[f"rs_{horizon}"] = out[horizon] - median
    return out


# ── Builder ──────────────────────────────────────────────────────────────────

@dataclass
class BuildOutcome:
    universe: pd.DataFrame
    funnel: List[ExclusionResult] = field(default_factory=list)
    sessions_loaded: int = 0
    warnings: List[str] = field(default_factory=list)

    def report(self) -> Dict:
        return {
            "generated_at": datetime.now().isoformat(),
            "sessions_loaded": self.sessions_loaded,
            "eligible_symbols": int(len(self.universe)),
            "funnel": [r.to_dict() for r in self.funnel],
            "warnings": self.warnings,
        }


class UniverseBuilder:
    """Assembles history, applies the hard exclusions, emits the feature table."""

    def __init__(self, source: UniverseSource, cfg: Optional[UniverseConfig] = None):
        self.source = source
        self.cfg = cfg or UniverseConfig()

    # -- history -----------------------------------------------------------

    def load_history(self, end: Optional[date] = None) -> pd.DataFrame:
        """Long-format OHLCV+delivery over the configured lookback.

        Walks backwards day by day. Weekends are skipped without a request;
        holidays surface as a missing file and are simply absent from the
        result — there is no holiday calendar to maintain.
        """
        end = end or date.today()
        frames: List[pd.DataFrame] = []
        loaded = 0
        for offset in range(self.cfg.history_days):
            day = end - timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            bhav = self.source.bhavcopy(day)
            if bhav is None or bhav.empty:
                continue
            deliv = self.source.delivery(day)
            if deliv is not None and not deliv.empty:
                bhav = bhav.merge(
                    deliv[["symbol", "series", "deliv_qty", "deliv_pct"]],
                    on=["symbol", "series"], how="left",
                )
            frames.append(bhav)
            loaded += 1

        if not frames:
            return pd.DataFrame()
        history = pd.concat(frames, ignore_index=True)
        logger.info(f"universe: loaded {loaded} sessions, "
                    f"{history['symbol'].nunique()} distinct symbols")
        return history.sort_values(["symbol", "date"])

    # -- exclusions --------------------------------------------------------

    def _exclude(self, df: pd.DataFrame, funnel: List[ExclusionResult],
                 name: str, keep_mask: pd.Series, detail: str = "") -> pd.DataFrame:
        before = len(df)
        kept = df[keep_mask]
        funnel.append(ExclusionResult(name, True, before - len(kept), len(kept), detail))
        return kept

    def _skip(self, df: pd.DataFrame, funnel: List[ExclusionResult],
              name: str, why: str) -> pd.DataFrame:
        """Record a rule that did NOT run.

        The distinction between "filtered clean" and "never filtered" has to
        survive into the report, or a consumer will read an unfiltered list
        as a safe one.
        """
        funnel.append(ExclusionResult(name, False, 0, len(df), why))
        return df

    def build(
        self,
        end: Optional[date] = None,
        surveillance_symbols: Optional[Iterable[str]] = None,
        fno_ban_symbols: Optional[Iterable[str]] = None,
        mis_allowed_symbols: Optional[Iterable[str]] = None,
        mis_denied_symbols: Optional[Iterable[str]] = None,
        circuit_bands: Optional[Dict[str, float]] = None,
        surveillance_incomplete: Optional[Sequence[str]] = None,
    ) -> BuildOutcome:
        """Run the whole Tier-0 job.

        The four list arguments are the exclusions that cannot be scraped
        reliably. Pass None and the rule is skipped AND recorded as skipped —
        never silently treated as passed.

        `surveillance_incomplete` names sources that were expected but did not
        contribute (e.g. ["ASM"]). A partial list still filters and is still
        applied, but the funnel and the warnings say so, because "applied"
        must never be readable as "fully checked".
        """
        warnings: List[str] = []
        history = self.load_history(end)
        if history.empty:
            return BuildOutcome(pd.DataFrame(), [], 0,
                                ["no sessions loaded — check the source"])

        sessions = int(history["date"].nunique())
        funnel: List[ExclusionResult] = [
            ExclusionResult("loaded", True, 0, int(history["symbol"].nunique()),
                            f"{sessions} sessions")
        ]

        # -- series: EQ only. BE/BZ are Trade-to-Trade (no intraday at all).
        eq = history[history["series"].isin(TRADEABLE_SERIES)]
        funnel.append(ExclusionResult(
            "series_not_tradeable", True,
            int(history["symbol"].nunique()) - int(eq["symbol"].nunique()),
            int(eq["symbol"].nunique()),
            f"kept {sorted(TRADEABLE_SERIES)}; BE/BZ are T2T, SM/ST are SME",
        ))
        if eq.empty:
            return BuildOutcome(pd.DataFrame(), funnel, sessions,
                                ["no EQ-series rows in history"])

        # -- features (also drops symbols with too little history)
        rows = []
        for symbol, grp in eq.groupby("symbol", sort=False):
            feat = compute_symbol_features(grp, self.cfg)
            if feat is not None:
                rows.append(feat)
        if not rows:
            return BuildOutcome(pd.DataFrame(), funnel, sessions,
                                [f"no symbol had >= {self.cfg.min_sessions} sessions"])
        feats = pd.DataFrame(rows)
        funnel.append(ExclusionResult(
            "insufficient_history", True,
            int(eq["symbol"].nunique()) - len(feats), len(feats),
            f"needs >= {self.cfg.min_sessions} sessions",
        ))

        # -- structural exclusions ------------------------------------------
        c = self.cfg
        feats = self._exclude(feats, funnel, "price_floor",
                              feats["close"] >= c.min_price,
                              f"close >= Rs {c.min_price:g}")
        feats = self._exclude(feats, funnel, "turnover_floor",
                              feats["median_turnover_20d"] >= c.min_turnover_rupees,
                              f"20d median turnover >= Rs {c.min_turnover_rupees / 1e7:g} cr")
        if c.min_vwap_consistency > 0:
            feats = self._exclude(
                feats, funnel, "turnover_not_self_consistent",
                feats["vwap_consistency"].fillna(0.0) >= c.min_vwap_consistency,
                f"turnover/volume lands in the day's range on >= "
                f"{c.min_vwap_consistency:.0%} of sessions")
        else:
            feats = self._skip(
                feats, funnel, "turnover_not_self_consistent",
                "CHECK DISABLED — turnover and volume are not verified against "
                "the day's price range, so a corrupt row can top the ranking")
        feats = self._exclude(feats, funnel, "adr_floor",
                              feats["adr_pct"] >= c.min_adr_pct,
                              f"ADR >= {c.min_adr_pct:g}% (room to pay the stop)")
        feats = self._exclude(feats, funnel, "tick_size_ratio",
                              feats["tick_pct"] <= c.max_tick_pct,
                              f"one tick <= {c.max_tick_pct:g}% of price")

        # -- list-based exclusions ------------------------------------------
        if surveillance_symbols is not None:
            banned = {str(s).strip().upper() for s in surveillance_symbols}
            _gap = list(surveillance_incomplete or [])
            _detail = f"{len(banned)} symbols on ASM/GSM"
            if _gap:
                _detail += (f" — PARTIAL: {'/'.join(_gap)} contributed nothing, "
                            f"so those names were NOT checked")
                warnings.append(
                    f"Surveillance rule only PARTIALLY applied — "
                    f"{'/'.join(_gap)} unavailable. Names on the missing "
                    f"list(s) are still in the universe; they carry 100% "
                    f"margin and are MIS-blocked at the broker.")
            feats = self._exclude(feats, funnel, "surveillance_asm_gsm",
                                  ~feats["symbol"].isin(banned), _detail)
        else:
            feats = self._skip(
                feats, funnel, "surveillance_asm_gsm",
                "NO LIST SUPPLIED — ASM/GSM names carry 100% margin and are "
                "MIS-blocked at the broker; they are NOT excluded here",
            )
            warnings.append("ASM/GSM list absent — universe is NOT surveillance-filtered")

        if fno_ban_symbols is not None:
            banned = {str(s).strip().upper() for s in fno_ban_symbols}
            feats = self._exclude(feats, funnel, "fno_ban",
                                  ~feats["symbol"].isin(banned),
                                  f"{len(banned)} symbols in F&O ban")
        else:
            feats = self._skip(feats, funnel, "fno_ban", "NO LIST SUPPLIED")

        if mis_denied_symbols is not None:
            # DENY-list: names only symbols asked AND refused, so a symbol
            # that was never asked is kept. The allow-list branch below
            # cannot do that — see NseHttpSource/KiteBrokerSource.mis_denied
            # for why that difference is a ratchet rather than a rounding
            # error.
            denied = {str(s).strip().upper() for s in mis_denied_symbols}
            feats = self._exclude(feats, funnel, "mis_not_allowed",
                                  ~feats["symbol"].isin(denied),
                                  f"{len(denied)} refused MIS by the broker "
                                  f"(unasked symbols kept)")
        elif mis_allowed_symbols is not None:
            # ALLOW-list: only for a list you supplied and know to be
            # COMPLETE. Anything absent is excluded.
            allowed = {str(s).strip().upper() for s in mis_allowed_symbols}
            feats = self._exclude(feats, funnel, "mis_not_allowed",
                                  feats["symbol"].isin(allowed),
                                  f"{len(allowed)} MIS-eligible at the broker "
                                  f"(supplied list treated as complete)")
        else:
            feats = self._skip(
                feats, funnel, "mis_not_allowed",
                "NO LIST SUPPLIED — some symbols may be CNC-only at the broker",
            )
            warnings.append("Broker MIS list absent — universe may contain CNC-only names")

        if circuit_bands is not None:
            bands = {str(k).strip().upper(): float(v) for k, v in circuit_bands.items()}
            band_series = feats["symbol"].map(bands)
            # Unknown band = keep. A missing entry is missing information, not
            # evidence of a tight band, and this rule must not silently empty
            # the universe when a band file is partial.
            keep = band_series.isna() | (band_series >= c.min_circuit_band_pct)
            feats = self._exclude(feats, funnel, "circuit_band",
                                  keep, f"band >= {c.min_circuit_band_pct:g}% "
                                        f"(unknown band kept)")
        else:
            feats = self._skip(
                feats, funnel, "circuit_band",
                "NO BAND FILE SUPPLIED — a 2%/5% band cannot contain an "
                "intraday R-multiple",
            )
            warnings.append("Circuit-band file absent — universe may contain 2%/5% band names")

        feats = add_cross_sectional_ranks(feats)
        feats = feats.sort_values("median_turnover_20d", ascending=False).reset_index(drop=True)
        # A feature that came out NULL for every surviving symbol was not
        # computed at all. Same principle as a skipped exclusion rule: it
        # occupies a column, it can be ranked on, and it will be researched
        # against realised R — while carrying nothing. The usual cause is a
        # lookback longer than the loaded history (a 200-day SMA needs 200
        # sessions; --history-days 120 yields about 82), which is invisible
        # in the output because the column is present and simply empty.
        if not feats.empty:
            dead = [c for c in feats.columns if feats[c].isna().all()]
            if dead:
                shown = ", ".join(dead[:6]) + (" ..." if len(dead) > 6 else "")
                warnings.append(
                    f"{len(dead)} feature(s) NULL for all {len(feats)} symbols "
                    f"— never computed: {shown}. A lookback longer than the "
                    f"{sessions} sessions loaded is the usual cause; raise "
                    f"--history-days.")

        return BuildOutcome(feats, funnel, sessions, warnings)


# ── Persistence ──────────────────────────────────────────────────────────────

DEFAULT_UNIVERSE_FILE = "universe.csv"
DEFAULT_REPORT_FILE = "universe_report.json"


def write_universe(outcome: BuildOutcome,
                   universe_path: str = DEFAULT_UNIVERSE_FILE,
                   report_path: str = DEFAULT_REPORT_FILE) -> None:
    """Write the feature table and the exclusion report.

    Both, always, together. The universe file alone is unfalsifiable — you
    cannot tell a 900-name list that was correctly filtered from a 900-name
    list where every surveillance rule silently no-opped.
    """
    Path(universe_path).parent.mkdir(parents=True, exist_ok=True)
    outcome.universe.to_csv(universe_path, index=False)
    Path(report_path).write_text(json.dumps(outcome.report(), indent=2, default=str))
    logger.info(f"universe: wrote {len(outcome.universe)} symbols -> {universe_path}")


def load_universe(path: str = DEFAULT_UNIVERSE_FILE) -> pd.DataFrame:
    """Read back the universe file. Empty frame when it does not exist yet."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"universe file {path} not found — returning empty")
        return pd.DataFrame()
    return pd.read_csv(p)


def format_funnel(outcome: BuildOutcome) -> str:
    """Human-readable exclusion funnel for the console / nightly log."""
    lines = [f"{'rule':<28} {'removed':>8} {'remaining':>10}  detail"]
    lines.append("-" * 96)
    for r in outcome.funnel:
        mark = " " if r.applied else "!"
        lines.append(f"{mark}{r.name:<27} {r.removed:>8} {r.remaining:>10}  {r.detail}")
    if outcome.warnings:
        lines.append("")
        lines.append("WARNINGS (rules marked ! above did not run):")
        lines.extend(f"  - {w}" for w in outcome.warnings)
    return "\n".join(lines)
