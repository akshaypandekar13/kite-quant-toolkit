"""Parsers and fetch plumbing of scripts/fno_history.py (no network)."""

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from scripts import fno_history as fh

LEGACY = (
    "INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,"
    "CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
    "FUTSTK,RELIANCE,25-Jul-2024,0,XX,3150,3200,3140,3187.35,3187.35,"
    "54084,427958,30303750,882000,05-JUL-2024,\n"
    "OPTSTK,RELIANCE,25-Jul-2024,3200,CE,50,60,45,55,55,100,800,250000,2500,05-JUL-2024,\n"
)

UDIFF = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2024-07-08,2024-07-08,FO,NSE,STF,1,,RELIANCE,,2024-07-25,2024-07-25,,,"
    "RELIANCE24JULFUT,3190,3215,3185,3208.10,3208,3187.35,3201.80,3208.10,28804250,"
    "-1499500,39175,31353890000,1,F1,250,,,,,\n"
    "2024-07-08,2024-07-08,FO,NSE,STO,2,,RELIANCE,,2024-07-25,2024-07-25,3200.00,PE,"
    "RELIANCE24JUL3200PE,40,45,38,41,41,42,3201.80,41,0,0,0,0,0,F1,250,,,,,\n"
)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", text)
    return buf.getvalue()


def test_legacy_and_udiff_share_one_schema_and_units():
    leg = fh.parse_fo(_zip(LEGACY), date(2024, 7, 5))
    new = fh.parse_fo(_zip(UDIFF), date(2024, 7, 8))
    assert list(leg.columns) == list(new.columns) == fh.COLS
    lf = leg[leg.instrument == "FUTSTK"].iloc[0]
    nf = new[new.instrument == "FUTSTK"].iloc[0]
    # OI in shares in both; contracts in contracts in both.
    assert lf.oi == 30303750 and nf.oi == 28804250
    assert nf.contracts == 39175
    # contracts x lot x price reproduces the traded value
    assert abs(nf.contracts * nf.lot * nf.close / 1e5 - nf.value_lakh) / nf.value_lakh < 0.01
    assert nf.underlying == 3201.80 and pd.isna(lf.underlying) and pd.isna(lf.lot)
    assert lf.opt_type == "" and nf.opt_type == ""
    assert new.iloc[1].instrument == "OPTSTK" and new.iloc[1].opt_type == "PE"
    assert leg.expiry.iloc[0] == pd.Timestamp("2024-07-25") == new.expiry.iloc[0]


def test_url_switches_format_on_udiff_date():
    assert "DERIVATIVES/2024/JUL/fo05JUL2024bhav" in fh.url_for("fo", date(2024, 7, 5))
    assert "BhavCopy_NSE_FO_0_0_0_20240708" in fh.url_for("fo", date(2024, 7, 8))
    assert fh.url_for("mto", date(2019, 1, 2)).endswith("MTO_02012019.DAT")


def test_mto_keeps_only_type_20_rows():
    blob = (b"Security Wise Delivery Position - Compulsory Rolling Settlement\n"
            b"10,MTO,02012019,371928763,0001716\n"
            b"Record Type,Sr No,Name of Security,Quantity Traded,Deliverable,% \n"
            b"20,1,20MICRONS,EQ,164131,66318,40.41\n"
            b"20,2,RELIANCE,EQ,1000,425,42.50\n")
    df = fh.parse_mto(blob, date(2019, 1, 2))
    assert list(df.symbol) == ["20MICRONS", "RELIANCE"]
    assert df.loc[1, "deliv_pct"] == 42.5 and df.loc[1, "traded_qty"] == 1000


def test_participant_file():
    blob = (b'"Participant wise Open Interest as on January 02, 2019",,,\n'
            b"Client Type,Future Index Long,Future Index Short\t,Total Long Contracts\n"
            b"Client,136626,188248,2543212\n"
            b"FII,10,20,30\n"
            b"TOTAL,136636,188268,2543242\n")
    df = fh.parse_participant(blob, date(2019, 1, 2))
    assert list(df.client_type) == ["Client", "FII", "TOTAL"]
    assert df.loc[0, "future_index_long"] == 136626
    assert "future_index_short" in df.columns


class FakeSource:
    def __init__(self, cache, holidays):
        self.cache, self.holidays, self.calls = Path(cache), holidays, []

    def _fetch(self, url, name, day):
        self.calls.append(name)
        if day in self.holidays:
            return None
        (self.cache / name).write_bytes(b"x")
        return b"x"


def test_download_skips_other_kinds_on_holiday_and_is_resumable(tmp_path):
    hol = date(2024, 7, 9)
    src = FakeSource(tmp_path, {hol})
    s = fh.download(date(2024, 7, 8), date(2024, 7, 10), source=src)
    assert [c for c in src.calls if "20240709" in c] == ["fo_20240709.csv.zip"]
    assert s["fo"] == {"have": 0, "got": 2, "none": 1}
    src.calls.clear()
    s = fh.download(date(2024, 7, 8), date(2024, 7, 10), source=src)
    assert src.calls == ["fo_20240709.csv.zip"]          # only the holiday is retried
    assert s["mto"]["have"] == 2


def test_calendar_adds_weekend_sessions_and_fetches_past_a_missing_fo(tmp_path):
    sat, gap = date(2024, 1, 20), date(2024, 1, 23)   # special Saturday; fo missing
    cal = [date(2024, 1, 19), sat, date(2024, 1, 22), gap]
    src = FakeSource(tmp_path, set())
    src._fetch = lambda url, name, day, f=src._fetch: (
        None if name.startswith("fo_") and day == gap else f(url, name, day))
    s = fh.download(date(2024, 1, 19), date(2024, 1, 25), source=src, calendar=cal)
    assert (tmp_path / "fo_20240120.csv.zip").exists()        # Saturday session
    assert (tmp_path / "mto_20240123.DAT").exists()           # fo gap, mto kept
    assert (tmp_path / "fo_20240124.csv.zip").exists()        # past calendar end
    assert s["fo"]["none"] == 1
