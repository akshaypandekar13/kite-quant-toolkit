"""scripts/depth_recorder.py: tick parsing, instrument selection, writer (no network)."""

import csv
import gzip
from datetime import date, datetime

import pandas as pd
import pytest

from scripts import depth_recorder as dr


def _full_tick(token=1):
    lv = lambda base, step: [dict(price=base + i * step, quantity=10 * (i + 1), orders=i + 1)
                            for i in range(5)]
    return dict(instrument_token=token, last_price=100.5, last_traded_quantity=3,
                average_traded_price=100.2, volume_traded=5000, total_buy_quantity=900,
                total_sell_quantity=800, oi=1200,
                last_trade_time=datetime.fromtimestamp(1_760_000_000),
                exchange_timestamp=datetime.fromtimestamp(1_760_000_001),
                depth=dict(buy=lv(100.45, -0.05), sell=lv(100.5, 0.05)))


def test_full_tick_row_layout_and_epochs():
    row = dr.tick_row(_full_tick(), 123)
    r = dict(zip(dr.COLS, row))
    assert len(row) == len(dr.COLS)
    assert r["exch_ts"] == 1_760_000_001 and r["ltt"] == 1_760_000_000   # local-zone round trip
    assert (r["b1p"], r["b1q"], r["b1o"]) == (100.45, 10, 1)
    assert r["b5p"] == pytest.approx(100.25) and r["a5p"] == pytest.approx(100.7)
    assert r["a1p"] == 100.5 and r["oi"] == 1200 and r["recv_ns"] == 123


def test_index_tick_has_blank_depth_not_zero():
    r = dict(zip(dr.COLS, dr.tick_row(dict(instrument_token=9, last_price=22000.0), 1)))
    assert r["b1p"] is None and r["a1q"] is None and r["exch_ts"] is None
    assert r["ltp"] == 22000.0


def _dumps():
    def inst(tok, ts, name, typ, seg, exp=None, strike=0.0, lot=1):
        return dict(instrument_token=tok, tradingsymbol=ts, name=name, instrument_type=typ,
                    segment=seg, expiry=exp, strike=strike, lot_size=lot, tick_size=0.05)
    nse = pd.DataFrame([inst(1, "AAA", "AAA", "EQ", "NSE"), inst(2, "BBB", "BBB", "EQ", "NSE"),
                        inst(3, "NIFTY 50", "NIFTY 50", "EQ", "INDICES"),
                        inst(4, "NIFTY BANK", "NIFTY BANK", "EQ", "INDICES")])
    d = lambda s: pd.Timestamp(s).date()
    nfo = [inst(10, "AAAOCTFUT", "AAA", "FUT", "NFO-FUT", d("2026-10-27"), lot=500),
           inst(11, "AAANOVFUT", "AAA", "FUT", "NFO-FUT", d("2026-11-24"), lot=500),
           inst(12, "AAASEPFUT", "AAA", "FUT", "NFO-FUT", d("2026-09-29"), lot=500),   # expired
           inst(20, "NIFTYOCTFUT", "NIFTY", "FUT", "NFO-FUT", d("2026-10-27"), lot=65)]
    t = 100
    for exp in ("2026-10-13", "2026-10-20", "2026-10-27", "2026-11-24"):
        for k in range(21000, 24050, 50):
            for typ in ("CE", "PE"):
                nfo.append(inst(t, f"N{exp}{k}{typ}", "NIFTY", typ, "NFO-OPT", d(exp), float(k), 65))
                t += 1
    return nse, pd.DataFrame(nfo)


def test_select_picks_cash_near_future_indices_and_option_band():
    nse, nfo = _dumps()
    inst = dr.select(nse, nfo, ["AAA", "BBB"], spot=22500.0, today=date(2026, 10, 12))
    aaa = inst[inst.symbol == "AAA"]
    assert set(aaa.kind) == {"cash", "fut"} and aaa[aaa.kind == "fut"].token.item() == 10
    assert "BBB" not in set(inst.symbol)                     # no future -> skipped
    assert set(inst[inst.kind == "index"].tradingsymbol) == {"NIFTY 50", "NIFTY BANK"}
    o = inst[inst.kind == "opt"]
    assert {str(e) for e in o.expiry.unique()} == {"2026-10-13", "2026-10-27"}   # nearest + monthly
    assert o.strike.min() >= 22500 * 0.97 and o.strike.max() <= 22500 * 1.03
    assert inst.token.is_unique


def test_writer_hourly_files_one_header_each(tmp_path):
    w = dr.TickWriter(tmp_path)
    w.start()
    ns = lambda h, m: int((datetime(2026, 10, 12, h, m) - datetime(1970, 1, 1)).total_seconds()
                          - 19800) * 10**9                   # IST wall clock -> epoch ns
    for h, m in ((9, 59), (9, 59), (10, 0)):
        w.put(dr.tick_row(_full_tick(), ns(h, m)))
    w.stop()
    w2 = dr.TickWriter(tmp_path)                              # restart mid-hour: append
    w2.start(); w2.put(dr.tick_row(_full_tick(), ns(10, 5))); w2.stop()
    read = lambda f: list(csv.reader(gzip.open(tmp_path / f, "rt")))
    nine, ten = read("ticks_09.csv.gz"), read("ticks_10.csv.gz")
    assert nine[0] == dr.COLS and len(nine) == 3
    assert ten[0] == dr.COLS and len(ten) == 3 and ten[1:][0][0] != dr.COLS[0]


def test_ist_hour_and_stop_time():
    assert dr.ist_hour(0) == 5                                # 1970-01-01 00:00 UTC = 05:30 IST
    assert dr.stop_time(datetime(2026, 10, 12, 9, 7)) == datetime(2026, 10, 12, 15, 31)


def test_freeze_ranks_by_recent_median_turnover(tmp_path):
    days = pd.bdate_range("2026-09-01", periods=30)
    rows = [dict(date=d, instrument="FUTSTK", symbol=s, value_lakh=v)
            for d in days for s, v in (("OLD", 100 if d < days[15] else 1), ("NEW", 50))]
    rows += [dict(date=d, instrument="FUTIDX", symbol="NIFTY", value_lakh=1e9) for d in days]
    p = tmp_path / "f.csv.gz"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert dr.freeze_universe(p, 1, tmp_path / "u.txt") == ["NEW"]
