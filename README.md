# Kite Quant Toolkit

Trading infrastructure for Zerodha Kite Connect and NSE: data pipelines,
live market recorders, cost models and a strict backtesting method.

This repository is a portfolio. It shows tools taken from a larger private
trading system I built and ran on a live Zerodha account. All the code here
runs on its own, and **174 automated tests** check it.

> Not investment advice. This repository does not sell or recommend any
> strategy, signal or return. See the [Disclaimer](#disclaimer).

---

## What I can build for you

| Service | What you get |
|---|---|
| **Kite Connect automation** | Login and daily token refresh, live WebSocket ticks, order placement with safety checks, Telegram alerts, deployment on AWS or a VPS |
| **Data pipelines** | Historical candles from Kite, NSE F&O bhavcopy, participant OI, delivery data, corporate results dates, all stored and checked |
| **Backtest audit** | I test your strategy rules with real Zerodha charges, a holdout period and a check against random entries, and you get an honest written report |
| **Options analytics** | Implied volatility, skew, put-call parity and OI build-up computed from NSE data |
| **Live market recording** | Tick and 5-level depth capture for research, with latency measurement |

---

## What is in this repository

| Area | Files | What it does |
|---|---|---|
| Kite access | `scripts/kite_token.py`, `src/kite_env.py` | Daily access-token login flow; `.env` loading |
| Historical data | `scripts/export_kite_history.py`, `scripts/multi_asset_history.py` | Bulk candle export (minute to day) for stocks, indices, ETFs and MCX, with resume and integrity checks |
| NSE F&O data | `scripts/fno_history.py` | NSE F&O bhavcopy, delivery and participant-wise OI, 2019 onward |
| Options features | `scripts/fno_features.py` | Black-76 implied volatility per expiry, 30-day IV and skew, OI build-up across rollover, corporate actions found from option re-strikes |
| Earnings dates | `scripts/earnings_dates.py` | Point-in-time result timestamps from NSE (2007 onward) |
| Live depth recorder | `scripts/depth_recorder.py` | Records ticks with 5-level depth for about 300 instruments; never places an order |
| Order latency probe | `scripts/latency_probe.py` | Measures order and cancel round trips with orders that cannot fill; several tested safety rails |
| Costs | `src/costs.py`, `scripts/_vehicles.py` | Zerodha charges (brokerage, STT, exchange, GST, SEBI, stamp) per instrument |
| Statistics | `scripts/_stats.py`, `scripts/_spread.py` | Session-clustered t-statistics; bid-ask spread estimated from OHLC bars |
| Market-making simulator | `scripts/mm_sim.py` | Queue position, latency, cancel races, adverse selection, per-fill charges |
| Strategy tests | `scripts/fno_test.py`, `scripts/condor_test.py`, `scripts/multi_asset_test.py`, `scripts/calendar_test.py` | Pre-registered tests (rules fixed before any result is seen) |
| Calendars | `src/expiry.py`, `src/clock.py` | NSE expiry calendar, including moved expiries; IST clock helpers |
| Write-ups | `docs/` | Pre-registrations and results |

---

## Selected results

All of these were measured on real data.

**The data matches outside references**
- NIFTY 30-day implied volatility computed from option prices vs India VIX:
  correlation **0.98**.
- The forward implied by put-call parity vs the actual futures price: within
  **5.5 to 8.9 basis points**.
- Earnings dataset: **57,770 result announcements**, 1,256 stocks, with NSE's
  own broadcast timestamps.

**What a live Zerodha feed actually delivers** (depth recorder, AWS Mumbai)
- About **16,600 ticks a minute** across 312 instruments, and the writer kept
  up with no backlog.
- Each instrument gets about **one snapshot a second** (median gap 0.999 s).
- Ticks arrive about **0.24 s after** the exchange timestamp.
- What this means: a retail Kite feed cannot see sub-second order-book
  effects. This is worth knowing before anyone pays to build such a strategy.

**Strategy research, reported honestly**
I tested strategies with the rules written down first, real costs, a holdout
period and no re-fitting after a failure. Most of them **failed**, and the
write-ups say so:

| Test | Result |
|---|---|
| F&O long-short: skew, parity deviation, pairs (68 monthly cycles) | Fail: net −57, −67 and −1 bps a cycle after costs |
| NIFTY monthly iron condor (80 cycles) | Fail: 70% win rate but negative expectancy; max drawdown 146% of a 2R account |
| Multi-asset allocation: GTAA, dual momentum, ML ranker, risk parity | No rule beat a simple equal-weight mix |
| Calendar effects, earnings drift, news sentiment | The effects exist, but costs or the opening auction absorb them |

The point of this table: a test method that rejects bad ideas before they are
traded is what protects your capital. If your strategy is good, the same
method will show that too.

---

## The private trading system (shown on request)

The live intraday engine is not in this repository. It includes:
- KiteTicker WebSocket streaming, candle building, a watchdog and
  self-restart when the feed stalls;
- a 3-strategy consensus (candlestick patterns, trend, confluence) and a
  market-regime detector;
- position sizing by risk, partial exit at 1R, phased stop-loss and time stops;
- a gate audit: every rejected signal is logged and re-scored later against
  real prices;
- Telegram alerts, an event-driven replay backtester and full journals.

I can show a demo or walk through it on a call.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your own Kite Connect keys
python -m pytest -q             # 174 tests, no API keys or network needed

python -m scripts.kite_token    # daily login
python -m scripts.export_kite_history --help
python -m scripts.fno_history --help
python -m scripts.depth_recorder --help
```

You need your own Kite Connect subscription. No market data ships with this
repository; each script downloads it from the source with your credentials.

---

## Disclaimer

- This is software engineering and research work. It is **not investment
  advice** and makes **no claim of profit**. The results above include
  strategies that lost money in testing.
- Market data belongs to NSE, MCX and Zerodha under their terms. This
  repository contains no data, and you are responsible for following those
  terms.
- Anyone running automated orders must follow SEBI's and their broker's rules
  for algorithmic trading.

## Contact

Available for freelance work:

- **Fiverr:** https://www.fiverr.com/s/7jj9YBL
- **Telegram:** https://t.me/kite_zerodha_automation

© 2026 Akshay Pandekar. All rights reserved. See [LICENSE](LICENSE).
