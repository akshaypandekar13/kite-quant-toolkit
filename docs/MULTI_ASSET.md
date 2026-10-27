# Multi-asset research: pre-registration (frozen 2026-10-11, before any strategy return was computed)

Client's brief: find ways to make money across F&O,
equity, gold, forex and crypto, using every algorithm we can build,
including ML and LLMs, tested on history.

**What was already ruled out** (earlier project notes, docs/FNO*.md,
docs/CONDOR_PREREG.md): every intraday rule family; daily candlesticks;
breakouts; swing filter stacks; F&O long-short (skew, parity, pairs); and
the NIFTY iron condor. The near-miss is R5 (stock-level ML momentum, fresh t
2.52, fails one year criterion). This round tests things never tested here:
**allocation ACROSS asset classes and sectors**, with low turnover, so costs
stop deciding the answer.

## 1. Scope decisions, made before any return was computed

| asset class | decision | why |
|---|---|---|
| Indian equity (large / mid / small) | IN | NIFTYBEES 2008+, NIFTY MIDCAP 100 / SMLCAP 100 indices 2008+ |
| Gold, silver | IN | MCX continuous futures 2010+ spliced into GOLDBEES (2015+) / SILVERBEES (2022+). Check: 2017–26 CAGR MCX gold 18.7% vs GOLDBEES 17.3%; silver 30.1% vs 28.2% (2022–26). The ETF is used wherever it exists |
| US equity | IN | MON100 (Nasdaq 100 in INR) 2015+ |
| Indian gilts | IN | LTGILTBEES 2016-07+ (growth ETF, 5.9% CAGR 2017–26) |
| Indian sectors | IN (sector test only) | 12 NSE sector indices 2008+ |
| Crude, natural gas, base metals | OUT | Kite's continuous series is unadjusted; roll days cannot be located reliably (OI-jump detection finds 3–6 rolls a year where 6–12 occur); crude printed ₹1 in April 2020; lot sizes |
| Forex (USDINR etc.) | OUT | Kite serves only 2021-10 → 2023-06 for the currency futures; too short |
| Crypto | OUT | Not on Kite, and India's tax (30% flat on gains, no loss set-off, 1% TDS on every sale) charges any active strategy ~2% a round trip before a price moves |
| LLM sentiment | NOT BACKTESTABLE | No point-in-time news archive here, and an LLM asked about a past date already knows what happened next (look-ahead built into the model). Any LLM backtest would be contaminated. It can only be tested forward |

**Total return caveats.** NIFTYBEES carries dividends (12.3% vs index
11.0% CAGR 2017–26); the midcap/smallcap/sector INDICES are price-only, so
they are understated by ~1%/yr. This applies equally to strategy and
benchmark. LIQUIDBEES pays interest as units, so its price is flat: **cash
earns an ASSUMED 6.0% a year** (sensitivity: 4.0%).

## 2. Mechanics (all strategies)

- Decision at the close of the last session of each month, from data up to
  that close. Trade at the NEXT session's OPEN. Hold to the next decision.
- An asset is eligible once it has 12 months of history.
- Long only, delivery, no leverage.
- **Costs** per side, on traded value: 0.10% spread + charges for liquid
  ETFs (NIFTYBEES, GOLDBEES, SILVERBEES, MON100, LTGILTBEES), 0.20% for
  index/sector proxies; stamp 0.015% on buys; exchange 0.00297%; STT 0.001%
  on equity-ETF sells. Plus 0.40%/yr expense drag on index/sector proxies
  (no fee is embedded in an index level).
- **Tax** reported separately, not in the pass bar: an estimate with 20% on
  gains realised within 12 months, 12.5% after.

## 3. Strategies (frozen, zero variant budget)

| id | rule | source |
|---|---|---|
| **S1 GTAA** | Assets: large, mid, small, gold, silver, US, gilts. Equal slots; an asset whose month-end close is above its 10-month SMA gets its slot, otherwise the slot goes to cash | Faber 2007 |
| **S2 DUAL** | Same assets except gilts. Rank by 12-month return; hold the top 2, each only if its 12-month return beats cash; a failing slot goes to gilts (cash before gilts exist) | Antonacci 2014 |
| **S3 SECTOR** | 12 sector indices. Hold the top 3 by 6-month return, equal weight | Moskowitz & Grinblatt 1999 |
| **S5 ML** | All S1 assets + 12 sectors (19). Features per instrument at each month-end: return over 1/3/6/12 months, 3-month daily vol, distance from 10-month SMA, 12-month max drawdown; common: India VIX level and 1-month change. Target: next month's return minus cash. GradientBoostingRegressor(depth 2, 200 trees, learning rate 0.05, subsample 0.8, random_state 0), trained each month on all earlier months whose label had ended; starts once 36 months of labels exist. Hold the top 3 predictions above 0, equal weight; no qualifier means cash | the ML family |

**Benchmarks:** B1 NIFTYBEES buy and hold. B2 equal weight of S1's
eligible assets, rebalanced monthly (the "1/N" portfolio: does timing beat
plain diversification?). S3's benchmark is the NIFTY 50 index (price, like
its sectors).

## 4. Periods

- **Development:** decisions 2011-10 → 2021-12.
- **Holdout:** 2022-01 → 2026-09. All four are scored on it once, at the
  same time as development, because nothing is chosen between them. Any
  later change is a new pre-registration.

## 5. Pass bar (per strategy, all of)

- **P1 edge:** monthly net return minus its benchmark (B2; NIFTY 50 for S3),
  development + holdout, t ≥ 2.5 (Bonferroni over 4 at 5%).
- **P2 holdout:** mean monthly excess on the holdout > 0.
- **P3 risk:** maximum drawdown no worse than the benchmark's.

**Labels:** a strategy that fails P1 but passes P3 with a higher Sharpe is
reported as a **risk reducer, not an edge**: a way to hold the same assets
with smaller losses, which can still be worth using. "Win rate" is not a
criterion.

**On a pass:** a forward paper record, not money.

---

## Amendment A1 (data gaps; found on the first run, before any result was read as a verdict)

Kite's index history has holes: NIFTY IT on 2013-05-03, NIFTY INFRA from
2018-12-10 to 2019-02-20 (52 sessions), LTGILTBEES 2016-09 → 2017-02
(before it was eligible). The first run carried a close forward for only 5
sessions, so a sector held through the IT hole turned S3's equity into NaN
from 2013-05. Every gap now carries the last close (a holding earns nothing
through the gap and the whole move on return). A test pins it and fails on
the old code. S3's first-run number (1.6% CAGR to 2013-04) was the bug. S5's
features changed too: 11.0% → 9.0% CAGR. No other number moved by more than
0.0004.

## RESULT (2026-10-11): 0 of 4 PASS. Holdout read once, as frozen.

Cash 6.0%. Monthly, net of costs. Development = holding months to 2022-01,
holdout = 2022-02 → 2026-10-09.

| | CAGR | Sharpe | max DD | dev CAGR | holdout CAGR | turnover/yr | after-tax CAGR (est.) | excess/month vs bench | t | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 NIFTYBEES hold | 11.8% | 0.42 | −28.8% | 14.1% | 6.9% | 0.07 | 10.4% | | | | | |
| **B2 equal weight, 7 assets** | **13.8%** | **0.64** | **−17.6%** | 11.8% | 18.3% | 0.55 | 12.6% | | | | | |
| B NIFTY 50 index (S3's bench) | 10.4% | 0.34 | −29.4% | 13.0% | 5.3% | | 9.1% | | | | | |
| S1 GTAA | 10.3% | 0.50 | **−10.8%** | 8.0% | 15.3% | 2.3 | 8.8% | −0.30% | −1.74 | ✗ | ✗ | ✓ |
| S2 DUAL | 12.6% | 0.41 | −25.2% | 8.2% | 22.7% | 5.3 | 10.8% | −0.01% | −0.03 | ✗ | ✓ | ✗ |
| S3 SECTOR | 10.0% | 0.29 | −48.7% | 9.0% | 12.4% | 7.0 | 7.3% | +0.05% | 0.18 | ✗ | ✓ | ✗ |
| S5 ML (GBM) | 9.0% | 0.26 | −20.1% | 8.2% | 10.7% | 13.6 | 6.8% | −0.30% | −1.04 | ✗ | ✗ | ✗ |

**Cash at 4%** changes no verdict (S1 9.7%, S2 12.3%, S3 10.1%, S5 8.2%).

**What it says:**
- **No timing or selection rule beat holding the assets in equal weight.**
  Trend (S1), dual momentum (S2), sector momentum (S3) and the ML ranker (S5)
  all earn less than or about the same as B2, and trade 4–25× as much.
- **S1 is a risk reducer, not an edge**: drawdown −10.8% vs −17.6%, at
  3.5 points a year less return. It fails P1 by the pre-registered rule.
- **The ML ranker is the worst of the four** after tax. Nineteen
  instruments a month, ~180 months: there is not enough data for a model to
  learn more than momentum does, and it trades 14× a year.
- **The one result worth keeping is the benchmark.** B2 beats NIFTY on
  return (13.8% vs 11.8%), Sharpe (0.64 vs 0.42) and drawdown (−17.6% vs
  −28.8%). It is diversification, not an edge: monthly correlations
  2016–26 are equity vs gold −0.01, equity vs gilts 0.02, equity vs US 0.26.
- **The caveat on B2:** its holdout (18.3% a year) rides gold and silver's
  run (B2 +30% in 2025 while NIFTY made +12%). The assets were chosen today,
  knowing gold boomed; equal weight is a published rule (DeMiguel, Garlappi
  & Uppal 2009), but this period's size is not a forecast.

**Holdout opened once. No paper record of S1–S5** (none passed).

---

# Round 2: pre-registration (frozen 2026-10-11, after round 1's result, before any round-2 return was computed)

Round 1 found no timing edge; diversification itself (B2) was the best
book. Round 2 asks whether B2 can be improved by **portfolio construction**
(published, parameter-free) or by putting **the repo's one near-edge, R5**,
in charge of the equity part. Mechanics, costs and cash as round 1.

| id | rule | source | benchmark | months |
|---|---|---|---|---|
| **C1 RISK PARITY** | B2's eligible assets, weighted by 1 / (60-session daily-return vol), monthly | Asness, Frazzini & Pedersen 2012 (inverse-vol form) | B2 | 2011-11 → |
| **C2 VOL TARGET** | B2's weights scaled by min(1, 10% / the 60-session realised vol of B2's current weights); the rest in cash. No leverage | Moreira & Muir 2017 | B2 | 2011-11 → |
| **C3 R5 SLEEVE** | B2 with LARGE, MID and SMALL replaced by one asset R5S held at 3/7 weight (until gilts are eligible the denominator follows B2's eligible count) | this repo's R5 | B2 | 2021-11 → |

**R5S (the sleeve), survivorship-neutralised:** daily return =
NIFTY MIDCAP 100 return + (R5 harness return − EW268 return), where
- R5 = `structural/harness/R5_unseen_daily.csv` (R5 trained only on the 143,
  run on the 268 unseen stocks; its charges and slippage are inside);
- EW268 = equal-weighted daily mean return of the 268 stocks in
  `data/history_fresh` with ≥ 250 sessions of history, each day's return
  clipped to ±40%.

So the sleeve keeps R5's SELECTION over its own universe and drops the
universe's level, which is survivorship. R5S is traded at the liquid-ETF
cost. **Caveat stated in advance:** 2021–24 overlaps R5's development years
(its features were chosen on the 143 over that time); only the stocks are
unseen.

**Pass bar (each, all of):** P1 monthly excess over B2, t ≥ 2.4 (Bonferroni
over 3); P2 holdout (holding months from 2022-02) excess > 0; P3 max
drawdown no worse than B2's over the same months. C1/C2 that fail P1 but
pass P3 with a higher Sharpe are labelled risk reducers. Zero variant budget.

## Round 2 RESULT (2026-10-11): 0 of 3 PASS

| | months | CAGR | Sharpe | max DD | B2 same months: CAGR / Sharpe / DD | turnover/yr | excess/month | t | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 risk parity | 179 | 12.0% | 0.58 | −18.1% | 13.8% / 0.64 / −17.6% | 1.36 | −0.16% | −1.77 | ✗ | ✗ | ✗ |
| C2 vol target 10% | 179 | 11.9% | 0.58 | −17.6% | 13.8% / 0.64 / −17.6% | 0.93 | −0.16% | −1.97 | ✗ | ✗ | ✓ |
| C3 R5 sleeve | 59 | **24.3%** | **1.13** | −14.3% | 17.4% / 0.93 / −11.0% | 0.71 | **+0.52%** | 1.81 | ✗ | ✓ | ✗ |

- **Risk parity and vol targeting make B2 worse.** Inverse-vol weighting
  overweights gilts (low vol, 6% return); the vol target sits in cash after
  sell-offs and misses the rebounds. Neither is even a risk reducer here.
- **C3 is the closest anything in this repo has come to a pass**, and it
  still fails: t 1.81 against 2.4, and a deeper drawdown than B2.
- **What C3 rests on, R5's excess over its own universe (EW268), by year:**
  2021 (Q4) +9.3, 2022 −1.9, 2023 +20.5, 2024 +24.4, 2025 −0.8, 2026 (to
  Oct) +21.3 points. Monthly mean +1.1%, **t 1.66 over 60 months**, 4 of 6
  years positive. The same picture as the 2026-09-29 fresh test (t 2.52 on
  the staggered measure, failing the year criterion): **a selection effect
  that is probably real and not proven.**
- **Survivorship is visible in the benchmark itself:** EW268 made +63.6% in
  2023 against the midcap index's +46.6%. C3 removes the universe's level;
  R5's excess over it may still carry some of the same bias.

**Combined rounds 1–2: 0 of 7.** No asset-timing, sector, ML-ranking or
portfolio-construction rule beat holding equity, gold, silver, US equity
and gilts in equal weight. Nothing goes to a paper record from this round;
R5's existing forward record (`paper_r5.py`) is the test that can settle C3.

# Round 3: pre-registration (frozen 2026-10-12, before any round-3 return was computed)

Client chose all three follow-ups from round 2's report: calendar effects,
post-earnings drift, and an LLM news test. Four hypotheses, so **t ≥ 2.5**
(one-sided Bonferroni over 4) wherever a t is a criterion. Zero variant
budget. Costs are charged on every trade; uncertain spreads are ASSUMED and
labelled.

## D1 Calendar effects on NIFTY (`scripts/calendar_test.py`)

| id | trade | source |
|---|---|---|
| **D1a TOM** | buy NIFTYBEES at the close of each month's second-last session; sell at the close of the next month's third session (days −1..+3) | Lakonishok & Smidt 1988; McConnell & Xu 2008 (incl. India) |
| **D1b PRE-HOLIDAY** | buy at the close of the session before a pre-holiday session; sell at the pre-holiday session's close | Ariel 1990 |

- **Holiday:** a weekday inside the NIFTY 50 calendar's span with no session.
  A muhurat bar recorded on a holiday hides that holiday (disclosed, not fixed).
- **Costs per side:** ASSUMED 10 bps spread, exchange charges, stamp on buys,
  ETF STT on sells, DP ₹18.10 per sell at a ₹2.5L book. Cash 6% when flat.
- **Data:** NIFTYBEES and NIFTY 50 closes from 2008-01 to 2026-09-30.
  The effect itself is also reported on index daily returns (window days vs
  other days), not as a criterion.
- **Pass (each, all of):** P1 per-trade excess over cash for the held days,
  t ≥ 2.5; P2 holdout (trades entered from 2019-01-01) mean excess > 0;
  P3 excess positive in ≥ 60% of calendar years.

## D2 Post-earnings-announcement drift (`scripts/earnings_dates.py`, `scripts/earnings_drift.py`)

Source: Brandt, Kishore, Santa-Clara & Venkatachalam 2008 (drift after the
announcement RETURN); Chan, Jegadeesh & Lakonishok 1996. No analyst
estimates are available here, so the surprise is the market's own reaction.

- **Dates:** NSE's own dissemination timestamps, the earliest over the legacy
  results endpoint (2007 → Jan 2025) and the integrated-filing endpoint
  (2025 →), standalone or consolidated, original filings only. Downloaded for
  the 1,270 stocks in `data/history_8y` whose 60-session median traded value
  ever reached ₹10 cr; the filter below is applied point in time.
- **Event day d0:** the first session whose 15:30 close is at or after the
  timestamp.
- **Eligible at d0−2:** ≥ 250 sessions of history, 60-session median traded
  value ≥ ₹10 cr, price ≥ ₹20.
- **EAR:** return close(d0−2) → close(d0+1) minus the equal-weight eligible
  universe (EW) over the same window.
- **Quintiles:** breakpoints from events whose signal day (d0+1) lies in the
  previous 250 sessions; at least 200 such events, else unclassified.
- **D2 (the test):** buy every top-quintile event at open(d0+2), sell at
  close(d0+61), equal weight across open positions, cash when none.
  Benchmark EW (costless). Daily returns clipped ±40%.
- **D2T (the ₹2.5L implementation, reported, not a criterion):** 20 equal
  sub-accounts; when more events arrive than free slots, the highest EAR first.
- **Costs per side:** delivery STT 0.1%, stamp on buys, exchange + SEBI +
  GST, ASSUMED 10 bps spread; D2T adds DP ₹18.10 per sell on its slot value.
- **Pass (D2, all of):** P1 monthly excess over EW t ≥ 2.5; P2 holdout
  (months from 2024-01) mean excess > 0; P3 ≥ 5 full calendar years with
  positive excess. On a pass, D2T must also beat EW net before a paper record.
- **Caveat stated in advance:** the store holds today's listings only, so EW
  is survivorship-inflated. The test is selection WITHIN that universe; the
  top-minus-bottom event CAR is reported as the cleaner scientific measure.

## D3 LLM announcement sentiment, post-cutoff pilot (`scripts/llm_news.py`)

An LLM cannot be backtested on news it may have read in training. The
scoring model (Claude Opus 5.5, this session; training cutoff June 2026)
has never seen anything from **2026-07-01 → 2026-09-30**, so that window is
an honest, if short, test. (It ends 30 Sep because the price store,
`data/history_8y`, ends 1 Oct.) Design follows Lopez-Lira & Tang 2023.

- **Universe:** the 100 names in `depth_universe.txt`.
- **Items:** NSE corporate announcements broadcast after 15:30 or before
  09:15 (so the next open is the first chance to trade), excluding the
  routine categories frozen in `llm_news.ROUTINE`.
- **Scoring, blind:** the scorer sees symbol, company, category and NSE's
  announcement text only; never a date, a price or a return. Prompt frozen
  in `llm_news.PROMPT`: GOOD / BAD / UNKNOWN for the stock's price over the
  next trading day.
- **Trade:** the next session after the broadcast, open → close; long GOOD,
  short BAD (intraday). Return net of NIFTY 50's open → close, then net of
  ASSUMED 9 bps a round trip (MIS charges 5.6 + spread 3).
- **Pass (pilot):** P1 mean (GOOD − BAD) net excess per trade, t ≥ 2.5 over
  the pooled long and short trades with signs applied; P2 both legs net
  positive. A pass leads to a forward paper record (needs an API key; client
  action), never to money directly. The pilot's small n is disclosed.

## D1 RESULT (2026-10-12): 0 of 2 PASS

NIFTYBEES, 2008-01 → 2026-09; per-trade excess over cash for the held days,
net of costs.

| | trades | net/trade | excess/trade | t | dev / holdout excess | years + | CAGR | max DD | in market | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1a TOM | 224 | +30.2 bps | +21.0 bps | 1.31 | +20.0 / +22.4 | 10 of 19 | 8.2% | −17.9% | 19% | ✗ | ✓ | ✗ |
| D1b PRE-HOLIDAY | 254 | −6.9 bps | −9.2 bps | −1.25 | −5.1 / −14.8 | 4 of 19 | 4.5% | −12.5% | 5% | ✗ | ✗ | ✗ |

NIFTYBEES buy-and-hold over the same span: 7.9% CAGR, max DD −59.7%
(the span starts at the January 2008 peak).

- **The turn-of-the-month effect is visible in the index:** NIFTY 50
  averages +13.1 bps on window days against +1.4 bps on all other days
  (t 2.45). The per-trade excess is about the same before and after 2019
  (+20.0 and +22.4 bps), so it has not decayed.
- **It is too noisy to trade alone.** About 22 bps of round-trip cost takes
  40% of the gross, which leaves t 1.31, and only 10 of 19 years are
  positive. As a book it matched buy-and-hold's CAGR while in the market 19%
  of the time, with a −18% drawdown instead of −60%. That is a description of
  this sample, not a pass.
- **Pre-holiday:** +17 bps a day in the index (t 1.81), but it doesn't cover
  the cost of a one-day round trip.

## D2 RESULT (2026-10-12): FAIL (P1). The drift is real, but on the loser side

13,621 events (2,733 top, 2,799 bottom quintile) on 1,256 stocks from NSE's
own timestamps; portfolio months 2018-03 → 2026-09.

| | CAGR | Sharpe | max DD | EW: CAGR / DD | excess/month | t | dev / holdout | years + | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **D2** all top-quintile events | 25.0% | 0.72 | −34.1% | 16.0% / −36.9% | +0.74% | **1.58** | +0.97% / +0.24% | 6 of 7 | ✗ | ✓ | ✓ |
| D2T 20 slots, ₹2.5L | 15.8% | 0.49 | −47.9% | 15.9% / −36.9% | +0.02% | 0.07 | +0.19% / −0.34% | 4 of 7 | (reported only) | | |

**Event time, top minus bottom quintile, 60 sessions after the signal:**
**+3.67%, t 2.98** over 80 signal months (averaged by month first). The top
quintile earns **+0.8%** over EW; the bottom loses **−2.5%**.

- **The effect the literature describes is here,** at t 2.98 on the
  long-short measure. About three quarters of it is in the **losers keeping
  on losing**.
- **A long-only retail account cannot hold that side.** The winners'
  drift (+0.8% over 60 sessions) is about what costs and noise take. D2 is
  6 of 7 years positive but t 1.58. As 20 positions, D2T is zero.
- **Holdout weaker than development** (+0.24% vs +0.97% a month).
- **What this suggests, NOT tested and NOT a pass:** shorting bottom-quintile
  names through stock futures would hold the strong leg. That is a new
  hypothesis and would need its own pre-registration. Only F&O names have
  futures (about 200 of 1,256), and a stock-futures lot is about ₹6L, more
  than this account (see the F&O entry). Not pursued.

## D3 RESULT (2026-10-12): FAIL. The LLM reads the news right, and the open has already priced it

1,192 after-hours announcements from the 100 names, 2026-07-01 → 2026-09-30,
scored blind (no dates, no prices, shuffled) by Claude Opus 5.5 with the
frozen prompt: 135 GOOD, 58 BAD, 999 UNKNOWN. Scores were committed
(`docs/llm_pilot_scores.csv`, commit 83efb9f) before any outcome was
computed.

| label | n | open → close vs NIFTY (the trade) | overnight gap vs NIFTY (descriptive, not a criterion) |
|---|---|---|---|
| GOOD | 135 | +2.6 bps | **+46.7 bps, t 4.51** |
| BAD | 58 | +4.6 bps | −18.2 bps, t −0.90 |
| UNKNOWN | 999 | −8.0 bps | +11.5 bps |

**The trade:** net −8.6 bps a trade after the assumed 9 bps of costs,
t −0.72; GOOD net −6.4, BAD net −13.6. P1 ✗, P2 ✗.

- **The model's reading is informative:** GOOD items gap 65 bps above BAD
  items. On news the model cannot have seen in training, it separates good
  news from bad about as well as the published US results.
- **None of it is left after the open.** The whole reaction happens in the
  opening auction, the first price at which a retail order can trade
  after-hours news. From the open to the close there is nothing. This is
  the efficient-markets prediction, measured.
- **Where it could still pay, NOT tested:** announcements filed DURING
  market hours, if read and traded within seconds. That is a latency race
  against desks with direct feeds (Route A/B territory). It is not a retail
  edge.
- Small sample, one quarter, one scorer. The gap columns were looked at
  after the verdict and are description, not a test.

## Round 3 conclusion: 0 of 4 (rounds 1–3 combined: 0 of 11)

Three of the four effects were found **real** and **not tradeable** by
this account:

| effect | real? | why it does not pay here |
|---|---|---|
| turn of the month | index t 2.45 | 22 bps of costs per trade leave t 1.31 |
| post-earnings drift | long-short t 2.98 | the strong leg is the short side; long-only is t 1.58 |
| LLM news reading | gap t 4.51 | fully priced in the opening auction |
| pre-holiday | no | — |

That is the pattern of all eleven tests: the anomalies in the textbooks
exist in Indian data, and each is already harvested down to roughly the
cost of trading it, or sits on a side (short, pre-open) a retail cash
account cannot reach.
