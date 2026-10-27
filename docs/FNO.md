# F&O long-short: blueprint (draft, 2026-10-10)

> **RESULT 2026-10-10: the pre-registered test FAILED, 0 of 3**
> (`docs/FNO_PREREG.md` RESULT).
> - H1 skew: −56.9 bps a cycle, t −1.50.
> - H2 parity deviation: −67.3 bps, t −1.81.
> - H3 pairs: −1.1 bps, t −0.13.
>
> Gross signs are opposite to the published ones or zero. The holdout was
> not opened and no paper record was started. The data and features below
> remain valid.

The client's brief asked for a market-neutral long-short framework in NSE F&O
with three parts: an ML signal, an LLM news/macro veto, and one rare,
high-probability setup. This document turns the brief into something that can
be tested. Every number here is measured on the data in `data/fno` unless it
is marked **ASSUMED** or **VERIFY**.

Status: data and features are BUILT. No strategy has been run. Nothing below
is a result.

---

## 0. What exists now

| table | rows | what it is |
|---|---|---|
| `futures.csv.gz` | 1.08M | every stock/index future, every session 2019-01-01 → 2026-10-09 |
| `options/options_YYYY.csv.gz` | 28M | every option with OI or turnover |
| `delivery.csv.gz` | 4.33M | delivery % per cash symbol (MTO) |
| `participant_oi/vol.csv.gz` | 9,635 | Client / DII / FII / Pro positions and volume |
| `kite/day/` | 1,927 sessions | INDIA VIX, NIFTY 50, NIFTY BANK |
| `features/expiry_iv.csv.gz` | 731K | per (date, underlying, expiry): forward, ATM IV, 25Δ put/call IV, skew, OI/volume by side |
| `features/iv_daily.csv.gz` | 355K | per (date, underlying): 30-day IV and skew, front/back IV, put-call ratios |
| `features/oi_buildup.csv.gz` | 361K | per (date, underlying): roll-safe return, all-expiry OI change, buildup class, carry, delivery z-score |
| `features/corp_actions.csv.gz` | 215 | every re-strike: 78 multiplicative, 124 dividend, 13 demerger relistings |

Built by `scripts/fno_history.py` (download, build) and
`scripts/fno_features.py`; 20 tests; every mutation tried was caught except
one equivalent mutant.

**Checks the features passed:**
- NIFTY `iv30` vs INDIA VIX: correlation **0.98** over 1,926 days, ratio
  **0.92** every year (VIX is variance-swap style and carries the skew, so it
  should sit above ATM). On 24 Mar 2020: VIX 83.6, iv30 76.7.
- Put-call-parity forward vs the futures close: median gap **5.5–8.9 bps**
  each year.
- NIFTY put skew positive on **95.5%** of days.
- Dividend ex-dates: raw futures return minus NIFTY = **−3.47%** against a
  median D/P of 2.96% (corr 0.84). The futures do drop by D, so the build
  adds D back.
- The 13 relisted days are all known demergers. The real crashes (YESBANK
  Mar-2020, ZEEL Jan-2019, the COVID days, POLICYBZR Sep-2026) are NOT
  flagged as corporate actions.

**Survivorship-free universe for free.** The bhavcopy lists which stocks
had F&O on each day: 351 names over the period, 138 of them gone by now,
and 137–258 listed at any one time. Every equity study in this repo had to
argue around survivorship; this one does not have to.

---

## 1. Operational mechanics: "take delivery and sell on expiry"

**1a. Stock F&O is physically settled; index F&O is cash settled.**
VERIFY with broker: SEBI phased this in from 2018, and every stock derivative
was physically settled by Oct 2019. A stock future held into expiry, or an
in-the-money stock option, ends as a delivery of shares at the settlement
price. Brokers charge rising delivery margins in expiry week and may
square off positions early; Zerodha has its own rules. **Rule for this
framework: every position is closed or rolled before expiry week.** Delivery
is never the plan.

**1b. What "buy delivery, sell the future, deliver at expiry" actually is.**
It is cash-and-carry arbitrage. You buy shares, sell the same-expiry future,
and lock in the basis. Measured: near→next futures carry, median by year,
annualised: 2019 4.8%, 2020 2.8%, 2021 4.4%, 2022 4.0%, 2023 7.0%, 2024 7.3%,
2025 6.3%, 2026 5.9%. Futures sit about +24 bps over the underlying with about 18 days left.
Delivery costs **22.2 bps** a round trip at ₹6.2L (0.1% STT on each leg), and
a month of carry is about 50 bps. That leaves roughly a money-market rate:
**a benchmark, not an edge.** It is control C0 below.

**1c. Covered call / protective put.** Covered call: long shares or a future,
short an OTM call. It sells the variance risk premium, has capped upside and
full downside, and is not market-neutral. Protective put: pays that premium
instead. Neither is a long-short. Both stay possible as overlays once a
signal exists.

**1d. Market-neutral in practice.** There are two ways to cancel the market:
- **Index hedge:** long a stock, short beta × notional of NIFTY futures.
- **Pair:** long stock A, short stock B in the same sector.

Measured on daily returns adjusted for corporate actions:

| | daily sd | variance removed |
|---|---|---|
| single stock future (median) | 214 bps | — |
| hedged with NIFTY at its own beta (median beta 1.13) | 181 bps | **27%** (p10–p90: 15–40%) |
| 30 most-correlated liquid pairs (median ρ 0.69) | 143 bps | **54%** |
| NIFTY itself | 110 bps | — |

**"Market-neutral" does not mean "low risk."** Only a quarter of a typical
stock's daily variance is the market. Even a good pair keeps 46% of a single
leg's variance.

**1e. Contract sizes decide what is buildable on ₹2.5L.**
- **Stock futures:** median **₹6.2L** a lot (p10–p90 ₹4.5–9.4L; cheapest
  VEDL ₹3.0L).
- **Index futures:** NIFTY **₹14.7L**, BANKNIFTY ₹16.6L.
- **Margin:** ASSUMED about 20% of notional for NRML; VERIFY on Kite's margin
  calculator.

What follows:
- **An index hedge cannot be built.** One NIFTY lot hedges about ₹13L of
  stock. Hedging a single ₹6.2L stock lot with it over-hedges by about 2×,
  and the margin alone is about ₹3L.
- **One stock pair is the most this account can hold.** Two lots of about
  ₹6.2L at about ₹1.2L margin each use roughly all of ₹2.5L.
- **One pair is a large position for this account.** A 1-sd day is
  143 bps × ₹6.2L ≈ **₹8,800 = 3.5% of equity**. A 1-sd month is about
  **₹40K = 16%**. Gaps cannot be stopped.
- **Lot rounding breaks exact neutrality.** One lot of each leg gives a hedge
  ratio fixed by the lot sizes, not by the beta. The residual exposure
  has to be measured per pair.

**This is the first gate. A pair framework on ₹2.5L cannot meet the brief's
"capital preservation" requirement at minimum size.** It either needs more
capital (about ₹10L+ for 2–3 pairs with a buffer) or has to be paper only.

---

## 2. Feature map: the brief vs the data

| brief asks for | column(s) | status |
|---|---|---|
| Price action | `ret` (roll-safe, adjusted for corporate actions); OHLC in futures | ✅ |
| Order flow / delta | — | ❌ needs tick data. Kite does not serve tick history; can only be collected forward |
| Volume | `contracts`, `value_lakh`; options `vol_ce/vol_pe` | ✅ |
| Delivery % | `deliv_pct`, `deliv_pct_z20` | ✅ |
| OI buildup | `buildup`, `oi_chg_pct`, `near_oi_share` (roll) | ✅ sign-only classes, no fitted threshold |
| IV skew | `skew25`, `skew30`, `iv_p25/iv_c25` | ✅ |
| IV level / term | `iv30`, `iv_front`, `iv_back` | ✅ |
| Put-call ratios | `pcr_oi`, `pcr_vol` | ✅ |
| Positioning | participant OI (FII / Pro / Client, index and stock futures) | ✅ market-level only |
| Basis / carry | `cal_spread_ann`; `underlying` from Jul 2024 | ✅ |
| Parity deviation | `parity` vs `fut_close` in `expiry_iv` | ✅ |
| Realised vol | from `ret` | to build |
| Earnings dates | — | ❌ to source (NSE board-meeting announcements) |
| News / transcripts / macro | — | ❌ no archive; see §4 |

Note on stock skew: the median `skew30` across stocks is **−0.0007** and is
positive on only 43% of days, unlike the index. Single-stock smiles here are
much flatter than NIFTY's. Any skew signal works on small cross-sectional
differences.

---

## 3. Hypotheses: test published effects, do not mine

The lesson of this repo (0 of 37 swing rules, 0 of 21 intraday families, the
July-7 overfit) is that searching combinations of features on one dataset
finds noise. So the first tests are effects with **published out-of-sample
evidence**, each defined in advance and with a pass bar:

| id | hypothesis | source | portfolio |
|---|---|---|---|
| **H1** | Stocks with a steep put skew underperform | Xing, Zhang & Zhao (2010), JFQA | monthly: long lowest-skew quintile, short highest, via futures |
| **H2** | Options-implied forward above the futures price (calls rich) predicts outperformance | Cremers & Weinbaum (2010), JFQA | same structure on `parity/fut_close − 1` |
| **H3** | Distance-method pairs mean-revert | Gatev, Goetzmann & Rouwenhorst (2006), RFS; weaker since 2000 (Do & Faff 2010, FAJ) | same-sector pairs, futures legs, close before expiry week |
| **H4** | OI buildup / delivery / participant positioning predict returns | no strong published out-of-sample support | exploratory only, Bonferroni-counted |
| **C0** | Cash-and-carry | — | the benchmark every strategy must beat after costs |

**SUPERSEDED by `docs/FNO_PREREG.md` (frozen 2026-10-10).** That file holds
the exact definitions, the pass bar (t ≥ 2.4, 5 of 7 years, both legs) and
the holdout. C0 is reported there, not a pass condition. Draft as first
written:
- net of costs (stock futures **3.44 bps** a round trip at ₹6.2L per leg,
  plus a measured spread);
- t ≥ 2 after the clustering correction (`scripts/_stats.py`); Bonferroni
  across H1–H4;
- positive in at least 5 of 7 years;
- both legs contribute;
- survives dropping the best year;
- beats C0.

H1 and H2 are designed as cross-sectional long-shorts. They need the
quintile book (many names), so on ₹2.5L they can only run as a paper or
index-level test. That is fine for deciding whether an edge exists.

---

## 4. ML framework

**Models.**
- **Ridge first:** the frozen R5 swing model was a ridge.
- **Then gradient boosting:** sklearn `HistGradientBoostingRegressor`, already
  installed. LightGBM or XGBoost would add nothing that matters at this scale.
- **No LSTM.** There are about 1,900 sessions × about 200 names of daily,
  tabular, cross-sectional data. Sequence models need far more data per
  name, and their failures are harder to audit.

**Target.**
- Forward return of the hedged position (pair spread, or stock minus
  beta × NIFTY) over a fixed horizon that **ends before expiry week**,
  **net of costs**.
- A regression or rank target, **not "win / lose."** On 29 Sep "success rate"
  was rejected as an objective with evidence: random entries with a wide stop
  "won" 81% of the time and lost money. The brief's "even one trade a month
  but very high win rate" is the same trap. A 90%-win setup that loses 10×
  on the other 10% is a losing system. What gets optimised is expectancy and
  drawdown, with win rate reported.

**Validation.** Same discipline as R5 / V8:
- expanding-window walk-forward by year, with labels purged where they
  overlap the training end;
- point-in-time universe from the bhavcopy;
- features ranked cross-sectionally per date;
- **the last 12 months held out and touched once**;
- a leak test for every feature, plus mutation tests;
- a budget of variants fixed in advance, every variant logged.

**What the ML predicts.** The brief's "structural breakout levels" were
already tested on these names and failed every time: the daily breakouts in
swing batch 2, the 21 intraday level-break variants, and the breakout rule
run through the bot's own replay. The ML here only ranks expected hedged returns. If ranking
cannot beat the published single-factor versions of H1/H2, the ML adds
nothing.

---

## 5. The LLM layer: what it can and cannot be

**It cannot be backtested here.**
1. **No archive.** There is no point-in-time archive of news, transcripts or
   central-bank text.
2. **Look-ahead contamination.** An LLM scoring a 2021 headline already
   "knows" what happened after 2021 from its training data. Any historical
   sentiment score is contaminated. The published LLM-for-returns results
   that hold up use text dated after the model's cutoff.

**What replaces it in the backtest: a deterministic event filter.** Most of
what a "news risk veto" is meant to block is scheduled:
- **earnings date** inside the holding period (to source: NSE board-meeting
  announcements);
- a **corporate action** (have it: `corp_actions`);
- **F&O ban list** (NSE publishes a daily file; VERIFY the history goes
  back);
- **expiry week** (have it).

These can be tested and measured: compare returns on blocked vs unblocked
days.

**The LLM goes forward-only, in shadow, logged.**
- **Inputs:** the day's news for each candidate, after the close.
- **Output:** a fixed schema: `{risk_score 0–100, events[], conflicts_with_signal: bool, rationale}`.
- **Use:** recorded beside each signal; it **vetoes nothing** until enough
  forward observations show that vetoed setups did worse than kept ones.
  This is the same rule the regime override had to pass.
- **Default if it cannot show value:** removed.

---

## 6. The unified setup (draft checklist)

A trade fires only when ALL of these hold. Thresholds are placeholders to be
frozen, not fitted:

1. **Signal.** The hypothesis that passed (H1/H2/H3, or the ML that beat
   them) puts the pair or stock in its extreme bucket.
2. **Liquidity.** Both legs are in the top half of F&O names by futures
   turnover, and the near-month carries the OI (`near_oi_share`), so there
   is no thin-contract fill.
3. **Events clear.** No earnings, corporate action, ban-list entry or expiry
   week inside the holding period.
4. **OI agrees (only if H4 passed).** The buildup class on the long leg is
   not SHORT_BUILDUP, and on the short leg not LONG_BUILDUP.
5. **Size fits.** Margin for both legs ≤ 50% of equity, leaving a buffer for
   mark-to-market.
6. **LLM (shadow only).** Logged, not acted on.

---

## 7. Risk management

**Sizing.**
- One lot per leg is the floor and also the ceiling at ₹2.5L.
- Hedge ratio = lots × lot size × price per leg. Record the residual beta;
  skip the pair if it exceeds 0.3 × the notional of one leg.

**Exits.**
- **Stop:** on the pair's spread, in units of its own trailing sd (for
  example 2.5 sd adverse), checked at the close. An intraday stop on a
  two-leg futures position gets legged badly.
- **Time stop:** the horizon. **Always exit before expiry week.**

**Loss limits.**
- **Per trade:** loss capped at 2% of equity (₹5K). With one pair at
  143 bps/day ≈ ₹8.8K a 1-sd day, **that cap would be hit by an ordinary
  day.** This is the clearest single sign that ₹2.5L is too small for this
  structure.
- **Account:** stop trading at −10% of equity from peak; review before
  restarting.

**Gap risk.** Measured on the overnight book earlier: a single name's worst
gap was −1,160 bps. Pairs reduce this but do not remove it.

---

## 8. Plan

1. ✅ Client decided 2026-10-10: **paper only.**
2. ✅ `docs/FNO_PREREG.md` written and frozen 2026-10-10 (client: paper
   only).
3. `scripts/fno_test.py`:
   - H1/H2 as quintile long-shorts;
   - H3 as distance-method pairs;
   - C0 as the benchmark;
   - costs from `src/costs.py`;
   - leak and mutation tests in the house style.
4. Only if something passes: the ML ranking (§4), then a forward paper
   record, then a live LLM shadow log. Money comes after a forward record,
   never before.

---

## Known limits of the data

- **Prices are raw** (not split-adjusted). `oi_buildup.ret` is adjusted; any
  other price series built from these tables must use `corp_actions`.
- **No underlying price before Jul 2024.** UDiFF carries it from Jul 2024;
  before that use the cash bhavcopy in `data/nse_hist` (raw).
- **Option prices are closes of contracts that traded that day.** Strikes
  that traded early and went quiet carry stale closes into the IV. IV fill
  rates: ATM 84–89% of expiry rows.
- **Flat discount rate (6.5%) in Black-76.** The forward carries the carry,
  so the rate only discounts the premium.
- **Files NSE does not have:** F&O 2021-03-30; participant OI 2023-10-31 and
  volume 2024-08-27.
- **Demerger days (13)** have no recoverable return.
