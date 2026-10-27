# F&O long-short: pre-registration (FROZEN 2026-10-10)

Client decision 2026-10-10: **paper only.** No money moves on any outcome of
this document. A pass leads to a forward paper record (§9), not to orders.

Written after the data and features were built (`docs/FNO.md`) and **before
any hypothesis below was computed on them.** Up to this point no portfolio,
quintile, pair or signal return has been calculated. Everything that has
been looked at, listed so a reader can judge what could have leaked into
the choices below:
- the data checks in `docs/FNO.md` §0;
- the risk and cost figures in `docs/FNO.md` §1;
- the cross-stock distribution of `skew30` (median −0.0007, positive on 43%
  of stock-days). This is a level, not a return;
- the daily-return correlations and spread sd of the 30 most-correlated
  pairs among the 60 most liquid names (median correlation 0.69, spread
  143 bps/day). These are co-movement, not a pairs-strategy return. H3
  selects pairs by SSD over a rolling window, not from that list.

**Rules for changing this file:**
- Do NOT edit any part of this file after the first run.
- A bug fix that changes a number is recorded in §11 with the old and new
  value, and is never applied silently.
- A new idea gets a new pre-registration, never an edit here.

---

## 1. Data and calendar

- **Source:** the tables listed in `docs/FNO.md` §0, as of commit `e3d673f`
  (data to 2026-10-09).
- **Expiry calendar:** the distinct `expiry` dates of FUTSTK contracts in
  `futures.csv.gz`. This is the monthly stock expiry, whichever weekday NSE
  used that month.
- **Cycle k:** formed at the monthly expiry `E_k` and traded until just
  before `E_{k+1}`.
  - **Signal day `s_k`:** the first session after `E_k`. Every signal input
    is taken from bhavcopy data dated ≤ `s_k`.
  - **Entry:** the futures **close** of the session after `s_k` (`s_k+1`).
    This is the next session, because the signal uses `s_k`'s own close. A
    futures close is the last-30-minute weighted average, which can be
    reached by trading in that window.
  - **Exit:** the futures **close** of the 5th session before `E_{k+1}`, so
    every position is closed before expiry week (when delivery margins start).
  - **Contract:** the one expiring at `E_{k+1}`, the near month after `E_k`.
- **Periods:**
  - **Development:** cycles with entry 2019-02-01 → 2025-09-30.
  - **Holdout:** cycles with entry 2025-10-01 → 2026-10-09. Opened once,
    only for a hypothesis that has already passed development.

## 2. Universe (point-in-time, every cycle)

A stock is eligible in cycle k only if ALL of these hold:
1. It has a FUTSTK contract expiring at `E_{k+1}` on `s_k` and on `s_k+1`.
2. It ranks in the top 100 of FUTSTK underlyings by the median of daily
   futures `value_lakh` (all expiries summed) over the 20 sessions ending
   `s_k`.
3. It has at least 250 sessions of `oi_buildup.ret` history before `s_k`
   (needed by H3).
4. For H1 and H2 only: the signal input exists on at least 3 of the 5
   sessions ending `s_k`.

Indices are never in the universe. No other filter applies. In particular,
names are not excluded for corporate actions that happen *later* in the
cycle: those are unknowable on `s_k`, and excluding them would be
look-ahead.

## 3. Returns, corporate actions and costs

- **Return of a leg:** the contract's close-to-close returns over the hold,
  compounded. Each day's return is adjusted exactly as in
  `fno_features.oi_buildup`:
  - **multiply:** divide by the factor;
  - **subtract:** add the dividend back;
  - **relisted:** the leg is closed at the previous session's close (that
    day's return cannot be recovered).
- **Weights:** equal notional per name within a leg. Long book = +1, short
  book = −1. The portfolio is dollar-neutral. Paper, so fractional lots are
  allowed. §8 states what real lots would do.
- **Costs, per name, per side:**
  - **Broker charges:** from `src/costs.py` with the stock-futures model in
    `scripts/_vehicles.py` (`_FUT`), at that contract's real notional (lot ×
    price where the lot is known, from Jul 2024; ₹6.2L before that).
  - **Spread:** ASSUMED 5 bps per side (10 bps a round trip) on every leg.
    This is not measured history; the paper record (§9) measures it.
- **Cycle return of a hypothesis:** the long-book return minus the
  short-book return, minus costs, as a fraction of one book's notional.

## 4. H1: volatility skew (Xing, Zhang & Zhao 2010)

- **Signal:** the mean of `iv_daily.skew30` (25-delta put IV minus 25-delta
  call IV, 30-day constant maturity) over the 5 sessions ending `s_k`.
- **Long / short:** sort the eligible names by the signal into quintiles.
  - **Long:** the LOWEST quintile (flattest put skew).
  - **Short:** the HIGHEST quintile.
- **Predicted sign:** long minus short > 0.
- **Deviation from the paper:** they used OTM put IV minus ATM call IV.
  The 25-delta risk reversal is used here because it exists for every name
  on every day in these tables. Fixed now, not chosen later.

## 5. H2: parity deviation (Cremers & Weinbaum 2010)

- **Signal:** the mean over the 5 sessions ending `s_k` of
  `parity / fut_close − 1`, from `expiry_iv` for the `E_{k+1}` expiry.
  - A positive value means the options imply a higher forward than the
    future trades at, i.e. calls are rich relative to puts.
- **Long / short:** sort into quintiles.
  - **Long:** the HIGHEST quintile.
  - **Short:** the LOWEST quintile.
- **Predicted sign:** long minus short > 0.
- **Deviation from the paper:** they used the call-minus-put IV spread on
  American options. On NSE's European options, with the future as the
  forward, that spread is exactly a gap between the parity forward and the
  futures price, so this is the same quantity. Option closes in
  thinly-traded strikes are stale, and that noise works against H2, never
  for it.

## 6. H3: distance-method pairs (Gatev, Goetzmann & Rouwenhorst 2006)

- **Formation**, on `s_k`:
  - For every eligible name, build a cumulative-return index from adjusted
    `oi_buildup.ret` over the 250 sessions ending `s_k`, normalised to 1 at
    the start.
  - For every pair of eligible names, compute the sum of squared differences
    (SSD) of the two indices over that window.
  - Keep the **20 pairs with the smallest SSD**, using each name in at most
    one pair (greedy: take the smallest SSD, remove both names, repeat).
  - No sector restriction, as in the paper.
  - σ = the standard deviation of the pair's index difference over the
    formation window.
- **Trading**, within cycle k:
  - **Spread:** each day, rebase both indices to 1 at the close of `s_k`.
    The spread is the difference between the two rebased indices.
  - **Open:** when the spread's absolute value first exceeds **2σ** at a
    close, open at the NEXT session's close. Long the lower leg, short the
    higher one, equal notional.
  - **Close:** at the next session's close after the spread crosses zero, or
    at the cycle exit, whichever comes first.
  - At most one round trip per pair per cycle.
  - If either leg has a relisted event, the pair is closed at the previous
    close.
- **Cycle return:** the mean, across the 20 pairs, of each pair's net return.
  A pair that never opens contributes 0. This is the paper's
  "committed capital" convention.
- **Predicted sign:** > 0.
- **Deviations from the paper:**
  - They traded for 6 months after a 12-month formation. Here trading is
    limited to one cycle (about 15 sessions) so positions never reach expiry
    week.
  - They used daily total-return indices; here the inputs are futures
    returns adjusted for corporate actions.

## 7. Control C0 (reported, not a pass condition)

**Cash-and-carry:** buy shares at delivery, sell the `E_{k+1}` future, hold
to `E_{k+1}`, deliver. Per cycle, on the eligible universe:
- **Return:** the median of `cal_spread_ann` on `s_k` × (calendar days held
  / 365), minus delivery costs (22.2 bps) and futures costs.
- **Purpose:** a reference for what the same capital earns with little
  risk.

This supersedes the "beats C0" line in `docs/FNO.md` §3. A dollar-neutral
long-short is self-financing, so its excess return and C0's return on
capital are not the same unit.

## 8. Pass bar (development)

One series per hypothesis: its net cycle return, about 80 development
cycles. A hypothesis PASSES development only if ALL of these hold:

| # | criterion |
|---|---|
| P1 | mean net cycle return > 0 with **t ≥ 2.4**. That is Bonferroni over 3 hypotheses at two-sided 5%, on one portfolio return per cycle; cycles do not overlap, so no clustering correction is needed |
| P2 | positive in at least **5 of the 7** calendar years 2019–2025 (by entry date) |
| P3 | mean still > 0 after removing the best calendar year |
| P4 | H1/H2: **both legs contribute.** Long book minus the equal-weight universe > 0, AND the equal-weight universe minus short book > 0. H3: the long-leg and short-leg mean returns over the equal-weight universe are both reported, and their difference must be > 0 |
| P5 | net of costs as defined in §3, at the ASSUMED 5 bps a side |

**Holdout**, opened once, only for a hypothesis that passed P1–P5: its mean
net cycle return over the holdout must be > 0 AND at least half its
development mean. No t-statistic is required, because about 12 cycles cannot
carry one.

**Reported but never used to pass or fail:** spread at 0 and 10 bps a side;
terciles instead of quintiles (H1/H2); the 1-day signal instead of the 5-day
mean (H1/H2); 10 and 30 pairs (H3); the portfolio's beta to NIFTY; turnover;
the worst cycle; maximum drawdown.

**Variant budget: zero.** The primary specification above is the only one
that can pass. If nothing passes, the record is "F&O long-short: 0 of 3",
and nothing here is re-run with other settings.

**Real lots (reported, informational):** for each passing hypothesis,
re-run the cycle book with whole lots (1 lot per name, real lot sizes from
Jul 2024). Report its margin use (ASSUMED 20% NRML) against ₹2.5L.

## 9. If something passes: the forward paper record

- `scripts/fno_paper.py`, on the same pattern as `scripts/overnight_paper.py`:
  - **Signal:** after the close of each `s_k`, record the cycle's longs and
    shorts (or pairs) using this document's exact rules.
  - **Fills:** during 15:00–15:25 on `s_k+1` and on the exit day, record
    `kite.quote` touch prices for every leg. This measures the real spread
    that §3 assumes.
- It never places an order.
- **Minimum record:** 12 cycles (one year) before any capital discussion.
- **Forward verdict at 12 cycles:** the mean net forward cycle return, at
  touch prices, must be > 0. If it is ≤ 0, the hypothesis is recorded as
  failing forward. The paper record may continue, but only as a curiosity.

## 10. Tests and audits required before the first run

`scripts/fno_test.py` ships with tests, each built to fail on a planted bug:
- **Look-ahead:** a signal value dated `s_k+1` must not change the cycle-k
  portfolio. Entry must be at `s_k+1`'s close, not `s_k`'s.
- **Expiry boundary:** the exit is the 5th session before `E_{k+1}`, and no
  position is held into expiry week.
- **Corporate actions:**
  - a 1:1 bonus inside the hold leaves a flat stock's return at 0;
  - a dividend inside the hold is added back;
  - a relisted leg is closed at the previous close.
- **Universe:** a name without a contract on `s_k+1` is excluded, and a name
  that LATER leaves F&O is still included in cycles where it was listed (no
  survivorship).
- **Costs:** charged on both books, both sides.
- **Quintile direction:** swapping long and short flips the sign.
- **H3:** pair formation uses only the 250 sessions ending `s_k`; a pair
  that never diverges contributes 0; at most one round trip per pair.
- **Holdout guard:** the development run cannot read a cycle whose entry is
  ≥ 2025-10-01 (it raises).

Mutation checks are run against each of these, in the house style.

## 11. Amendments

**A1 (2026-10-10, after run 1). Bug fix: one contract under two expiry
dates.**
- **The defect:** NSE moved the June-2023 expiry for Bakri Eid. The bhavcopy
  labels the series 2023-06-29 until 06-27 and 2023-06-28 on its final day.
  The calendar treated the two dates as separate contracts, so the cycle
  entering 2023-05-29 found no "contract expiring at E_{k+1}" on its signal
  day. Its universe was empty and the cycle was silently dropped.
- **The fix:** `expiry_aliases` merges expiry dates ≤ 7 days apart onto the
  one that actually traded on the final day. It is the only such case in
  2019–2026. This restores the §1 definition rather than changing it.

| | run 1 (bug) | run 2 (fixed) |
|---|---|---|
| H1 | n 67, −59.7 bps, t −1.55 | n 68, −56.9 bps, t −1.50 |
| H2 | n 67, −67.5 bps, t −1.79 | n 68, −67.3 bps, t −1.81 |
| H3 | n 67, −1.6 bps, t −0.19 | n 68, −1.1 bps, t −0.13 |
| verdict | 0 of 3 | 0 of 3 |

**A2 (disclosure, not a change). 2019 contributes no cycles.** §2.3
requires 250 sessions of history, and the data starts 2019-01-01, so the
universe is empty until Jan 2020. Development is therefore 68 cycles,
Feb 2020 → Sep 2025, and P2 can count at most 6 of the 7 years. I did not
see this consequence when writing §2. It does not decide anything: every
hypothesis fails P1 by a wide margin.

---

## RESULT (2026-10-10): 0 of 3 pass. The holdout stays closed.

Development: 68 monthly cycles, Feb 2020 → Sep 2025, about 95 names per
cycle. Net = after charges and the assumed 5 bps a side spread
(26.8 bps a cycle for a quintile book).

| | net / cycle | t | years + | gross / cycle | gross t | verdict |
|---|---|---|---|---|---|---|
| H1 skew | **−56.9 bps** | −1.50 | 1 of 7 | −32.9 | −0.85 | FAIL P1–P4 |
| H2 parity deviation | **−67.3 bps** | −1.81 | 1 of 7 | −40.7 | −1.08 | FAIL P1–P4 |
| H3 distance pairs | **−1.1 bps** | −0.13 | 2 of 7 | +3.1 (at 0 spread) | +0.37 | FAIL P1–P3 |
| C0 cash-and-carry | +13.1 bps | — | — | carry 5.3%/yr | — | reference |

*Gross figures were run 1's breakdown; run 2's gross is within 3 bps.*

**H1 and H2:**
- The gross sign is the OPPOSITE of the published one, and not
  significant.
- Neither leg beats the universe in the predicted direction.
- Costs alone (26.8 bps a cycle) are about as large as the whole gross
  number.

**H3:**
- Only 4.1 of 20 pairs open in a cycle. A 2σ divergence (σ from 250
  formation days) rarely happens inside the ~15-session window this cycle
  allows.
- Zero at gross, and negative net.

**Reported only (never used to pass):** no variant has a positive t above
0.4. At 10 bps a side, H1 and H2 reach t −2.0 / −2.35, which is the cost
line rather than a signal.

**Checks that the result is real and not a bug:**
- Leg returns match a compounding of the adjusted daily series to the
  bps, on spot-checked names.
- The universe's equal-weight return tracks NIFTY at a cross-cycle
  correlation of 0.95.

**Per §8: nothing here is re-run with other settings. The holdout is not
opened. No forward paper record starts.** The record is "F&O long-short:
0 of 3."
