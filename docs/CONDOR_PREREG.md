# NIFTY monthly iron condor: pre-registration (FROZEN 2026-10-10)

Client decision 2026-10-10: **paper only.** A pass leads to a forward paper
record (§6), never to orders.

**Hypothesis.** NIFTY index options are priced above the volatility that
then happens: the variance risk premium (Carr & Wu 2009, RFS; Bakshi &
Kapadia 2003, RFS). If that holds here, a monthly short iron condor earns
more than cash-and-carry after costs, with a drawdown an account can
survive. It is unconditional: no signal, no timing.

Written before any condor, strangle or option-selling return was computed.
The same rules as `docs/FNO_PREREG.md` apply: no edits after the first run,
amendments go in §8 with old and new numbers, and a new idea needs a new
document.

**Everything looked at so far** that touches this question, disclosed so a
reader can judge:
- **IV levels:** the NIFTY `iv30` medians by year (2019 14.5%, 2020 20.8%,
  2021 15.9%, 2022 18.1%, 2023 10.7%, 2024 13.4%, 2025 11.7%, 2026 12.5%), and
  its 0.92 ratio to VIX.
- **Realised volatility:** NIFTY's daily return sd over 2019–2026 is
  110 bps, about 17.5% a year. Compared naively with the IV medians, it
  suggests realised vol has often been ABOVE implied over this period
  (2020 dominates the sd). That is evidence against the hypothesis and is
  recorded here before the test, not explained away after it.
- **Strike availability**, from the check above: 90–239 traded NIFTY
  strikes about 27–34 days out, 50-point grid; far wings untraded in
  about a third of cycles.
- **The F&O long-short result** (0 of 3).

---

## 1. Instrument and calendar

- **Instrument:** NIFTY monthly options, European, cash-settled. Only the
  monthly series: weeklies changed under SEBI's Nov-2024 rules, monthlies
  did not.
- **Monthly expiries:** the distinct `expiry` dates of NIFTY FUTIDX in
  `futures.csv.gz`. One series carried under two dates within 7 days is
  merged onto the date that traded last (`fno_test.expiry_aliases`, the
  June-2023 holiday move).
- **Cycle k**, from expiry `E_k` to `E_{k+1}`:
  - **Selection day `s_k`:** the first session after `E_k`.
  - **Entry:** the **close** of the session after `s_k`.
  - **Exit:** the **close** of the 5th session before `E_{k+1}` (primary).
    Index options are cash-settled, so this is about expiry-week gamma and
    expiry-day margin, not delivery.
  - Position: one condor per cycle, never overlapping.
- **Periods:**
  - **Development:** entries 2019-02-01 → 2025-09-30 (about 80 cycles).
  - **Holdout:** entries 2025-10-01 → 2026-10-09 (about 12 cycles).
    Opened once, only after a development pass.

## 2. The condor (strikes chosen on `s_k`, traded on the next session)

- **Forward F:** the close of the NIFTY future expiring `E_{k+1}` on `s_k`.
- **IV and delta:** for every `E_{k+1}` option that traded on `s_k`
  (`contracts > 0`), compute IV and delta with the Black-76 functions in
  `scripts/fno_features.py` (rate 6.5%; T = calendar days / 365).
- **Short put:** the traded put with delta closest to **−0.15**.
- **Short call:** the traded call with delta closest to **+0.15**.
  Ties go to the strike further from F.
- **Wing width W:** 4% of F.
- **Long put:** the listed put strike closest to (short put − W). **Long
  call:** the listed call strike closest to (short call + W). Each must be
  strictly beyond its short strike. "Listed" means present in that day's
  options table (it has OI or turnover); it need not have traded.
- **No valid condor:** if any of the four legs cannot be chosen, the cycle
  is recorded as skipped. No substitution.

## 3. Prices, P&L and costs

- **Leg price on a day:** that day's **close** if the leg traded (`contracts
  > 0`), else NSE's **settle** price, which for an untraded option is
  NSE's own model mark. Every fallback is counted and reported.
- **Entry credit C:** short put + short call − long put − long call, at
  entry prices.
- **Exit debit D:** the same four legs at exit prices.
- **Gross P&L per unit:** C − D, in index points.
- **Risk capital per unit:** R = max(call wing width, put wing width) − C.
  This is the most the condor can lose at expiry.
- **Costs** (per unit, in points). The current Zerodha schedule is applied
  to every year (VERIFY; it is harsher than the pre-Oct-2024 STT):
  - **Brokerage:** ₹20 per order, 8 orders (4 legs × 2 sides), spread over
    the lot. Use the most recent NIFTY lot in the data for the whole period.
  - **STT:** 0.1% of premium on every sell.
  - **Exchange:** 0.03503% of premium.
  - **SEBI fee:** ₹10 per crore.
  - **Stamp:** 0.003% on buys.
  - **GST:** 18% on brokerage + exchange + SEBI fee.
  - **Spread, ASSUMED:** **0.5 index points per leg per side**, i.e. 4
    points per condor round trip.
- **Net P&L per unit:** C − D − costs.
- **Return on risk:** r = net P&L / R.
- **C0, per cycle:** NIFTY's near→next futures carry (`cal_spread_ann` on
  `s_k`) × calendar days held / 365. This is what the margin could earn in
  cash-and-carry.
- **Excess:** e = r − C0.

## 4. Pass bar (development). One hypothesis, so no Bonferroni.

| # | criterion |
|---|---|
| P1 | mean e > 0 with **t ≥ 2.0** over the development cycles (one non-overlapping return per cycle) |
| P2 | mean e > 0 in at least **5 of the 7** years 2019–2025 (by entry date) |
| P3 | mean e still > 0 after removing the best calendar year |
| P4 | **drawdown:** an account holding **2R** per condor (each condor risks at most half of it) has equity 1 + ½ Σ r. Its peak-to-trough fall must be **≤ 30%** |

**Win rate is reported and is NOT a criterion.** A condor wins most months
by construction; 29 Sep established that a high hit rate is not an edge.

**Holdout**, opened once, only after P1–P4 pass: its mean e must be > 0 AND
at least half the development mean.

**Variant budget: zero.** Reported only, never used to pass:
- short delta 0.10 / 0.20;
- wing width 3% / 6%;
- hold to expiry, settled at intrinsic on NIFTY's expiry-day close;
- spread 0.25 / 1.0 / 2.0 points;
- excluding cycles with any settle-price fallback.

**Always reported:** win rate, mean credit and R, put side vs call side
P&L, worst cycle, the 3 worst cycles, max drawdown, fallback counts, and the
mean rupee P&L per lot at the most recent lot size.

## 5. What a pass would and would not mean

- **A pass:** the premium survived costs and drawdown on 2019–2025 NIFTY
  monthlies. That earns a forward paper record, not capital.
- **A fail:** "NIFTY monthly condor: FAIL". No re-run with other deltas,
  wings or exits.

## 6. Forward paper record (only after a development AND holdout pass)

- `scripts/condor_paper.py`, on the pattern of `overnight_paper.py`:
  - on `s_k`, record the strikes chosen by §2;
  - during 15:00–15:25 on the entry and exit days, record `kite.quote`
    touch prices for all four legs. This measures the spread §3 assumes.
- It never places an order.
- **Minimum record:** 12 cycles. At 12 cycles the mean forward e at touch
  prices must be > 0, or the record is a forward FAIL.

## 7. Tests required before the first run

`scripts/condor_test.py`, each test built to fail on a planted bug:
- **Strike selection** uses only `s_k` data; prices dated after `s_k`
  cannot change the strikes.
- **Entry** is at the next session's close.
- **Exit** is the 5th session before `E_{k+1}`.
- **Black-76 delta sign and the 0.15 targets:** a synthetic chain with a
  known IV gives the analytically correct strikes.
- **Wings:** strictly beyond the shorts, at the listed strike nearest ± W.
- **The settle fallback** is used only for untraded legs, and counted.
- **P&L sign:** a condor whose legs all expire worthless earns C − costs;
  one where the underlying ends beyond a wing loses R + costs (on the
  hold-to-expiry variant).
- **Costs:** STT on sells only, stamp on buys only, 8 orders of brokerage.
- **The expiry alias** merges the June-2023 series.
- **Holdout guard:** the development run raises on any entry
  ≥ 2025-10-01.

## 8. Amendments

**A1 (2026-10-10, BEFORE the first run). Fills a case §3 did not define.**
A leg's row can be **absent** from the options table on a day: the table
keeps rows with OI or turnover, so a strike whose OI falls to zero
disappears. Measured before any return was computed: 0.07% of series-days.
- **Absent at entry:** the cycle is skipped, because the contract cannot be
  traded.
- **Absent at exit:** the leg is valued at intrinsic against the
  `E_{k+1}` futures close, and counted in the report.

Also confirmed when building: the expiry-alias rule merges TWO NIFTY
series. One is June 2023 (06-29 → 06-28). The other is September 2025
(09-25 → 09-30), from NSE's switch of the monthly expiry from Thursday to
Tuesday. Both fall under the rule as written in §1.

**A2 (2026-10-10, after the run; a disclosure, not a change).** The
2020-03-02 cycle lost 1.75× its risk capital, which a condor cannot do.
- **The cause:** on 2020-03-19 the short 10,350 put printed ₹2,450 on 2
  contracts (settle 2,076, intrinsic ≈ 2,144). That marks the 450-wide put
  spread at 725.
- **How rare:** it is the only cycle of 80 whose exit marks fall outside
  [0, width].
- **Primary result:** stands as run, under §3's "close if traded" rule.
- **With that spread capped at its 450 bound:** mean e −0.0188, t −0.69,
  max DD 116%, P1–P4 all still FAIL. The bad print does not decide the
  verdict.

---

## RESULT (2026-10-10): FAIL. The holdout stays closed.

Development: 80 cycles, 2019-02 → 2025-09, all traded (none skipped). 3 legs
used the settle fallback; none were absent at exit.

| | value | criterion |
|---|---|---|
| mean excess over C0, e (per cycle, on risk capital) | **−0.0279** | P1 needs > 0 at t ≥ 2: **t −0.86 FAIL** |
| years with mean e > 0 | 4 of 7 (2019, 2022, 2024, 2025) | P2 needs 5: **FAIL** |
| mean e without the best year (2025) | −0.0357 | P3: **FAIL** |
| max drawdown, 2R account | **146%** (116% with A2's cap) | P4 ≤ 30%: **FAIL** |
| win rate | 70% | not a criterion |

**By year (mean e):** 2019 +0.018, **2020 −0.197**, 2021 −0.083, 2022 +0.035,
2023 −0.012, 2024 +0.007, 2025 +0.042.

**The shape is the textbook one for option selling.**
- **The months:** 70% of months win, and the mean credit is 87.9 points
  against a mean risk of 607.7.
- **The losses:** a few months take it all back. 2020-03 lost −666 pts,
  2024-09 −624 and 2021-08 −416. That is 2 years out of 7 outside the
  COVID crash: the tail is not only 2020.
- **By side:** the put side made +7.1 pts a cycle and the call side −9.9.
- **Costs:** 7.2 pts a cycle, small next to the tail.
- **Per lot** (lot 65): mean −₹647 a cycle.

**Sensitivity (reported only):** no variant is positive.
- Best is hold-to-expiry: e −0.0037, t −0.11.
- delta 0.10 / 0.20: −0.022 / −0.014.
- wings 3% / 6%: −0.040 / −0.019.
- spread 0.25 → 2.0 pts: −0.024 → −0.050.
- Every variant's max drawdown is above 100%.

This matches the realised-above-implied fact disclosed before the run.

**Per §4: nothing is re-run with other settings. The holdout is not opened.
No paper record.** The record is "NIFTY monthly iron condor: FAIL".

