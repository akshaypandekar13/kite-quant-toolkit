"""
Bid-ask spread estimation
=========================
The overnight strategy nets +3.95 bps a night before slippage. Crossing a
midcap futures spread at 09:16 costs somewhere between "half of that" and "all
of it", so the spread estimate IS the decision — and an estimator that
understates it would green-light a losing trade.

Every fixture builds a price series with a KNOWN spread: an efficient log-mid
random walk, observed through a bid-ask bounce. The estimator must recover the
number that was put in. The tests that matter most are the ones pinning where
estimation is IMPOSSIBLE, because that is where a plausible-looking number
would do the damage.
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._spread import (  # noqa: E402
    CALM_WINDOW, SpreadEstimate, abdi_ranaldo_s2, corwin_schultz_s,
    estimate_spread, spread_bps,
)


def _bars(true_bps, sigma_bps, n_bars=4000, trades=60, drift_bps=0.0, seed=0):
    """OHLC bars off ONE continuous price path with a known spread.

    The walk must NOT restart each bar. An earlier version of this fixture did,
    which made consecutive bars independent rather than a price series;
    Abdi-Ranaldo depends on the close-to-next-bar-midrange relationship and
    read a flat ~35 bps regardless of the truth, while Corwin-Schultz, which
    mostly uses within-bar ranges, looked healthy. The broken fixture pointed
    at the wrong estimator.
    """
    rng = np.random.default_rng(seed)
    s = true_bps / 1e4
    steps = rng.normal(drift_bps / 1e4 / trades,
                       sigma_bps / 1e4 / math.sqrt(trades), n_bars * trades)
    mid = np.exp(np.cumsum(steps))
    px = (mid * (1.0 + rng.choice([-0.5, 0.5], size=n_bars * trades) * s)
          ).reshape(n_bars, trades)
    return px.max(1), px.min(1), px[:, -1]


def _recover(true_bps, **kw):
    h, l, c = _bars(true_bps, **kw)
    return spread_bps(abdi_ranaldo_s2(np.log(h), np.log(l), np.log(c)))


# ── it recovers what was put in ─────────────────────────────────────────────


@pytest.mark.parametrize("true_bps", [0.0, 3.0, 5.0, 10.0, 20.0])
def test_a_known_spread_comes_back(true_bps):
    got = _recover(true_bps, sigma_bps=12.0, n_bars=40000, seed=int(true_bps) + 1)
    assert got == pytest.approx(true_bps, abs=1.2), f"{got:.2f} vs {true_bps}"


def test_a_zero_spread_reads_zero_and_is_not_floored_upward():
    """THE failure mode that would fake an edge into existence — in reverse.

    Corwin-Schultz's usual recipe clamps each negative pair estimate to zero
    before averaging, which reports ~8 bps on a spread of exactly nothing.
    Averaging S^2 first and rooting after must not do that.
    """
    got = _recover(0.0, sigma_bps=12.0, n_bars=40000, seed=99)
    assert abs(got) < 1.2, got


def test_it_is_monotone_in_the_true_spread():
    vals = [_recover(t, sigma_bps=12.0, n_bars=30000, seed=7)
            for t in (0.0, 5.0, 10.0, 20.0)]
    assert all(b > a + 3.0 for a, b in zip(vals, vals[1:])), vals


@pytest.mark.parametrize("sigma", [8.0, 20.0, 40.0])
def test_volatility_alone_does_not_create_a_spread(sigma):
    """A violent but drift-free window must still read ~0 at a true zero,
    or every wide-ranging name would look expensive to trade."""
    got = _recover(0.0, sigma_bps=sigma, n_bars=40000, seed=int(sigma))
    assert abs(got) < 2.5, f"sigma {sigma} manufactured {got:.2f} bps"


# ── where estimation is IMPOSSIBLE, and must not quietly return a number ────


def test_drift_destroys_the_estimate_which_is_why_09_16_is_not_measured():
    """THE MOST IMPORTANT TEST HERE.

    Abdi-Ranaldo assumes a martingale. At 09:16 the gap is still resolving —
    the ladder measures midcap dropping 5.24 bps in that single minute. Under
    drift the estimator does not merely lose precision, it goes hard negative,
    and a reader who saw '-19' might take it for 'no spread'.

    Pinned so nobody later points this module at the opening window.
    """
    calm = _recover(5.0, sigma_bps=60.0, n_bars=30000, drift_bps=0.0, seed=3)
    drifting = _recover(5.0, sigma_bps=60.0, n_bars=30000, drift_bps=20.0, seed=3)
    assert calm == pytest.approx(5.0, abs=2.0), calm
    assert drifting < -10.0, f"drift should wreck it, got {drifting:.2f}"


def test_removing_the_windows_own_drift_makes_it_worse_not_better():
    """The obvious repair, pinned as a dead end so it is not retried.

    Subtracting the realised per-bar drift removes real price variation and
    correlates the residuals, manufacturing a large POSITIVE spread where
    there is none — which is the direction that would kill a good strategy
    rather than the one that would green-light a bad one, but wrong either way.
    """
    h, l, c = _bars(0.0, sigma_bps=60.0, n_bars=20000, drift_bps=20.0, seed=5)
    lh, ll, lc = np.log(h), np.log(l), np.log(c)
    mu = (lc[-1] - lc[0]) / (len(lc) - 1)
    adj = mu * np.arange(len(lc))
    detrended = spread_bps(abdi_ranaldo_s2(lh - adj, ll - adj, lc - adj))
    assert detrended > 10.0, (
        f"detrending was expected to manufacture a spread, got {detrended:.2f}")


def test_the_trustworthy_flag_refuses_a_drifting_window():
    ok = SpreadEstimate(bps=4.0, n_pairs=5000, n_sessions=100,
                        drift_bps_per_bar=0.3, window=CALM_WINDOW)
    drifty = SpreadEstimate(bps=4.0, n_pairs=5000, n_sessions=100,
                            drift_bps_per_bar=6.0, window=(1.0, 10.0))
    thin = SpreadEstimate(bps=4.0, n_pairs=10, n_sessions=2,
                          drift_bps_per_bar=0.1, window=CALM_WINDOW)
    assert ok.trustworthy
    assert not drifty.trustworthy and not thin.trustworthy


# ── why Abdi-Ranaldo and not Corwin-Schultz ─────────────────────────────────


def test_corwin_schultz_understates_when_bars_hold_few_trades():
    """Documents the choice of estimator, and the direction of the error.

    A sparsely-traded bar's observed range understates the true one, so C-S
    reads LOW. An understated spread OVERSTATES the edge — on a 3.95 bps
    margin that is the difference between shipping and not. A-R is unaffected
    at the same trade count.
    """
    h, l, c = _bars(10.0, sigma_bps=25.0, n_bars=40000, trades=20, seed=4)
    cs = float(np.mean(corwin_schultz_s(h, l))) * 1e4
    ar = spread_bps(abdi_ranaldo_s2(np.log(h), np.log(l), np.log(c)))
    assert cs < 6.0, f"C-S should understate 10 bps badly, got {cs:.2f}"
    assert ar == pytest.approx(10.0, abs=2.0), ar
    assert ar > cs + 3.0


def test_corwin_schultz_recovers_when_bars_are_dense():
    """The bias is discretisation, not a wrong formula — so it goes away with
    enough trades. Without this, the test above could be passing because the
    C-S implementation is simply broken."""
    h, l, _ = _bars(20.0, sigma_bps=25.0, n_bars=30000, trades=400, seed=6)
    cs = float(np.mean(corwin_schultz_s(h, l))) * 1e4
    assert cs == pytest.approx(20.0, abs=3.5), cs


# ── the frame-level entry point ─────────────────────────────────────────────


def _frame(true_bps, n_sessions=120, bars_per_session=90, seed=0,
           start_minute=225):
    h, l, c = _bars(true_bps, sigma_bps=12.0,
                    n_bars=n_sessions * bars_per_session, seed=seed)
    idx = []
    for d in range(n_sessions):
        day = date(2025, 1, 6) + timedelta(days=d)
        idx += [datetime.combine(day, time(9, 15))
                + timedelta(minutes=start_minute + i)
                for i in range(bars_per_session)]
    return pd.DataFrame({"High": h, "Low": l, "Close": c,
                         "Open": c}, index=pd.DatetimeIndex(idx))


def test_estimate_spread_recovers_from_a_dataframe():
    est = estimate_spread(_frame(6.0, seed=11))
    assert est is not None and est.trustworthy, est
    assert est.bps == pytest.approx(6.0, abs=1.5), est.bps
    assert est.n_sessions == 120
    assert est.cost_of_a_round_trip_bps() == est.bps


def test_pairs_never_straddle_the_overnight_boundary():
    """A cross-session pair would put the whole overnight gap into the spread
    — the exact quantity this module exists to price the cost of, which would
    make the slippage estimate scale with the edge itself."""
    df = _frame(5.0, n_sessions=60, bars_per_session=90, seed=12)
    est = estimate_spread(df)
    # 90 bars per session -> 89 within-session pairs; 60 sessions.
    assert est.n_pairs == 60 * 89, est.n_pairs


def test_bars_outside_the_window_are_ignored():
    df = _frame(5.0, seed=13, start_minute=225)
    assert estimate_spread(df, window=(0.0, 60.0)) is None
    assert estimate_spread(df, window=CALM_WINDOW) is not None


def test_degenerate_input_returns_none_rather_than_a_number():
    assert estimate_spread(None) is None
    assert estimate_spread(pd.DataFrame()) is None
    df = _frame(5.0, n_sessions=5, seed=14).drop(columns=["Close"])
    assert estimate_spread(df) is None


def test_a_nan_bar_is_skipped_not_propagated():
    df = _frame(5.0, n_sessions=40, seed=15)
    df.iloc[0, df.columns.get_loc("High")] = np.nan
    est = estimate_spread(df)
    assert est is not None and est.n_sessions == 39
    assert est.bps == pytest.approx(5.0, abs=2.0)


def test_spread_bps_reports_a_negative_mean_as_negative():
    """Clamping to zero would turn 'the assumption failed' into 'free to
    trade', which is the most expensive possible rounding."""
    assert spread_bps(np.array([-4e-6, -4e-6])) < 0
    assert math.isnan(spread_bps(np.array([])))


def test_sessions_that_drift_in_opposite_directions_are_still_flagged():
    """The drift guard must use the MEAN ABSOLUTE per-session drift.

    A signed average is ~0 for a window that trends hard every day in
    alternating directions — which is exactly what an opening window does, and
    exactly the case that breaks Abdi-Ranaldo. It would mark the most
    untrustworthy possible sample as trustworthy, and the reported spread
    would be a large negative number wearing a clean bill of health.
    """
    frames, n_sessions, per = [], 40, 90
    for d in range(n_sessions):
        h, l, c = _bars(5.0, sigma_bps=12.0, n_bars=per,
                        drift_bps=(40.0 if d % 2 == 0 else -40.0), seed=100 + d)
        day = date(2025, 3, 3) + timedelta(days=d)
        idx = [datetime.combine(day, time(9, 15)) + timedelta(minutes=225 + i)
               for i in range(per)]
        frames.append(pd.DataFrame({"High": h, "Low": l, "Close": c, "Open": c},
                                   index=pd.DatetimeIndex(idx)))
    est = estimate_spread(pd.concat(frames))
    assert est is not None and est.n_sessions == n_sessions
    assert est.drift_bps_per_bar > 5.0, (
        f"alternating drift averaged away to {est.drift_bps_per_bar:.2f}")
    assert not est.trustworthy, "a hard-drifting window passed the guard"


# ── the CLI's verdict logic ─────────────────────────────────────────────────








# ── an instrument with no quoted market ─────────────────────────────────────




# ── premium against spread ──────────────────────────────────────────────────


def _legs_frame(spec, n_sessions=60):
    """spec: {symbol: plateau_bps}. Builds exit_5/exit_10 columns around it."""
    rows = []
    rng = np.random.default_rng(0)
    for sym, plat in spec.items():
        for i in range(n_sessions):
            rows.append({"symbol": sym,
                         "session": date(2025, 1, 6) + timedelta(days=i),
                         "overnight_bps": plat * 1.8,
                         "exit_5_bps": plat + rng.normal(0, 1.0),
                         "exit_10_bps": plat + rng.normal(0, 1.0),
                         "exit_45_bps": plat * 0.5})
    return pd.DataFrame(rows)


def _spread_frame(spec):
    return pd.DataFrame([{"symbol": s, "spread_bps": v, "pairs": 5000,
                          "sessions": 120, "drift": 0.2, "trustworthy": True}
                         for s, v in spec.items()])
















# ── is a bucket's edge broad, or two names? ─────────────────────────────────






