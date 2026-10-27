"""
Effective bid-ask spread from OHLC bars — and where that is impossible.

WHY THIS EXISTS
---------------
The overnight premium survives every test this project has thrown at it:
real, survivorship-free (midcap index +12.90 bps, t 4.44), and still +7.09 bps
on the 09:16-09:25 plateau where an equity FUTURES position can actually be
exited. Net of the 3.14 bps futures round trip that is +3.95 bps a night.

Every one of those figures is mid-to-mid. Exiting a midcap futures basket at
09:16 means CROSSING A SPREAD in the most violent minute of the session, and
at 4 bps of round-trip slippage the entire edge is gone. That is the last
unknown, and it decides the strategy.

TWO PUBLISHED ESTIMATORS, ONE OF WHICH IS WRONG HERE
----------------------------------------------------
Both recover a spread from high/low/close. Simulated against a price series
with a KNOWN spread (`tests/test_spread.py` builds it), at 25 bps/bar vol:

    trades/bar   true    Corwin-Schultz (2012)    Abdi-Ranaldo (2017)
        20         0            -4.24                    +0.96
        20        10            +0.48                    +9.33
       200         0            +0.71                    -1.56
       200        10            +8.90                   +10.04

Corwin-Schultz is biased DOWNWARD by 3-10 bps whenever bars hold few trades,
because a sparsely-traded bar's observed high-low range understates the true
one. An understated spread overstates the edge — the dangerous direction — and
on a 3.95 bps margin it flips the answer. **Abdi-Ranaldo is the estimator used
here.** Corwin-Schultz is kept only so the comparison stays runnable.

A NOTE ON THE FIRST VERSION OF THAT COMPARISON: it restarted the random walk
inside every bar, so consecutive bars were independent rather than one price
path. Abdi-Ranaldo depends on the close-to-next-bar relationship and read a
flat ~35 bps regardless of the truth; Corwin-Schultz, which mostly uses
within-bar ranges, looked fine. The bug therefore pointed at the correct
estimator and away from it.

THE WINDOW THAT CANNOT BE MEASURED
----------------------------------
Abdi-Ranaldo assumes the efficient price is a martingale. At 09:16 it is not —
the gap is still resolving, and the ladder measures midcap falling 5.24 bps in
that first minute. Simulated, at 60 bps/bar vol and 20 bps/bar of drift, a
true 5 bps spread reads **-19.43**. The obvious repair, removing the window's
own mean drift, is worse: it manufactures **+27 bps at a true zero**, because
it subtracts real price variation and correlates the residuals.

So the opening spread is NOT estimable from bars, by either method, and this
module does not offer a number for it. What it offers instead is the CALM
mid-session window, where drift is ~0 and recovery is near-exact (true 3 ->
2.86, true 5 -> 4.91, true 20 -> 19.81 at 20 bps/bar vol).

**Spreads are at their WIDEST at the open and their narrowest mid-session, so
the calm-window figure is a strict LOWER BOUND on what an 09:16 exit pays.**
That makes the test one-sided but still decisive in one direction: if even the
lower bound exceeds the edge, the strategy is dead; if it does not, the
question is open and needs real quoted data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["abdi_ranaldo_s2", "corwin_schultz_s", "spread_bps", "SpreadEstimate",
           "estimate_spread", "CALM_WINDOW"]

#: Session minutes (09:15 = 0) of the calm window: 13:00-14:30. Chosen because
#: T2 measured per-minute drift here as indistinguishable from zero across all
#: entry buckets, which is the one assumption Abdi-Ranaldo cannot do without.
CALM_WINDOW = (225.0, 315.0)

_CS_K = 3.0 - 2.0 * math.sqrt(2.0)


def abdi_ranaldo_s2(log_high: np.ndarray, log_low: np.ndarray,
                    log_close: np.ndarray) -> np.ndarray:
    """Per-pair S^2 from Abdi & Ranaldo (RFS 2017), eq. (13).

        S^2 = 4 (c_t - eta_t)(c_t - eta_{t+1}),   eta = (h + l) / 2

    Returned UNAGGREGATED and unclamped. Individual values are frequently
    negative — that is sampling noise around a small positive mean, and
    clamping each one before averaging is what gives Corwin-Schultz its
    notorious 8-bps floor at a true spread of zero. Average first, then take
    the root.
    """
    eta = (log_high + log_low) / 2.0
    return 4.0 * (log_close[:-1] - eta[:-1]) * (log_close[:-1] - eta[1:])


def corwin_schultz_s(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Corwin & Schultz (JF 2012), for comparison only — see the module
    docstring for why it is not the estimator used."""
    lo_hl = np.log(high / low) ** 2
    beta = lo_hl[:-1] + lo_hl[1:]
    gamma = np.log(np.maximum(high[:-1], high[1:])
                   / np.minimum(low[:-1], low[1:])) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_K - np.sqrt(gamma / _CS_K)
    return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def spread_bps(s2: np.ndarray) -> float:
    """Mean of S^2, then the root, in basis points.

    A negative mean is returned as a NEGATIVE spread rather than clamped to
    zero. A negative reading is not "no spread" — it is the estimator telling
    you the assumption failed or the sample is too small, and hiding that
    behind a 0.0 turns a diagnostic into a fake measurement.
    """
    v = np.asarray(s2, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    m = float(v.mean())
    return (math.sqrt(m) if m > 0 else -math.sqrt(-m)) * 1e4


@dataclass(frozen=True)
class SpreadEstimate:
    """A spread, and everything needed to distrust it."""

    bps: float
    n_pairs: int
    n_sessions: int
    drift_bps_per_bar: float
    window: tuple

    @property
    def trustworthy(self) -> bool:
        """Drift is the assumption that fails first, and it fails hard: at 20
        bps/bar a true 5 bps spread reads -19. Anything above ~2 bps/bar of
        drift makes the number meaningless."""
        return (abs(self.drift_bps_per_bar) < 2.0 and self.n_pairs >= 500
                and self.bps == self.bps)

    def cost_of_a_round_trip_bps(self) -> float:
        """Entering at 15:25 and exiting at 09:16 crosses the spread twice, at
        half the spread each time — one full spread in total."""
        return self.bps


def estimate_spread(df: pd.DataFrame, window: tuple = CALM_WINDOW,
                    session_open_minute: float = 9 * 60 + 15
                    ) -> Optional[SpreadEstimate]:
    """Abdi-Ranaldo over `window`, pooled across sessions.

    Pairs are formed WITHIN a session only. A pair straddling the overnight
    boundary would put the whole gap into the estimate — the very quantity
    this is meant to price the cost of.
    """
    if df is None or len(df) == 0:
        return None
    need = {"High", "Low", "Close"}
    if not need.issubset(df.columns):
        return None
    idx = pd.DatetimeIndex(df.index)
    minute = (idx.hour * 60 + idx.minute).to_numpy(dtype=float) - session_open_minute
    keep = (minute >= window[0]) & (minute <= window[1])
    if not keep.any():
        return None
    sub = df[keep]
    dates = np.array([t.date() for t in pd.DatetimeIndex(sub.index)])

    chunks, drifts = [], []
    for sess in pd.unique(dates):
        m = dates == sess
        h = np.log(sub["High"].to_numpy(dtype=float)[m])
        l = np.log(sub["Low"].to_numpy(dtype=float)[m])
        c = np.log(sub["Close"].to_numpy(dtype=float)[m])
        if c.size < 3 or not (np.isfinite(h).all() and np.isfinite(l).all()
                              and np.isfinite(c).all()):
            continue
        chunks.append(abdi_ranaldo_s2(h, l, c))
        drifts.append((c[-1] - c[0]) / (c.size - 1) * 1e4)
    if not chunks:
        return None
    allv = np.concatenate(chunks)
    return SpreadEstimate(
        bps=spread_bps(allv), n_pairs=int(allv.size), n_sessions=len(chunks),
        # MEAN ABSOLUTE drift: sessions drift up and down, and averaging the
        # signed value would report ~0 for a window that trends hard every day
        # in alternating directions — exactly the case that breaks A-R.
        drift_bps_per_bar=float(np.mean(np.abs(drifts))), window=tuple(window))
