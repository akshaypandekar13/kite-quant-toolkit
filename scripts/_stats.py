"""
Session-clustered statistics for the offline analysis scripts.

WHY THIS EXISTS
---------------
Three separate copies of

    se = values.std(ddof=1) / sqrt(len(values));  t = mean / se

live in `validate_excursion.py` and `replay_backtest.py`, and none of them
accounts for the fact that ~99 symbols all move together on a given day. A
session is the independent unit; the rows inside it are not independent draws,
so a pooled standard error is too small and every t-stat built on one is too
large.

HOW MUCH TOO LARGE — MEASURED, NOT ASSUMED
------------------------------------------
On `rejection_postmortem.csv` (1,136 rows, 10 sessions, ~114 signals/session)
the one-way ANOVA intraclass correlation is ICC = 0.0101, giving a design
effect of

    sqrt(1 + (m0 - 1) * ICC) = 1.46

So a reported |t| = 2.0 is really about 1.4 — real, and much smaller than the
order-of-magnitude inflation a naive "sessions are the only unit" argument
predicts. Both extremes are wrong: rows are not independent, but neither are
they perfectly correlated within a day. The design effect interpolates between
them, and it is cheap to compute, so it should be reported rather than argued
about.

This module deliberately does NOT live in `_analysis_common.py`, whose own
docstring scopes it to "CSV-parsing / symbol / history-sizing conventions".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["SessionStats", "session_stats", "format_stats", "CHARGE_BPS"]

#: Round-trip charges on a Rs225K Zerodha MIS equity position, in basis points
#: of notional. Rs127: STT 44%, brokerage 32%, the rest exchange/GST/stamp/SEBI.
#: The measured friction identity's numerator (0.0594% of notional) is the same
#: quantity, so 5.94 bps and this constant are two spellings of one number.
#: Drift smaller than this is not tradeable no matter how significant it is.
CHARGE_BPS = 5.94


@dataclass(frozen=True)
class SessionStats:
    """A mean, with both the honest and the flattering error bar attached."""

    mean: float
    n: int
    n_sessions: int
    sd: float
    se_pooled: float
    se_clustered: float
    t_pooled: float
    t_clustered: float
    icc: float
    design_effect: float

    @property
    def significant(self) -> bool:
        """|t| > 2 on the CLUSTERED error. The pooled one does not get a vote."""
        return bool(abs(self.t_clustered) > 2.0)

    @property
    def beats_charges(self) -> bool:
        """True when the mean clears round-trip friction, in bps.

        Only meaningful when `mean` is expressed in basis points. A drift that
        is statistically real and smaller than `CHARGE_BPS` is a measurement,
        not a trade.
        """
        return bool(self.mean > CHARGE_BPS)


def session_stats(df: pd.DataFrame, value_col: str,
                  session_col: str = "session") -> Optional[SessionStats]:
    """Mean of `value_col` with a session-clustered standard error.

    Returns None when there is nothing to measure (no rows, or every value
    NaN). A single session yields a clustered SE of NaN rather than a
    confident-looking number: one day is one observation of the thing that
    actually varies.
    """
    if df is None or len(df) == 0 or value_col not in df:
        return None
    if session_col not in df:
        # Deliberately loud. Quietly falling back to a pooled error bar here
        # would hand the caller exactly the flattering t-stat this module
        # exists to stop, and it would look identical to a correct one.
        raise KeyError(
            f"session_stats needs a {session_col!r} column to cluster on; "
            f"got {list(df.columns)}. Pass one, or rename the date column.")
    sub = df[[value_col, session_col]].dropna()
    n = len(sub)
    if n == 0:
        return None

    x = sub[value_col].to_numpy(dtype=float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se_pooled = sd / math.sqrt(n) if n > 1 else float("nan")

    groups = sub.groupby(session_col)[value_col]
    sizes = groups.size().to_numpy(dtype=float)
    k = len(sizes)

    icc, m0 = 0.0, 1.0
    if k > 1 and n > k and sd > 0:
        means = groups.mean().to_numpy(dtype=float)
        # One-way random-effects ANOVA. m0 is the unbalanced-design group size,
        # NOT the arithmetic mean of the group sizes — with uneven sessions the
        # two differ and the plain mean biases ICC upward.
        msb = float((sizes * (means - mean) ** 2).sum() / (k - 1))
        within = sub[value_col].to_numpy(dtype=float) - \
            sub.groupby(session_col)[value_col].transform("mean").to_numpy(dtype=float)
        msw = float((within ** 2).sum() / (n - k))
        m0 = float((n - (sizes ** 2).sum() / n) / (k - 1))
        denom = msb + (m0 - 1.0) * msw
        if denom > 0 and m0 > 1:
            # A negative ICC is a sampling artefact, not anti-correlation
            # between sessions; clamp so the design effect cannot fall below 1
            # and quietly make the error bar smaller than the pooled one.
            icc = max((msb - msw) / denom, 0.0)

    design = math.sqrt(max(1.0 + (m0 - 1.0) * icc, 1.0))
    se_clustered = se_pooled * design if se_pooled == se_pooled else float("nan")
    if k < 2:
        se_clustered = float("nan")

    def _t(se: float) -> float:
        return mean / se if se and se == se and se > 0 else float("nan")

    return SessionStats(
        mean=mean, n=n, n_sessions=k, sd=sd,
        se_pooled=se_pooled, se_clustered=se_clustered,
        t_pooled=_t(se_pooled), t_clustered=_t(se_clustered),
        icc=icc, design_effect=design,
    )


def format_stats(s: Optional[SessionStats], unit: str = "") -> str:
    """One line, with the clustered t first because it is the real one."""
    if s is None:
        return "no data"
    u = f" {unit}" if unit else ""
    return (f"mean {s.mean:+.4f}{u}  n={s.n:,} over {s.n_sessions} sessions  "
            f"t={s.t_clustered:+.2f} (clustered; pooled would read "
            f"{s.t_pooled:+.2f}, inflated {s.design_effect:.2f}x)")


def stats_row(label: str, s: Optional[SessionStats]) -> dict:
    """The same numbers as a dict, for assembling a DataFrame report."""
    if s is None:
        return {"cohort": label, "n": 0}
    return {
        "cohort": label, "n": s.n, "sessions": s.n_sessions,
        "mean": round(s.mean, 4),
        "t_clustered": round(s.t_clustered, 2) if s.t_clustered == s.t_clustered else np.nan,
        "t_pooled": round(s.t_pooled, 2) if s.t_pooled == s.t_pooled else np.nan,
        "design_effect": round(s.design_effect, 2),
        "icc": round(s.icc, 4),
    }
