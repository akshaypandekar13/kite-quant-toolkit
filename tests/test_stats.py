"""
Session-clustered statistics
============================
Every t-stat this project has acted on was pooled across rows, but ~99 symbols
move together on a given day, so the rows inside a session are not independent
draws. The design effect corrects for that, and these tests pin it at both
extremes where the right answer is known in closed form:

    rows perfectly correlated within a session  ->  design effect = sqrt(m)
    rows independent                            ->  design effect = 1

Getting either end wrong silently rescales every significance claim downstream,
in a direction nobody would notice.
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._stats import (  # noqa: E402
    CHARGE_BPS, SessionStats, format_stats, session_stats, stats_row,
)


def _frame(values_by_session):
    rows = []
    for i, vals in enumerate(values_by_session):
        d = date(2026, 1, 5) + timedelta(days=i)
        rows.extend({"session": d, "r": float(v)} for v in vals)
    return pd.DataFrame(rows)


# ── the two extremes, where the answer is closed-form ───────────────────────


def test_perfect_within_session_correlation_gives_design_effect_sqrt_m():
    """Every row in a session identical: the session IS the observation.

    With m rows per session and no within-session variance, the effective
    sample size is the number of sessions, so the pooled SE understates by
    exactly sqrt(m).
    """
    m = 25
    df = _frame([[v] * m for v in (0.5, -0.3, 0.9, -0.1, 0.4, 0.2)])
    s = session_stats(df, "r")
    assert s is not None
    assert s.icc == pytest.approx(1.0, abs=1e-6)
    assert s.design_effect == pytest.approx(math.sqrt(m), rel=1e-6)
    assert s.se_clustered == pytest.approx(s.se_pooled * math.sqrt(m), rel=1e-6)


def test_independent_rows_give_a_design_effect_of_one():
    """No session structure: the pooled error bar was right all along."""
    rng = np.random.default_rng(4)
    df = _frame([rng.normal(0, 1, 120) for _ in range(30)])
    s = session_stats(df, "r")
    assert s.icc < 0.02, s.icc
    assert s.design_effect == pytest.approx(1.0, abs=0.12)


def test_a_negative_icc_is_clamped_instead_of_shrinking_the_error_bar():
    """Sessions MORE alike than chance allows produce a negative raw ICC.

    Constructed, not sampled: every session alternates +1/-1, so all session
    means are exactly 0 while the within-session variance is 1. That makes
    MSB < MSW and the raw ICC negative. Left unclamped it would make the
    corrected error bar TIGHTER than the naive one — the exact opposite of the
    correction's purpose, and invisible because it only ever nudges t upward.
    """
    df = _frame([[1.0, -1.0] * 10 for _ in range(12)])
    s = session_stats(df, "r")
    assert s.icc == 0.0, "a negative ICC survived into the design effect"
    assert s.design_effect == pytest.approx(1.0)
    assert s.se_clustered >= s.se_pooled - 1e-12


def test_the_clustered_error_is_never_smaller_than_the_pooled_one():
    rng = np.random.default_rng(9)
    for seed_shift in range(6):
        df = _frame([rng.normal(0, 1, 8) for _ in range(12)])
        s = session_stats(df, "r")
        assert s.design_effect >= 1.0 - 1e-12
        assert s.se_clustered >= s.se_pooled - 1e-12, seed_shift


# ── the mean itself, and the unbalanced-design correction ───────────────────


def test_the_mean_is_the_plain_pooled_mean():
    """Clustering changes the error bar, never the estimate."""
    df = _frame([[1.0, 2.0, 3.0], [4.0] * 10])
    s = session_stats(df, "r")
    assert s.mean == pytest.approx(df["r"].mean())
    assert s.n == 13 and s.n_sessions == 2


def test_unbalanced_sessions_use_m0_not_the_mean_group_size():
    """One huge session and several small ones.

    m0 = (N - sum(n_i^2)/N) / (k-1) is below the arithmetic mean group size
    whenever the sizes are uneven; using the plain mean inflates ICC's leverage
    and overstates the correction.
    """
    df = _frame([[0.5] * 200] + [[0.5] * 2 for _ in range(5)])
    sizes = np.array([200.0] + [2.0] * 5)
    n, k = sizes.sum(), len(sizes)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    s = session_stats(df, "r")
    assert m0 < sizes.mean(), "fixture does not exercise the unbalanced case"
    # Zero variance everywhere -> ICC clamps to 0 and the effect is 1.
    assert s.design_effect == pytest.approx(1.0)
    # Now give the sessions different levels so ICC is genuinely 1.
    df2 = _frame([[0.5] * 200] + [[float(i)] * 2 for i in range(1, 6)])
    s2 = session_stats(df2, "r")
    assert s2.design_effect == pytest.approx(math.sqrt(m0), rel=0.02), \
        f"design {s2.design_effect:.3f} vs sqrt(m0) {math.sqrt(m0):.3f}"


# ── degenerate inputs must degrade honestly ─────────────────────────────────


def test_one_session_reports_no_clustered_error():
    """One day is one observation of the thing that varies. It must not come
    back with a confident-looking t."""
    s = session_stats(_frame([[0.1, 0.2, 0.3, 0.4]]), "r")
    assert s.n_sessions == 1
    assert math.isnan(s.se_clustered) and math.isnan(s.t_clustered)
    assert not math.isnan(s.t_pooled), "the pooled figure is still computable"


def test_no_rows_returns_none_rather_than_a_number():
    assert session_stats(pd.DataFrame({"session": [], "r": []}), "r") is None
    assert session_stats(None, "r") is None
    assert session_stats(_frame([[1.0]]), "missing_column") is None


def test_all_nan_values_return_none():
    df = _frame([[1.0, 2.0], [3.0, 4.0]])
    df["r"] = np.nan
    assert session_stats(df, "r") is None


def test_zero_variance_does_not_divide_by_zero():
    s = session_stats(_frame([[2.0] * 5, [2.0] * 5]), "r")
    assert s.mean == pytest.approx(2.0)
    assert s.sd == 0.0
    assert math.isnan(s.t_pooled) and math.isnan(s.t_clustered)


def test_nan_rows_are_dropped_not_propagated():
    df = _frame([[1.0, 2.0, np.nan], [3.0, 4.0]])
    s = session_stats(df, "r")
    assert s.n == 4 and s.mean == pytest.approx(2.5)


# ── the verdict helpers ─────────────────────────────────────────────────────


def test_significance_uses_the_clustered_t_not_the_pooled_one():
    """The whole point. A finding that is significant only on the pooled error
    is not significant."""
    m = 30
    df = _frame([[0.30] * m, [0.28] * m, [0.32] * m, [0.29] * m])
    s = session_stats(df, "r")
    assert abs(s.t_pooled) > abs(s.t_clustered)
    assert s.significant == (abs(s.t_clustered) > 2.0)


def test_beats_charges_is_a_separate_question_from_significance():
    """A drift can be real and still untradeable. 1 bp measured over a million
    rows is beautifully significant and loses money every round trip."""
    df = _frame([[1.0] * 50, [1.1] * 50, [0.9] * 50, [1.05] * 50])
    s = session_stats(df, "r")          # mean ~1 bp
    assert s.significant, "fixture should be statistically clear"
    assert not s.beats_charges, f"1 bp must not clear the {CHARGE_BPS} bp line"
    big = _frame([[20.0] * 50, [22.0] * 50, [18.0] * 50, [21.0] * 50])
    assert session_stats(big, "r").beats_charges


def test_format_and_row_report_the_clustered_figure_first():
    s = session_stats(_frame([[0.5] * 20, [0.3] * 20, [0.7] * 20]), "r")
    txt = format_stats(s, unit="bps")
    assert "clustered" in txt and "inflated" in txt and "bps" in txt
    row = stats_row("cohort", s)
    assert row["design_effect"] >= 1.0 and row["sessions"] == 3
    assert stats_row("empty", None)["n"] == 0
    assert format_stats(None) == "no data"


# ── against the real journal ────────────────────────────────────────────────


def test_it_reproduces_the_measured_journal_clustering():
    """Pins the number quoted in the module docstring and the project notes.

    rejection_postmortem.csv: ~1,136 rows over 10 sessions, ICC ~0.010,
    design effect ~1.46. If this drifts, the "a reported t=2.0 is really 1.4"
    claim drifts with it and every doc repeating it becomes wrong.
    """
    p = Path(__file__).resolve().parent.parent / "rejection_postmortem.csv"
    if not p.exists():
        pytest.skip("journal not present in this checkout")
    d = pd.read_csv(p, low_memory=False)
    d["session"] = pd.to_datetime(d["rejected_at"], errors="coerce").dt.date
    d["r"] = pd.to_numeric(d["pnl_r_static"], errors="coerce")
    s = session_stats(d.dropna(subset=["session", "r"]), "r")
    assert s.n_sessions == 10
    assert 0.0 <= s.icc < 0.05, s.icc
    assert 1.2 < s.design_effect < 1.8, s.design_effect


def test_a_missing_session_column_is_loud_not_silently_pooled():
    """Falling back to a pooled error bar here would hand the caller exactly
    the flattering t-stat this module exists to prevent, and the result would
    be indistinguishable from a correct one."""
    df = pd.DataFrame({"r": [1.0, 2.0, 3.0], "day": [1, 1, 2]})
    with pytest.raises(KeyError, match="session"):
        session_stats(df, "r")
    # Naming the column explicitly works.
    s = session_stats(df, "r", session_col="day")
    assert s is not None and s.n_sessions == 2


@pytest.mark.parametrize("true_icc", [0.0, 0.005, 0.045])
def test_the_estimator_works_at_the_group_sizes_the_reports_use(true_icc):
    """T2 reported ICC = 0.0000 for all seven buckets while T4, on overlapping
    rows, reported 0.0445. That asymmetry looked like a bug, so it is pinned
    here: with ~1,090 rows per session — T2's actual shape — a KNOWN ICC must
    come back recovered. It does, so T2's near-zero reading is a measurement of
    the data rather than a failure of the estimator.
    """
    rng = np.random.default_rng(0)
    per = 1090
    between, within = np.sqrt(true_icc), np.sqrt(1.0 - true_icc)
    frames = []
    for sess in range(200):
        shock = rng.normal(0.0, between)
        frames.append(pd.DataFrame({
            "session": sess,
            "r": shock + rng.normal(0.0, within, per)}))
    s = session_stats(pd.concat(frames, ignore_index=True), "r")
    assert s.icc == pytest.approx(true_icc, abs=0.006), s.icc
    expected = math.sqrt(1.0 + (per - 1) * true_icc)
    assert s.design_effect == pytest.approx(expected, rel=0.06)
