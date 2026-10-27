"""
Canonical symbol form
=====================
One join key for every CSV the bot writes.

WHY THIS EXISTS
---------------
The seven output files disagreed on how to spell a symbol:

    equity_curve.csv          Symbol      RPOWER          (stock_name)
    intratrade_mfe.csv        Symbol      RPOWER.NS       (raw watchlist key)
    early_exit_audit.csv      Symbol      RPOWER.NS
    expiry_size_audit.csv     Symbol      VBL.NS
    regime_overrides.csv      Symbol      <raw>
    rejected_signals.csv      Symbol      APOLLOHOSP.NS   + Stock_Name APOLLOHOSP
    rejection_postmortem.csv  symbol      APOLLOHOSP.NS

So the natural join — trade journal against intra-trade snapshots — matched
**0 of 838 rows** and returned a silently empty result rather than an error.
Any analysis that skipped a sanity check on the join rate would have drawn
conclusions from an empty frame. That is exactly the class of bug that is
invisible until someone prints the match rate.

CANONICAL FORM = the bare NSE trading symbol, uppercase, no exchange suffix
(RPOWER, not RPOWER.NS). The `.NS`/`.BO` suffixes are a Yahoo Finance artifact
and carry no information for a Zerodha-only bot; the broker itself uses the
bare form (see APITrader._strip_yahoo_suffix / ZerodhaFetcher.strip_yahoo_suffix).

Ambiguity note: a numeric instrument token (e.g. the Nifty "256265") is passed
through unchanged — it is already canonical and must not be uppercased or
stripped.
"""

from __future__ import annotations

_SUFFIXES = ('.NS', '.BO', '.NSE', '.BSE')


def canonical_symbol(symbol) -> str:
    """Return the canonical join key for a symbol.

    Idempotent — safe to apply to already-canonical values, which is what
    makes it usable both at every write site and in a one-off backfill.

        canonical_symbol('RPOWER.NS')  -> 'RPOWER'
        canonical_symbol('rpower.ns')  -> 'RPOWER'
        canonical_symbol('RPOWER')     -> 'RPOWER'
        canonical_symbol('M&M.NS')     -> 'M&M'
        canonical_symbol('256265')     -> '256265'   (instrument token)
        canonical_symbol(None)         -> ''
    """
    if symbol is None:
        return ''
    s = str(symbol).strip()
    if not s:
        return ''
    if s.isdigit():          # instrument token — already canonical
        return s
    upper = s.upper()
    for suffix in _SUFFIXES:
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper
