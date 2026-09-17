"""Per-ticker computation and universe orchestration.

``build_screener`` fetches (or reads from cache) everything for each ticker,
computes the metrics, persists intermediate artefacts for the detail panel and
returns the screener table as a DataFrame with one row per ticker.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..config import Settings
from ..data.cache import DataCache
from ..data.history import IVHistory
from ..data.provider import MarketDataProvider, Ratios, TickerBundle
from .chain import ChainAnalysis, analyse_chain
from .fundamentals import dividend_yield_from_history, fundamental_metrics
from .qlpricing import MarketEnv
from .risk_neutral import risk_metrics, risk_neutral_density
from .scoring import add_scores
from .technicals import compute_technicals

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------------------


def fetch_bundle(
    provider: MarketDataProvider,
    ticker: str,
    as_of: dt.date,
    settings: Settings,
    cache: DataCache | None,
    ratios: Ratios | None,
) -> TickerBundle:
    warnings: list[str] = []
    bars = cache.get_frame(as_of, "bars", ticker) if cache else None
    if bars is None:
        start = as_of - dt.timedelta(days=int(settings.history_days * 1.5) + 30)
        bars = provider.get_bars(ticker, start, as_of)
        if cache is not None and not bars.empty:
            cache.put_frame(as_of, "bars", ticker, bars)
    chain = cache.get_frame(as_of, "chain", ticker) if cache else None
    if chain is None:
        chain = provider.get_option_chain(ticker)
        if cache is not None and not chain.empty:
            cache.put_frame(as_of, "chain", ticker, chain)
    meta = cache.get_meta(as_of, ticker) if cache else None
    if meta is None:
        details = provider.get_details(ticker)
        dividends = provider.get_dividends(ticker, as_of - dt.timedelta(days=400))
        ratios = ratios or Ratios(ticker=ticker)
        if cache is not None:
            cache.put_meta(as_of, ticker, details, ratios, dividends)
    else:
        details, cached_ratios, dividends = meta
        ratios = ratios or cached_ratios
    if bars.empty:
        warnings.append("no price history")
    if chain.empty:
        warnings.append("no option chain")
    return TickerBundle(
        ticker, as_of, bars, chain, details, dividends, ratios or Ratios(ticker=ticker), warnings
    )


# --------------------------------------------------------------------------------------
# computation
# --------------------------------------------------------------------------------------


def compute_ticker(
    bundle: TickerBundle,
    settings: Settings,
    bench: pd.DataFrame | None = None,
    cache: DataCache | None = None,
) -> tuple[dict, ChainAnalysis | None]:
    t0 = time.perf_counter()
    row: dict = {"ticker": bundle.ticker, "as_of": bundle.as_of.isoformat()}
    row.update(compute_technicals(bundle.bars, bench))
    spot = row.get("price")
    if not bundle.chain.empty and bundle.chain["underlying_price"].notna().any():
        chain_spot = float(bundle.chain["underlying_price"].dropna().iloc[-1])
        if spot and abs(chain_spot / spot - 1) > 0.01:
            bundle.warnings.append(f"chain spot {chain_spot:.2f} differs from bar close {spot:.2f}")
        spot = spot or chain_spot
        row["chain_spot"] = chain_spot
    row.update(fundamental_metrics(bundle.ratios, bundle.details, spot))
    analysis = None
    if spot and not bundle.chain.empty:
        env = MarketEnv(bundle.as_of, settings.risk_free_rate)
        q = dividend_yield_from_history(bundle.dividends, spot, bundle.as_of)
        if q == 0 and bundle.ratios and bundle.ratios.dividend_yield:
            q = float(bundle.ratios.dividend_yield)
        row["div_yield_used"] = q
        analysis = analyse_chain(
            bundle.chain,
            env,
            spot,
            q,
            tuple(settings.horizons_days),
            settings.min_days_to_expiry,
            settings.max_days_to_expiry,
            settings.american_iv,
        )
        row.update(analysis.metrics)
        bundle.warnings.extend(analysis.warnings)
        if analysis.surface is not None:
            h = settings.primary_horizon_days
            t = h / 365.0
            from .chain import interp_forward

            fwd = interp_forward(analysis.fits, spot, t, settings.risk_free_rate, q)
            try:
                rnd = risk_neutral_density(analysis.surface, spot, fwd, t, settings.risk_free_rate)
                row.update(risk_metrics(rnd, settings.loss_thresholds, settings.gain_thresholds))
                if cache is not None:
                    cache.put_density(
                        bundle.as_of,
                        bundle.ticker,
                        pd.DataFrame(
                            {"strike": rnd.strikes, "density": rnd.density, "cdf": rnd.cdf}
                        ),
                    )
            except Exception as exc:  # pragma: no cover - defensive
                bundle.warnings.append(f"density failed: {exc}")
        if cache is not None and not analysis.cleaned.empty:
            cache.put_chain_summary(bundle.as_of, bundle.ticker, analysis.cleaned)
            cache.put_frame(bundle.as_of, "fits", bundle.ticker, fits_frame(analysis))
    # relations between option-implied and realised quantities
    iv30 = row.get(f"iv_{settings.primary_horizon_days}")
    if iv30 and row.get("rv_20"):
        row["iv_rv_ratio"] = iv30 / row["rv_20"]
        row["vol_risk_premium"] = iv30 - row["rv_20"]
    if row.get("expected_move") is not None and row.get("atr_pct"):
        row["expected_move_vs_atr"] = row["expected_move"] / (
            row["atr_pct"] * math.sqrt(settings.primary_horizon_days * 252 / 365)
        )
    row["n_warnings"] = len(bundle.warnings)
    row["warnings"] = "; ".join(bundle.warnings)
    row["compute_ms"] = round((time.perf_counter() - t0) * 1000)
    return row, analysis


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def build_screener(
    provider: MarketDataProvider,
    settings: Settings,
    as_of: dt.date,
    cache: DataCache | None = None,
    tickers: list[str] | None = None,
    progress=None,
) -> pd.DataFrame:
    tickers = tickers or settings.universe
    ratios = {}
    if settings.has_fundamentals or provider.name == "synthetic":
        try:
            ratios = provider.get_ratios(tickers)
        except Exception as exc:  # pragma: no cover
            log.warning("ratios unavailable: %s", exc)
    bench = None
    if settings.benchmark:
        try:
            start = as_of - dt.timedelta(days=int(settings.history_days * 1.5) + 30)
            bench = provider.get_bars(settings.benchmark, start, as_of)
        except Exception as exc:  # pragma: no cover
            log.warning("benchmark unavailable: %s", exc)
    history = IVHistory(settings.data_dir / "iv_history.parquet") if cache is not None else None

    # Phase 1: I/O in parallel. Phase 2: analytics serially — QuantLib's Python
    # bindings share global state (evaluation date, observers) and are not thread-safe.
    bundles: dict[str, TickerBundle] = {}
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, settings.max_workers)) as ex:
        futures = {
            ex.submit(fetch_bundle, provider, t, as_of, settings, cache, ratios.get(t)): t
            for t in tickers
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                bundles[t] = fut.result()
            except Exception as exc:
                log.exception("fetch for %s failed", t)
                rows.append(
                    {
                        "ticker": t,
                        "as_of": as_of.isoformat(),
                        "warnings": f"fetch failed: {exc}",
                        "n_warnings": 1,
                    }
                )
    for i, t in enumerate(tickers, 1):
        if t in bundles:
            try:
                rows.append(compute_ticker(bundles[t], settings, bench, cache)[0])
            except Exception as exc:
                log.exception("compute for %s failed", t)
                rows.append(
                    {
                        "ticker": t,
                        "as_of": as_of.isoformat(),
                        "warnings": f"compute failed: {exc}",
                        "n_warnings": 1,
                    }
                )
        if progress:
            progress(i, len(tickers), t)
    df = pd.DataFrame(rows)
    df = df.sort_values("ticker").reset_index(drop=True)
    if history is not None:
        hist_rows = df[["ticker"]].copy()
        hist_rows["date"] = pd.Timestamp(as_of)
        for c in ("iv30", "iv60", "iv90"):
            src = f"iv_{c[2:]}"
            hist_rows[c] = df[src] if src in df else np.nan
        hist_rows["rv20"] = df["rv_20"] if "rv_20" in df else np.nan
        hist_rows["spot"] = df["price"] if "price" in df else np.nan
        # rank/percentile use history strictly before as_of, then append today's observation
        ranks = [
            history.rank_and_percentile(t, iv, as_of)
            for t, iv in zip(df["ticker"], hist_rows["iv30"], strict=True)
        ]
        df["iv_rank"] = [r[0] for r in ranks]
        df["iv_percentile"] = [r[1] for r in ranks]
        history.append(hist_rows)
    df = add_scores(df)
    return df


def fits_frame(analysis: ChainAnalysis) -> pd.DataFrame:
    """Serialisable representation of the fitted smiles (for the detail panel)."""
    rows = []
    for f in analysis.fits:
        a, b, c = f.coef
        rows.append(
            {
                "expiry": pd.Timestamp(f.expiry),
                "t": f.t,
                "forward": f.forward,
                "a": a,
                "b": b,
                "c": c,
                "k_min": f.k_min,
                "k_max": f.k_max,
                "n": f.n,
                "rmse": f.rmse,
                "atm_iv": f.atm_iv,
                "forward_source": f.forward_source,
            }
        )
    return pd.DataFrame(rows)


def default_as_of(provider_name: str, today: dt.date | None = None) -> dt.date:
    """Most recent NYSE trading day (a refresh after the close uses today)."""
    import QuantLib as ql

    today = today or dt.date.today()
    cal = ql.UnitedStates(ql.UnitedStates.NYSE)
    d = ql.Date(today.day, today.month, today.year)
    while not cal.isBusinessDay(d):
        d = d - 1
    return dt.date(d.year(), d.month(), d.dayOfMonth())
