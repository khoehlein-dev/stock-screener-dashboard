"""Fundamental columns from Massive ratios / statements and dividend history."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..data.provider import Ratios, TickerDetails


def dividend_yield_from_history(dividends: pd.DataFrame, spot: float, as_of: dt.date) -> float:
    """Trailing 12-month cash dividends divided by spot (continuous-yield proxy)."""
    if dividends is None or dividends.empty or not spot:
        return 0.0
    start = pd.Timestamp(as_of) - pd.Timedelta(days=365)
    d = dividends[(dividends["ex_date"] > start) & (dividends["ex_date"] <= pd.Timestamp(as_of))]
    total = float(d["cash_amount"].fillna(0).sum())
    return max(0.0, min(total / spot, 0.25))


def fundamental_metrics(ratios: Ratios | None, details: TickerDetails, spot: float) -> dict:
    out: dict = {
        "name": details.name,
        "sector": details.sector,
        "market_cap": details.market_cap,
    }
    if ratios is None:
        out["has_fundamentals"] = False
        return out
    out["has_fundamentals"] = not ratios.is_empty()
    if ratios.market_cap:
        out["market_cap"] = ratios.market_cap
    out.update(
        {
            "pe": ratios.price_to_earnings,
            "pb": ratios.price_to_book,
            "ps": ratios.price_to_sales,
            "p_fcf": ratios.price_to_free_cash_flow,
            "ev_ebitda": ratios.ev_to_ebitda,
            "ev_sales": ratios.ev_to_sales,
            "roe": ratios.return_on_equity,
            "roa": ratios.return_on_assets,
            "debt_to_equity": ratios.debt_to_equity,
            "current_ratio": ratios.current,
            "dividend_yield": ratios.dividend_yield,
            "eps": ratios.earnings_per_share,
            "revenue_growth": ratios.revenue_growth_yoy,
            "gross_margin": ratios.gross_margin,
            "net_margin": ratios.net_margin,
        }
    )
    if ratios.free_cash_flow and out.get("market_cap"):
        out["fcf_yield"] = ratios.free_cash_flow / out["market_cap"]
    if out.get("pe") and out["pe"] > 0:
        out["earnings_yield"] = 1 / out["pe"]
    return out
