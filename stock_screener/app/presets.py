"""Column presets for the chooser."""

from __future__ import annotations

from .columns import COLUMNS

_ID = ["ticker", "name", "sector", "price", "change_1d", "market_cap"]

PRESETS: dict[str, list[str]] = {
    "Overview": [c.key for c in COLUMNS if c.default],
    "Momentum & Trend": _ID
    + [
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_1y",
        "mom_20",
        "mom_60",
        "mom_120",
        "trend_score",
        "sma50_dist",
        "sma200_dist",
        "rsi_14",
        "macd_hist",
        "adx_14",
        "volume_trend",
        "rel_volume",
        "obv_slope_20",
        "momentum_score",
    ],
    "Volatility & Options": _ID
    + [
        "rv_20",
        "rv_60",
        "iv_30",
        "iv_60",
        "iv_90",
        "iv_rank",
        "iv_percentile",
        "iv_rv_ratio",
        "vol_risk_premium",
        "term_slope",
        "skew_25d",
        "put_tail_10d",
        "expected_move",
        "atm_straddle_move_30",
        "implied_carry",
        "open_interest",
        "pc_oi_ratio",
        "fit_rmse",
    ],
    "Downside Risk": _ID
    + [
        "iv_30",
        "expected_move",
        "p_loss_5",
        "p_loss_10",
        "p_loss_20",
        "var_5",
        "es_5",
        "rn_semi_dev",
        "rn_skew",
        "rn_kurtosis",
        "tail_asymmetry",
        "p_gain_10",
        "max_dd_1y",
        "beta_1y",
        "risk_score",
    ],
    "Fundamentals": _ID
    + [
        "pe",
        "pb",
        "ps",
        "p_fcf",
        "ev_ebitda",
        "ev_sales",
        "roe",
        "roa",
        "debt_to_equity",
        "current_ratio",
        "dividend_yield",
        "fcf_yield",
        "revenue_growth",
        "gross_margin",
        "net_margin",
        "quality_score",
        "value_score",
    ],
    "All": [c.key for c in COLUMNS],
}
