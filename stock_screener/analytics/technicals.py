"""Technical indicators from daily OHLCV bars (pure pandas/numpy).

All functions accept a bars DataFrame indexed by date with columns
``open, high, low, close, volume`` and return floats (``None`` when the history
is too short).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _last(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def pct_return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return float(close.iloc[-1] / close.iloc[-1 - periods] - 1)


def ytd_return(close: pd.Series) -> float | None:
    year = close.index[-1].year
    prev = close[close.index.year < year]
    if prev.empty:
        return None
    return float(close.iloc[-1] / prev.iloc[-1] - 1)


def max_drawdown(close: pd.Series, window: int = 252) -> float | None:
    s = close.iloc[-window:]
    if len(s) < 5:
        return None
    peak = s.cummax()
    return float((s / peak - 1).min())


def realized_vol(close: pd.Series, window: int) -> float | None:
    lr = np.log(close).diff().dropna().iloc[-window:]
    if len(lr) < max(5, window // 2):
        return None
    return float(lr.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder RSI (exponential smoothing with alpha = 1/period)."""
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna().to_numpy()
    up = np.clip(delta, 0, None)
    down = np.clip(-delta, 0, None)
    # Wilder seeding: simple average of the first `period` changes, then recursive smoothing
    last_up, last_down = up[:period].mean(), down[:period].mean()
    for u, d in zip(up[period:], down[period:], strict=True):
        last_up = (last_up * (period - 1) + u) / period
        last_down = (last_down * (period - 1) + d) / period
    if last_down == 0:
        return 100.0
    rs = last_up / last_down
    return float(100 - 100 / (1 + rs))


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float | None:
    if len(close) < slow + signal:
        return None
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return float((line - sig).iloc[-1] / close.iloc[-1])  # normalised by price


def atr_pct(bars: pd.DataFrame, period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return float(atr.iloc[-1] / bars["close"].iloc[-1])


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> float | None:
    if len(close) < window:
        return None
    m = close.rolling(window).mean().iloc[-1]
    s = close.rolling(window).std(ddof=0).iloc[-1]
    if s == 0 or pd.isna(s):
        return None
    return float((close.iloc[-1] - (m - n_std * s)) / (2 * n_std * s))


def adx(bars: pd.DataFrame, period: int = 14) -> float | None:
    if len(bars) < 2 * period + 1:
        return None
    high, low, close = bars["high"], bars["low"], bars["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=bars.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=bars.index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return _last(out)


def obv_slope(bars: pd.DataFrame, window: int = 20) -> float | None:
    """Slope of on-balance volume over ``window`` days, normalised by average volume."""
    if len(bars) < window + 1:
        return None
    sign = np.sign(bars["close"].diff().fillna(0))
    obv = (sign * bars["volume"]).cumsum().iloc[-window:]
    x = np.arange(len(obv))
    slope = np.polyfit(x, obv.values, 1)[0]
    avg_vol = bars["volume"].iloc[-window:].mean()
    return float(slope / avg_vol) if avg_vol else None


def beta(close: pd.Series, bench_close: pd.Series, window: int = 252) -> float | None:
    r = np.log(close).diff()
    b = np.log(bench_close).diff()
    df = pd.concat([r, b], axis=1, join="inner").dropna().iloc[-window:]
    if len(df) < 60:
        return None
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else None


def compute_technicals(bars: pd.DataFrame, bench: pd.DataFrame | None = None) -> dict:
    """Return a flat dict of technical / historical-performance metrics."""
    out: dict = {}
    if bars is None or len(bars) < 30:
        return out
    close = bars["close"]
    px = float(close.iloc[-1])
    out["price"] = px
    out["change_1d"] = pct_return(close, 1)
    out["ret_1w"] = pct_return(close, 5)
    out["ret_1m"] = pct_return(close, 21)
    out["ret_3m"] = pct_return(close, 63)
    out["ret_6m"] = pct_return(close, 126)
    out["ret_1y"] = pct_return(close, 252)
    out["ret_ytd"] = ytd_return(close)
    out["max_dd_1y"] = max_drawdown(close, 252)
    out["rv_20"] = realized_vol(close, 20)
    out["rv_60"] = realized_vol(close, 60)
    out["rv_252"] = realized_vol(close, 252)
    if out["ret_1y"] is not None and out["rv_252"]:
        out["ret_vol_1y"] = math.log1p(out["ret_1y"]) / out["rv_252"]
    hi52 = close.iloc[-252:].max()
    lo52 = close.iloc[-252:].min()
    out["dist_52w_high"] = px / hi52 - 1
    out["dist_52w_low"] = px / lo52 - 1
    # momentum / trend
    out["mom_20"] = pct_return(close, 20)
    out["mom_60"] = pct_return(close, 60)
    out["mom_120"] = pct_return(close, 120)
    s50, s200 = sma(close, 50), sma(close, 200)
    out["sma50_dist"] = (px / s50.iloc[-1] - 1) if not pd.isna(s50.iloc[-1]) else None
    out["sma200_dist"] = (px / s200.iloc[-1] - 1) if not pd.isna(s200.iloc[-1]) else None
    if len(s50.dropna()) > 20:
        out["sma50_slope"] = float(s50.iloc[-1] / s50.iloc[-21] - 1)
    score = 0.0
    n = 0
    for key, w in (("sma50_dist", 1.0), ("sma200_dist", 1.0), ("sma50_slope", 2.0)):
        v = out.get(key)
        if v is not None:
            score += w * np.tanh(v * 10)
            n += w
    if out.get("sma50_dist") is not None and out.get("sma200_dist") is not None:
        s50v, s200v = s50.iloc[-1], s200.iloc[-1]
        score += 1.0 if s50v > s200v else -1.0
        n += 1.0
    out["trend_score"] = float(score / n) if n else None
    out["rsi_14"] = rsi(close, 14)
    out["macd_hist"] = macd_hist(close)
    out["atr_pct"] = atr_pct(bars)
    out["bb_pct_b"] = bollinger_pct_b(close)
    out["adx_14"] = adx(bars)
    # volume
    vol = bars["volume"]
    out["volume"] = float(vol.iloc[-1])
    out["avg_volume_20"] = float(vol.iloc[-20:].mean())
    out["dollar_volume_20"] = float((vol * close).iloc[-20:].mean())
    v60 = vol.iloc[-60:].mean()
    out["volume_trend"] = float(out["avg_volume_20"] / v60 - 1) if v60 else None
    out["rel_volume"] = float(vol.iloc[-1] / out["avg_volume_20"]) if out["avg_volume_20"] else None
    out["obv_slope_20"] = obv_slope(bars)
    if bench is not None and len(bench) > 60:
        out["beta_1y"] = beta(close, bench["close"])
    return out
