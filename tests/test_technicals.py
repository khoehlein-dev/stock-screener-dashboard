import numpy as np
import pandas as pd
import pytest

from stock_screener.analytics import technicals as T


def _bars(closes):
    idx = pd.bdate_range("2025-01-01", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6, "vwap": c}
    )


def test_rsi_extremes():
    up = _bars(np.linspace(100, 150, 40))
    down = _bars(np.linspace(150, 100, 40))
    assert T.rsi(up["close"]) == 100.0
    assert T.rsi(down["close"]) < 1e-9


def test_rsi_matches_explicit_wilder_loop():
    rng = np.random.default_rng(0)
    closes = pd.Series(100 + np.cumsum(rng.normal(0, 1, 80)))
    period = 14
    ch = closes.diff().dropna().to_numpy()
    gains, losses = np.clip(ch, 0, None), np.clip(-ch, 0, None)
    ag, al = gains[:period].mean(), losses[:period].mean()
    for g, ls in zip(gains[period:], losses[period:], strict=True):
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + ls) / period
    expected = 100 - 100 / (1 + ag / al)
    assert T.rsi(closes, period) == pytest.approx(expected, abs=1e-9)


def test_returns_and_vol():
    b = _bars(100 * np.exp(np.cumsum(np.full(300, 0.001))))
    m = T.compute_technicals(b)
    assert abs(m["ret_1m"] - (np.exp(0.021) - 1)) < 1e-9
    assert m["rv_20"] < 1e-9  # constant log returns -> zero vol
    assert m["max_dd_1y"] == 0.0
    assert m["trend_score"] > 0.3
    assert 0 <= m["bb_pct_b"] <= 1.2


def test_short_history_is_empty():
    assert T.compute_technicals(_bars(np.arange(10) + 100.0)) == {}
