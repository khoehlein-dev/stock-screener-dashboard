import datetime as dt

import pandas as pd

from stock_screener.analytics.pipeline import build_screener, default_as_of
from stock_screener.app.columns import BY_KEY, COLUMNS
from stock_screener.app.dashboard import apply_global_filters, column_defs, create_app
from stock_screener.app.presets import PRESETS
from stock_screener.data.cache import DataCache
from stock_screener.data.history import IVHistory
from stock_screener.data.massive_provider import parse_aggs, parse_chain, parse_ratios

AS_OF = dt.date(2026, 9, 17)


def test_default_as_of_skips_weekend():
    assert default_as_of("synthetic", dt.date(2026, 9, 19)) == dt.date(
        2026, 9, 18
    )  # Saturday -> Friday
    assert default_as_of("synthetic", dt.date(2026, 9, 17)) == dt.date(2026, 9, 17)


def test_build_screener_and_history(provider, settings, tmp_path):
    settings.data_dir = tmp_path
    cache = DataCache(tmp_path)
    df = build_screener(provider, settings, AS_OF, cache, ["AAPL", "MSFT"])
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    for col in ("iv_30", "p_loss_10", "expected_move", "rsi_14", "pe", "screener_score"):
        assert df[col].notna().all(), col
    assert cache.get_density(AS_OF, "AAPL") is not None
    assert cache.get_frame(AS_OF, "fits", "AAPL") is not None
    # a second run reads from cache and the IV history has one row per ticker/date
    df2 = build_screener(provider, settings, AS_OF, cache, ["AAPL", "MSFT"])
    assert df2["iv_30"].tolist() == df["iv_30"].tolist()
    hist = IVHistory(tmp_path / "iv_history.parquet")
    assert len(hist.df) == 2
    assert df["iv_rank"].isna().all()  # not enough history yet


def test_iv_rank_percentile(tmp_path):
    hist = IVHistory(tmp_path / "h.parquet", min_observations=5)
    rows = pd.DataFrame(
        {
            "ticker": "X",
            "date": pd.bdate_range("2026-01-01", periods=10),
            "iv30": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
            "iv60": 0.3,
            "iv90": 0.3,
            "rv20": 0.2,
            "spot": 100.0,
        }
    )
    hist.append(rows)
    rank, pct = hist.rank_and_percentile("X", 0.425, AS_OF)
    assert rank == (0.425 - 0.2) / (0.65 - 0.2)
    assert pct == 0.5


def test_column_registry_consistency():
    keys = [c.key for c in COLUMNS]
    assert len(keys) == len(set(keys))
    for name, cols in PRESETS.items():
        assert all(k in BY_KEY for k in cols), name
    defs = column_defs(["ticker", "iv_30", "unknown"], {"ticker", "iv_30"})
    assert [d["field"] for d in defs] == ["ticker", "iv_30"]
    assert defs[0]["pinned"] == "left"


def test_global_filters():
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "sector": ["S1", "S2", "S1"],
            "market_cap": [1e9, 5e9, None],
            "dollar_volume_20": [1e7, 1e8, 1e6],
            "open_interest": [10, 1000, None],
            "iv_30": [0.2, None, 0.3],
            "has_fundamentals": [True, True, False],
        }
    )
    assert apply_global_filters(df, ["S1"], 0, 0, 0, False, False)["ticker"].tolist() == ["A", "C"]
    assert apply_global_filters(df, None, 2e9, 0, 0, False, False)["ticker"].tolist() == ["B"]
    assert apply_global_filters(df, None, 0, 0, 0, True, True)["ticker"].tolist() == ["A"]


def test_dashboard_builds_and_detail_renders(provider, settings, tmp_path):
    settings.data_dir = tmp_path
    cache = DataCache(tmp_path)
    df = build_screener(provider, settings, AS_OF, cache, ["AAPL"])
    cache.put_screener(AS_OF, df)
    app = create_app(settings, tmp_path)
    assert app.layout is not None
    store = app.store
    assert store.as_of == AS_OF
    from stock_screener.app.dashboard import density_figure, price_figure, smile_figure, term_figure

    row = store.df.iloc[0]
    assert len(price_figure(store.bars("AAPL"), "AAPL").data) == 4
    assert (
        len(smile_figure(store.chain("AAPL"), store.fits("AAPL"), row["price"], "AAPL").data) >= 2
    )
    assert len(term_figure(store.fits("AAPL"), row["rv_20"], "AAPL").data) == 1
    assert len(density_figure(store.density("AAPL"), row["price"], "AAPL").data) == 3


def test_massive_parsers_on_fixture_payloads():
    # Massive stamps daily bars at midnight US/Eastern (04:00 UTC)
    t_ms = int(pd.Timestamp("2026-09-17 00:00", tz="America/New_York").timestamp() * 1000)
    aggs = [{"t": t_ms, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100, "vw": 1.4}]
    bars = parse_aggs(aggs)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "vwap"]
    assert bars.index[0] == pd.Timestamp("2026-09-17")
    chain = parse_chain(
        [
            {
                "details": {
                    "ticker": "O:AAPL261016C00200000",
                    "contract_type": "call",
                    "strike_price": 200,
                    "expiration_date": "2026-10-16",
                    "exercise_style": "american",
                    "shares_per_contract": 100,
                },
                "day": {
                    "close": 5.1,
                    "vwap": 5.0,
                    "volume": 12,
                    "last_updated": 1758139200000000000,
                },
                "greeks": {"delta": 0.4, "gamma": 0.01, "theta": -0.05, "vega": 0.2},
                "implied_volatility": 0.31,
                "open_interest": 1234,
                "underlying_asset": {"price": 198.5, "ticker": "AAPL"},
            },
            {
                "details": {
                    "ticker": "O:AAPL261016P00200000",
                    "contract_type": "put",
                    "strike_price": 200,
                    "expiration_date": "2026-10-16",
                },
                "day": {},
                "open_interest": 0,
                "last_quote": {"bid": 1.0, "ask": 1.2},
            },
            {"details": {"contract_type": "call"}},  # malformed -> dropped
        ]
    )
    assert len(chain) == 2
    call = chain.iloc[0]
    assert call["price"] == 5.1 and call["price_source"] == "close" and call["iv"] == 0.31
    assert call["underlying_price"] == 198.5
    put = chain.iloc[1]
    assert put["price"] == 1.1 and put["price_source"] == "mid"  # quotes preferred when present
    r = parse_ratios(
        {
            "ticker": "AAPL",
            "date": "2026-09-16",
            "price_to_earnings": 30.2,
            "market_cap": 3e12,
            "unknown_field": 1,
        }
    )
    assert r.price_to_earnings == 30.2 and r.market_cap == 3e12
