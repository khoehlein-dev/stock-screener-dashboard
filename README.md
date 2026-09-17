# Stock Screener Dashboard

An options-aware stock screener: historical performance, market-implied
expectations and downside risk for a universe of US stocks, in one sortable,
filterable table with configurable columns and a per-ticker detail panel.

* **Data**: [Massive](https://massive.com) REST API (formerly Polygon.io) —
  Stocks Starter, Options Starter, Fundamentals add-on; Indices/Futures optional.
* **Analytics**: Python, pandas/NumPy/SciPy, and **QuantLib** for all option
  mathematics (Black-Scholes-Merton pricing and Greeks, implied-volatility
  inversion, `BlackVarianceSurface`, calendars/day counters).
* **UI**: Dash + AG Grid (community edition) + Plotly.

The functional and technical plan is in [`docs/PLAN.md`](docs/PLAN.md).

## Quick start (no API key needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
screener demo            # synthetic market data -> http://127.0.0.1:8050
```

## With Massive data

```bash
cp .env.example .env     # set SCREENER_MASSIVE_API_KEY and your universe
screener refresh         # fetch + compute for the last NYSE session (run after 16:20 ET)
screener serve           # http://127.0.0.1:8050
```

Useful options: `screener refresh --tickers AAPL,MSFT --as-of 2026-09-16 --workers 8`,
`screener columns` (lists every metric with its definition). Settings can also be
passed as environment variables prefixed `SCREENER_` (see `stock_screener/config.py`).

Each refresh persists raw bars, chains and metadata under `data/raw/<date>/`,
the screener table under `data/screener/`, and appends to `data/iv_history.parquet`
so that IV rank / percentile become available after ~20 sessions.

## What the table shows

| Group | Examples |
|---|---|
| Performance | returns 1w–1y/YTD, max drawdown, realised vol 20/60/252d, return/vol, beta, 52-week distance |
| Technicals | momentum 20/60/120, trend score, SMA distances, RSI-14, MACD histogram, ATR %, Bollinger %B, ADX |
| Volume | volume, 20d average, volume trend (20d vs 60d), relative volume, OBV slope |
| Volatility | IV 30/60/90 (constant maturity), IV rank/percentile, IV/RV, term slope, 25Δ skew, 10Δ put tail |
| Options: expectations | expected move, straddle move, risk-neutral mean/median/skew/kurtosis, P(gain ≥ x), implied carry, max pain |
| Options: downside risk | P(loss ≥ 5/10/20 %), VaR 5 %, expected shortfall 5 %, semi-deviation, tail asymmetry |
| Options: flow | put/call OI and volume ratios, option volume, open interest, liquidity score |
| Fundamentals | P/E, P/B, P/S, P/FCF, EV/EBITDA, ROE, ROA, D/E, dividend & FCF yield, revenue growth, margins |
| Scores | momentum, quality, value, risk and a blended screener score (percentile ranks) |
| Quality | contracts used, fit RMSE, vendor-IV share, clipped density mass, QuantLib surface cross-check, warnings |

All option-derived expectations are **risk-neutral** (they embed risk premia);
they are meant for cross-sectional comparison and hedging, not as unbiased
forecasts. See `docs/PLAN.md` §4 and §8 for the methodology and its limits.

## How the option analytics work

1. Chain snapshot (`/v3/snapshot/options/{ticker}`) is cleaned: fresh, liquid,
   7–400 days to expiry, prices above intrinsic; Massive's IV is used where
   present, otherwise IV is solved with QuantLib (European by default, CRR
   binomial American optional).
2. Per expiry: implied forward from put-call parity (carry fallback), vega-
   weighted quadratic smile fit in log-moneyness with smooth flattening outside
   the observed strikes.
3. Surface: linear in total variance across expiries, smooth in strike; a
   QuantLib `BlackVarianceSurface` built from the same fits serves as a
   consistency check (reported as a column).
4. Breeden–Litzenberger: European calls on a dense strike grid (QuantLib Black
   formula), second derivative → risk-neutral density → probabilities, VaR,
   expected shortfall, moments, expected move.

## Development

```bash
pytest -q          # 27 tests: technicals, QuantLib round trips, density vs lognormal, pipeline, UI
ruff check . && ruff format --check .
```

## Layout

```
stock_screener/
  config.py            settings (.env / SCREENER_* variables)
  data/                provider protocol, Massive + synthetic providers, cache, IV history
  analytics/           technicals, QuantLib pricing, chain fitting, risk-neutral density, scoring, pipeline
  app/                 column registry, presets, Dash application
  cli.py               screener refresh | serve | demo | columns
docs/PLAN.md           functional and technical plan
tests/                 pytest suite
```

## Known limitations

* Starter tiers deliver no option quotes, so contract prices come from the daily
  bar close (stale for illiquid contracts). Massive's own IV/Greeks are still
  delivered and are used as the primary input; liquidity filters and the fit
  RMSE column show how much to trust each row. If the plan includes quotes,
  the midpoint is used automatically.
* Data is 15-minute delayed; refresh after 16:20 ET for end-of-day chains.
* A single flat risk-free rate is used (configurable); the effect at ≤ 90 days is small.
* IV rank/percentile need stored history and are blank until ~20 sessions have been refreshed.
* Entitlements differ between accounts; optional endpoints that return 401/403/404
  degrade to blank columns with a warning rather than failing the refresh.
