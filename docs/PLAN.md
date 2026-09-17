# Stock Screener Dashboard — Functional and Technical Plan

Status: v1 plan, implemented in this repository (see `README.md` for how to run).
Date: 2026-09-17

## 1. Goal and scope

Build a screener dashboard that compares a configurable universe of US stocks
along three axes:

1. **Historical performance** — returns, drawdown, realised volatility, trend,
   momentum, RSI, volume and volume trend, computed from daily stock bars.
2. **Expected (market-implied) performance** — the forward-looking distribution of
   the stock price that the *end-of-day option chain* implies: implied volatility,
   expected move, skew, term structure, and the risk-neutral probability of gains
   or losses beyond thresholds.
3. **Risk profile** — implied downside risk (probability of a loss beyond x %,
   expected shortfall, downside deviation), tail asymmetry, implied-vs-realised
   volatility, and leverage/valuation context from fundamentals.

All option-derived measures are set in relation to the corresponding stock quote
(spot, implied forward, realised volatility) so that they are comparable across
names.

The deliverable is a Python application (`stock_screener` package) with:

* a data layer for the Massive REST API (Stocks Starter, Options Starter,
  Fundamentals add-on; Indices/Futures Starter optional),
* an analytics layer built on QuantLib for all option-pricing mathematics,
* a Dash + AG Grid table UI with configurable columns, rich filtering, sorting,
  presets and a per-ticker detail panel,
* a CLI to refresh data and serve the dashboard,
* a synthetic data provider so the whole stack runs and is testable without an
  API key.

Out of scope for v1: intraday/streaming data, order execution, portfolio
accounting, real-world (P-measure) return forecasting beyond what is disclosed as
a heuristic.

## 2. Data sources and tier constraints

Facts below come from the official Massive Python client source
(`massive` 2.8.x, endpoint paths) and public plan descriptions. They should be
re-verified against the account's actual entitlements; the code degrades
gracefully when a field is missing.

| Need | Endpoint (client method) | Tier | Notes / constraints |
|---|---|---|---|
| Daily OHLCV, 5 y history | `/v2/aggs/ticker/{t}/range/1/day/{from}/{to}` (`list_aggs`) | Stocks Starter | 15-min delayed; adjusted for splits (`adjusted=true`). |
| Latest stock quote (EOD) | `/v2/aggs/ticker/{t}/prev` (`get_previous_close_agg`), or last daily bar | Stocks Starter | Snapshot endpoints also exist; the last daily bar is sufficient for EOD screening. |
| Ticker reference | `/v3/reference/tickers`, `/v3/reference/tickers/{t}` (`list_tickers`, `get_ticker_details`) | Stocks Starter | Name, sector (SIC), market cap, shares outstanding. |
| Dividends (for yield q) | `/v3/reference/dividends` (`list_dividends`) | Stocks Starter | Used to build a continuous dividend yield for option pricing. |
| Server-side indicators | `/v1/indicators/{sma,ema,rsi,macd}/{t}` | Stocks Starter | Available, but we compute indicators locally from bars (one call per ticker instead of four, full reproducibility). Kept as optional cross-check. |
| Option chain snapshot | `/v3/snapshot/options/{underlying}` (`list_snapshot_options_chain`) | Options Starter | Returns per contract: `details`, `day` (OHLCV, vwap, previous_close), `greeks`, `implied_volatility`, `open_interest`, `underlying_asset.price`, `break_even_price`. **`last_quote`/`last_trade` are not delivered on Starter**, so the contract price used is `day.close` (fallback `day.vwap`). Massive's IV/Greeks are computed from their quote feed and are delivered. Paginated (default 250/page, max 250). |
| Option contract reference | `/v3/reference/options/contracts` | Options Starter | Only needed for historical back-fills; the snapshot carries `details`. |
| Option daily bars | `/v2/aggs/ticker/O:.../range/1/day/...` | Options Starter | One call per contract — used only for optional per-contract history, not for screening. |
| Financial ratios (TTM, daily) | `/stocks/financials/v1/ratios` (`list_financials_ratios`) | Fundamentals add-on | P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Sales, ROE, ROA, D/E, current, quick, dividend yield, market cap, EPS, FCF. Supports filtering by `ticker.any_of`. |
| Income / balance / cash-flow statements | `/stocks/financials/v1/{income-statements,balance-sheets,cash-flow-statements}` | Fundamentals add-on | Used for revenue growth, margins, net debt. |
| Index levels (optional) | `/v3/snapshot/indices`, `/v2/aggs/ticker/I:VIX/...` | Indices Starter | VIX level/term for market-regime context, SPX for beta. Without the tier, beta uses SPY (a stock ticker) and VIX columns are blank. |
| Futures (optional) | futures endpoints | Futures Starter | Not used in v1 (reserved for rates/VIX-futures term structure). |

Operational constraints:

* All Starter tiers are 15-minute delayed and advertise unlimited request counts;
  the client is still throttled (configurable concurrency and a token bucket) to
  be a good citizen and avoid 429 retries.
* EOD semantics: the refresh job should run after 16:20 ET so that the delayed
  snapshot reflects the closing state. Each refresh persists the cleaned chain
  summary per ticker and date, which builds the IV history needed for IV rank
  and percentile over time.
* Request budget for a 100-name universe ≈ 100 (bars) + 100 (details) +
  ~1 500–2 500 (chain pages) + ~4 (ratios, batched) ≈ 2–3 k requests, i.e. a few
  minutes at 8 concurrent requests.

## 3. Metric catalogue (screener columns)

Columns are grouped; every column has a key, a display name, a unit/format, a
tooltip and a group so that the UI can expose a column chooser and presets. The
authoritative list is `stock_screener/app/columns.py`.

### 3.1 Identification and quote
ticker, name, sector (SIC description), price (last close), change 1d %, market
cap, average dollar volume 20d.

### 3.2 Historical performance
return 1w / 1m / 3m / 6m / 1y / YTD; max drawdown 1y; realised volatility 20d /
60d / 252d (annualised, log returns); return/vol ratio 1y (Sharpe-like with
zero rate); beta 1y vs benchmark; 52-week high/low distance.

### 3.3 Technicals
momentum (rate of change 20d / 60d / 120d), trend score (price vs SMA50/SMA200,
SMA50 slope, SMA50>SMA200), SMA distance %, RSI-14 (Wilder), MACD histogram
(12/26/9), ATR-14 %, Bollinger %B (20, 2σ), ADX-14, volume 20d, volume trend
(20d/60d average ratio), relative volume (last / 20d avg), OBV slope 20d.

### 3.4 Option-implied expectations (from the EOD chain)
* ATM IV at constant 30 / 60 / 90 calendar days (interpolated in total variance).
* IV rank and IV percentile (from stored history; blank until ≥ 20 observations).
* IV/RV ratio (IV30 / realised 20d) — the volatility risk premium.
* Expected move 30d (%): from the risk-neutral standard deviation and, as a
  cross-check, from the ATM straddle price.
* Implied forward and implied carry (forward/spot − 1 − r·T): captures
  dividends and borrow.
* Term-structure slope: IV90 − IV30 (contango/backwardation).
* 25-delta skew: IV(25Δ put) − IV(25Δ call) at 30d; and risk reversal in vol
  points; skew as a fraction of ATM IV.
* Risk-neutral moments at 30d: mean return (should ≈ r − q), std, skewness,
  excess kurtosis.
* Probability of a gain ≥ +5 % / +10 % in 30d (risk-neutral).
* Put/call open-interest ratio, put/call volume ratio, total option volume,
  total open interest, max pain (nearest monthly).
* Chain quality: number of usable contracts, fit RMSE, liquidity score.

### 3.5 Option-implied downside risk
* Probability of a loss ≥ 5 % / 10 % / 20 % in 30d (risk-neutral CDF).
* Implied 5 % VaR (30d) and expected shortfall at 5 % (30d), both in % of spot.
* Downside deviation (semi-deviation) of the risk-neutral distribution.
* Tail asymmetry: P(loss ≥ 10 %) − P(gain ≥ 10 %).
* Left-tail/right-tail IV: IV at 10Δ put minus ATM.

### 3.6 Fundamentals (add-on)
P/E, P/B, P/S, P/FCF, EV/EBITDA, EV/Sales, ROE, ROA, debt/equity, current ratio,
dividend yield, FCF yield, revenue growth YoY, net margin, gross margin.

### 3.7 Composite scores
momentum score, quality score, risk score, value score, "screener score" —
percentile-rank blends across the loaded universe, with the weights in
`analytics/scoring.py`. They are conveniences, not forecasts.

## 4. Methodology (QuantLib)

All pricing mathematics is done through QuantLib-Python so that conventions
(day counts, calendars, curves, Black-Scholes-Merton process) are consistent and
auditable. Numerical results in the risk-neutral module use NumPy/SciPy on top
of QuantLib-priced grids.

### 4.1 Market setup
* Calendar `UnitedStates(NYSE)`, day counter `Actual365Fixed`, evaluation date =
  chain date.
* Risk-free curve: `FlatForward` at a configurable rate (default 4.0 %; can be
  replaced by a per-tenor curve later, e.g. from the Futures tier).
* Dividend yield q: trailing 12-month cash dividends / spot from the dividends
  endpoint (fallback: ratios `dividend_yield`, fallback 0).
* Spot S₀: `underlying_asset.price` from the chain snapshot (delayed), cross-
  checked against the last daily bar close; a discrepancy above 1 % is flagged.

### 4.2 Chain cleaning
Keep contracts with a positive `day.close` (or vwap) updated on the chain date,
`open_interest > 0` or `day.volume > 0`, 7 ≤ days-to-expiry ≤ 400, and a
moneyness |ln(K/F)| ≤ 1.5 · σ_ATM · √T (adaptive band). Prices below intrinsic
value are dropped. Massive's `implied_volatility` is used as the primary IV; when
it is missing, IV is solved with QuantLib (`VanillaOption.impliedVolatility`,
Black-Scholes-Merton process, European exercise; American exercise via a
binomial engine is available as an option and disabled by default for speed).
Out-of-the-money contracts are preferred per strike (puts below the forward,
calls above), which avoids early-exercise and deep-ITM staleness effects.

### 4.3 Implied forward
Per expiry, the forward is estimated from put-call parity on strikes with both a
call and a put: F = K + e^{rT}(C − P), median across the three strikes closest
to spot; fallback F = S₀·e^{(r−q)T}. Implied carry = ln(F/S₀)/T − r.

### 4.4 Volatility surface
Per expiry, IV as a function of log-moneyness k = ln(K/F) is smoothed with a
vega- and liquidity-weighted quadratic (convexity enforced; linear fallback for
sparse expiries) whose slope decays smoothly (C¹) outside the observed strike
range. ATM IV per expiry is the fit at k = 0. The surface (`SmileSurface`) is
linear in total variance σ²T between the bracketing expiries and flat in vol
before the first / after the last expiry — the same conventions as QuantLib's
`BlackVarianceSurface`, but smooth in strike, which the second derivative in
§4.5 requires (a bilinear grid produces density spikes at grid knots; QuantLib's
bicubic option also interpolates cubically in time and overshoots with few,
unevenly spaced expiries). A QuantLib `BlackVarianceSurface` is built from the
same fits on a dense strike grid and its value at the 30-day forward is compared
with the smooth surface; the difference is reported as the "QL check" column and
asserted in the tests (< 5 bp of vol).

### 4.5 Risk-neutral distribution (Breeden–Litzenberger)
For the 30-day horizon (and any configurable horizon), European call prices
C(K) are generated from the fitted surface on a dense strike grid with a
QuantLib `AnalyticEuropeanEngine`. The risk-neutral density is
f(K) = e^{rT} ∂²C/∂K², computed by finite differences and clipped at zero, then
renormalised; the tails outside the fitted strike range are completed with the
lognormal tails implied by the edge IVs. From f we obtain the CDF, the
probabilities of gains/losses beyond thresholds, VaR, expected shortfall,
semi-deviation, skewness and kurtosis. The standard deviation of the
distribution defines the primary "expected move".

Interpretation caveat (surfaced in the UI): these are *risk-neutral* quantities.
They embed risk premia; e.g. the implied probability of a large loss is usually
higher than the realised frequency. They are excellent for cross-sectional
comparison and for pricing hedges, not unbiased forecasts.

### 4.6 Greeks-based aggregates
Where Massive Greeks are present they are used directly for delta-bucketing
(25Δ, 10Δ); otherwise deltas are recomputed with QuantLib from the fitted IV.
Skew and risk reversals are read from the fitted smile at the strike where the
QuantLib delta equals the target.

## 5. Architecture

```
stock_screener/
  config.py              Settings (env / .env): API key, universe, tiers, rate, horizons
  data/
    provider.py          MarketDataProvider protocol (bars, details, dividends, chain, ratios, index)
    massive_provider.py  Massive REST implementation (massive.RESTClient), throttling, retries
    synthetic_provider.py Deterministic synthetic market (for demo/tests)
    cache.py             Parquet + JSON cache under data_dir, per ticker/date
    history.py           IV history store (append-only parquet) for IV rank/percentile
    universe.py          Universe from config / CSV / Massive ticker list filters
  analytics/
    technicals.py        Indicators from OHLCV (pure pandas/numpy)
    market.py            QuantLib market environment (curves, process, calendars)
    chain.py             Chain cleaning, forward estimation, smile/term fits
    qlpricing.py         QuantLib IV solving, Greeks, pricing grids, vol surface
    risk_neutral.py      Breeden–Litzenberger density and derived metrics
    fundamentals.py      Ratios/statements → fundamental columns
    scoring.py           Cross-sectional composite scores
    pipeline.py          Orchestrates per-ticker computation → screener DataFrame
  app/
    columns.py           Column registry (key, label, group, format, tooltip)
    presets.py           Column/filter presets
    dashboard.py         Dash app factory (AG Grid, chooser, detail panel)
  cli.py                 `screener refresh|serve|demo`
tests/                   Unit tests (synthetic data, QuantLib cross-checks)
docs/PLAN.md             This document
```

Flow: `refresh` → provider fetch (parallel per ticker) → cache → per-ticker
analytics → `screener.parquet` (+ `chains/<date>/<ticker>.parquet`,
`iv_history.parquet`) → `serve` loads the parquet and renders the grid. The UI
never calls the API directly, so the dashboard is fast and the API budget is
deterministic.

## 6. UI functionality

* **Table** (AG Grid community): server-side data is loaded once; client-side
  sorting (multi-column), per-column filters (number range, text, set), quick
  search, pinned ticker column, conditional colouring for signed metrics,
  tooltips with metric definitions, CSV export, pagination toggle.
* **Column chooser**: grouped checklist plus presets (Overview, Momentum,
  Volatility & Options, Downside Risk, Fundamentals, All). Column state (order,
  visibility) is kept in the browser session.
* **Global filters**: sector, market-cap band, min average dollar volume,
  min option liquidity, "has fundamentals". These are applied server-side to the
  row data.
* **Detail panel** (row click): price chart with SMA50/200 and volume; implied
  volatility smile per expiry and term structure; the 30-day risk-neutral
  density with loss/gain thresholds; a card with the key numbers and the data
  quality flags.
* **Status bar**: data date, universe size, tier flags, provider (Massive /
  synthetic), warnings (stale chains, missing fundamentals).

## 7. Validation and testing

* Technicals: RSI/SMA/ATR against hand-computed reference series.
* QuantLib: Black-Scholes-Merton prices and Greeks against closed forms,
  implied-vol round trip (European and American), put-call parity, total-variance
  interpolation of the surface, and Breeden–Litzenberger on a flat-vol synthetic
  chain reproducing the lognormal density, its moments and tail probabilities;
  smooth surface vs QuantLib `BlackVarianceSurface` agreement.
* Chain cleaning: adversarial fixtures (missing IV, zero OI, intrinsic
  violations, stale timestamps).
* Pipeline end-to-end on the synthetic provider; dashboard smoke test
  (layout builds, callbacks return).
* Provider: response parsing tested with recorded JSON fixtures shaped like the
  client models (network is not required).

## 8. Assumptions, limitations, risks

* Starter tiers have no option quotes; `day.close` is a last-trade-based price
  and is stale for illiquid contracts. Mitigations: liquidity filters, OTM
  preference, Massive IV as primary input, fit residual reported as a quality
  column. Users who upgrade to a tier with quotes get `last_quote.midpoint`
  automatically (the code prefers it when present).
* Delayed data: the chain snapshot pulled at 16:20 ET is the 16:05 ET state;
  minor differences from the official close are expected.
* Risk-neutral ≠ real-world. Expected-performance columns are market-implied
  and carry risk premia. No attempt is made to estimate the pricing kernel in v1.
* Fundamentals are TTM as-reported; sector classification is SIC based.
* IV rank/percentile need history; they populate as the refresh job runs daily.
  A back-fill using option daily bars is possible but expensive (one request per
  contract-day) and is left to the roadmap.
* Rates: a single flat rate is an approximation; for tenors ≤ 90 days the
  impact on IV and density is small (≪ 0.5 vol points), but it is configurable.
* American exercise: single-stock options are American. Using European IV for
  OTM options with T ≤ 90d introduces errors that are typically below the
  bid-ask noise; the binomial mode exists for validation.
* Entitlements may differ from the public plan matrix; the provider logs and
  tolerates 403/404 per endpoint and the pipeline continues with blanks.

## 9. Roadmap

1. v1 (this repo): EOD refresh, table, detail panel, synthetic mode, tests.
2. Scheduler and incremental refresh (only changed contracts), IV history
   back-fill, SVI smile fit with arbitrage checks.
3. Indices tier: VIX regime column, SPX beta, implied correlation.
4. Futures tier: rate curve from SOFR/Treasury futures; VIX futures term
   structure.
5. P-measure adjustment (e.g. Ross recovery or empirical pricing-kernel
   calibration) clearly separated from risk-neutral columns.
6. Alerts (screen conditions → notifications) and saved screens.

## 10. Derivatives tab: knock-out certificates and warrant strategies

### 10.1 Scope
A second top-level tab lists the leveraged retail products available at the
broker (Scalable Capital) for every screened underlying, samples buy-only
strategies of up to four instruments, evaluates them probabilistically against
the option-implied distribution of the underlying, and recommends the best
realization per strategy class and underlying. The dashboard never places
orders; the CLI is used for queries only.

### 10.2 Data source
The broker's CLI could not be verified from the development environment, so the
adapter (`derivatives/sources.py`) is configuration-driven: `config/derivatives.yaml`
holds the command template (placeholders `{underlying}`, `{type}`, `{ticker}`),
the ticker→identifier map, the output format (JSON / JSON lines / CSV) and the
field mapping with dotted paths and a product-type value map. A `file` source
reads exported CLI output with the same mapping and a `synthetic` source
generates a shelf from the underlying's spot and surface. `screener derivatives
probe` prints what the adapter parses. Products carry ISIN/WKN, issuer, type,
strike, barrier, ratio, bid/ask, maturity (None = open-end), leverage and,
when delivered, the financing rate.

### 10.3 Costs
`config/costs.yaml` is the only place with fee numbers: plan (FREE Broker,
PRIME+), venue (gettex, Xetra) order fees (fixed, percentage, minimum, free
above a threshold), partner-issuer conditions, exit-spread assumptions,
knock-out recovery, default KO financing rates, EUR/USD and optional taxes.
The file is flagged `verified: false` until checked against the current price
list; the UI shows the active fee description in its status line.

### 10.4 Product models
* Open-end KO: value = intrinsic × ratio / FX; the strike accrues the financing
  rate (`K_t = K·e^{fin·t}`, shorts with the opposite sign) and the barrier moves
  proportionally. On knock-out the holder receives `recovery × |barrier − strike|
  × ratio / FX` (zero for classic turbos).
* Warrants: Black-Scholes-Merton with the warrant's own implied volatility
  (solved with QuantLib from the mid) held constant over the holding period;
  American exercise and issuer vol changes are ignored (documented limitation).
* Product grids show fair value, premium over fair value, implied vs listed-
  option volatility, delta, leverage and barrier distance.

### 10.5 Strategy classes and sampling heuristics
Twelve buy-only classes: long/short KO, call/put warrant, KO long/short ladders
(2–4 legs, staggered barriers), warrant straddle and strangle, KO + warrant
hedge (long/put, short/call), KO long+short pair, call-warrant ladder. The
search space is bounded by product filters (price, spread, barrier distance,
leverage, maturity window, moneyness), spread-based bucketing (barrier-distance
buckets for KOs, moneyness × maturity buckets for warrants, ≤ 12 products per
type), class constraints (minimum barrier spacing, strike gaps, equal
maturities) and a cap per class with evenly spaced retention along the class
parameter (so the retained set still spans the whole range).

### 10.6 Evaluation
Terminal prices are drawn from the screener's risk-neutral density at the
strategy horizon (stratified inverse-CDF sampling, common random numbers across
products). The drift is configurable: risk-neutral (default), a fixed annual
premium, or the underlying's historical 6-month trend (capped). Knock-out
events are drawn with the Brownian-bridge crossing probability given start and
end price, using the surface vol at the barrier. Each leg is bought at the ask
(integer quantities from a budget split), sold at the modelled exit price minus
half the spread, with entry and exit order fees from the cost model. Metrics:
expected, median and std of return, P(profit), P(loss > 50 %), P(total loss),
P(knock-out), VaR/ES 5 %, 95th/99th percentiles, Omega, utility
`E[r] − λ·|ES₅|`, effective leverage, break-even moves and cost drag.

### 10.7 Recommendation and stability
Per underlying and class, realizations with P(total loss) above a configurable
limit are excluded; the rest are ranked by utility. The best is flagged with the
utility gap to the runner-up and a stability score (1 − dispersion of utility
among its neighbours along the class parameter). The detail view shows the
payoff at the horizon (deterministic and knock-out-adjusted) over the implied
distribution, the return histogram with VaR/ES, the risk/return scatter of all
sampled realizations with the recommendation highlighted, and the metrics
along the class parameter.

### 10.8 Limitations
Unverified CLI interface and fee values; FX risk not simulated (static
EUR/USD); warrant vol held constant; continuous barrier monitoring (overnight
gaps only through the terminal distribution); risk-neutral expectations embed
risk premia unless a drift mode is chosen; issuer credit risk ignored.
