"""QuantLib pricing primitives.

Everything that prices or inverts an option goes through QuantLib:

* :func:`bs_price` / :func:`bs_greeks` use ``ql.BlackCalculator`` (Black-Scholes-
  Merton with continuous dividend yield) on a scalar maturity in years.
* :func:`implied_vol` inverts a price with ``VanillaOption.impliedVolatility``
  on a ``BlackScholesMertonProcess``; European by default, American (CRR binomial)
  optionally.
* :class:`MarketEnv` holds the QuantLib evaluation date, calendar, day counter
  and flat curves so that all modules share one convention.
* :class:`VolSurface` wraps ``ql.BlackVarianceSurface`` (linear in total
  variance between expiries, constant extrapolation in strike).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import QuantLib as ql


def to_ql_date(d: dt.date) -> ql.Date:
    return ql.Date(d.day, d.month, d.year)


@dataclass
class MarketEnv:
    as_of: dt.date
    risk_free_rate: float = 0.04
    calendar: ql.Calendar = None
    day_counter: ql.DayCounter = None

    def __post_init__(self):
        self.calendar = self.calendar or ql.UnitedStates(ql.UnitedStates.NYSE)
        self.day_counter = self.day_counter or ql.Actual365Fixed()
        self.ql_date = to_ql_date(self.as_of)
        ql.Settings.instance().evaluationDate = self.ql_date

    def year_fraction(self, expiry: dt.date) -> float:
        return self.day_counter.yearFraction(self.ql_date, to_ql_date(expiry))

    def discount(self, t: float) -> float:
        return math.exp(-self.risk_free_rate * t)

    def process(self, spot: float, q: float, sigma: float) -> ql.BlackScholesMertonProcess:
        ql.Settings.instance().evaluationDate = self.ql_date
        s = ql.QuoteHandle(ql.SimpleQuote(spot))
        r_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.ql_date, self.risk_free_rate, self.day_counter)
        )
        q_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.ql_date, q, self.day_counter))
        v_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.ql_date, self.calendar, sigma, self.day_counter)
        )
        return ql.BlackScholesMertonProcess(s, q_ts, r_ts, v_ts)


def _opt_type(option_type: str) -> int:
    return ql.Option.Call if option_type.lower().startswith("c") else ql.Option.Put


def bs_price(
    option_type: str, spot: float, strike: float, t: float, r: float, q: float, sigma: float
) -> float:
    """Black-Scholes-Merton price via ``ql.blackFormula``."""
    if t <= 0:
        return max(0.0, (spot - strike) if option_type.startswith("c") else (strike - spot))
    fwd = spot * math.exp((r - q) * t)
    return ql.blackFormula(
        _opt_type(option_type), strike, fwd, sigma * math.sqrt(t), math.exp(-r * t)
    )


def black_price_from_forward(
    option_type: str, forward: float, strike: float, t: float, discount: float, sigma: float
) -> float:
    if t <= 0 or sigma <= 0:
        intrinsic = (forward - strike) if option_type.startswith("c") else (strike - forward)
        return discount * max(0.0, intrinsic)
    return ql.blackFormula(_opt_type(option_type), strike, forward, sigma * math.sqrt(t), discount)


def bs_greeks(
    option_type: str, spot: float, strike: float, t: float, r: float, q: float, sigma: float
) -> dict[str, float]:
    """Delta, gamma, vega (per 1.00 vol), theta (per year) via ``ql.BlackCalculator``."""
    if t <= 0:
        return {"delta": float("nan"), "gamma": float("nan"), "vega": 0.0, "theta": 0.0}
    fwd = spot * math.exp((r - q) * t)
    bc = ql.BlackCalculator(
        ql.PlainVanillaPayoff(_opt_type(option_type), strike),
        fwd,
        sigma * math.sqrt(t),
        math.exp(-r * t),
    )
    return {
        "delta": bc.delta(spot),
        "gamma": bc.gamma(spot),
        "vega": bc.vega(t),
        "theta": bc.theta(spot, t),
    }


def bs_delta(
    option_type: str, forward: float, strike: float, t: float, sigma: float, q: float
) -> float:
    """Spot delta with continuous dividend yield (closed form, matches BlackCalculator)."""
    if t <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
    n = _norm_cdf(d1)
    disc_q = math.exp(-q * t)
    return disc_q * n if option_type.startswith("c") else disc_q * (n - 1.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def implied_vol(
    env: MarketEnv,
    option_type: str,
    spot: float,
    strike: float,
    expiry: dt.date,
    price: float,
    q: float = 0.0,
    american: bool = False,
    initial_vol: float = 0.3,
) -> float | None:
    """Invert a price to Black-Scholes-Merton implied volatility with QuantLib.

    Returns ``None`` when the price is outside the no-arbitrage bounds or the
    solver fails (e.g. price below intrinsic value).
    """
    ql.Settings.instance().evaluationDate = env.ql_date
    ql_exp = to_ql_date(expiry)
    if ql_exp <= env.ql_date or price is None or price <= 0:
        return None
    process = env.process(spot, q, initial_vol)
    payoff = ql.PlainVanillaPayoff(_opt_type(option_type), strike)
    if american:
        exercise = ql.AmericanExercise(env.ql_date, ql_exp)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", 200))
    else:
        exercise = ql.EuropeanExercise(ql_exp)
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    try:
        return float(option.impliedVolatility(price, process, 1e-6, 200, 1e-4, 5.0))
    except RuntimeError:
        return None


class VolSurface:
    """Fitted implied-volatility surface backed by ``ql.BlackVarianceSurface``.

    ``expiries`` are dates, ``strikes`` an increasing array shared by all expiries
    and ``vols`` a matrix ``(len(strikes), len(expiries))`` of Black vols.
    Time interpolation is linear in total variance; strike interpolation is
    bilinear with constant extrapolation. Used as the QuantLib cross-check of the
    smooth :class:`~stock_screener.analytics.chain.SmileSurface`.
    """

    def __init__(
        self, env: MarketEnv, expiries: list[dt.date], strikes: np.ndarray, vols: np.ndarray
    ):
        if len(expiries) < 1:
            raise ValueError("at least one expiry is required")
        self.env = env
        self.expiries = list(expiries)
        self.strikes = np.asarray(strikes, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        ql.Settings.instance().evaluationDate = env.ql_date
        dates = [to_ql_date(e) for e in self.expiries]
        if len(dates) == 1:
            # BlackVarianceSurface needs >= 2 expiries; duplicate with a far date (flat in time).
            dates = dates + [dates[0] + 365]
            self.vols = np.column_stack([self.vols, self.vols])
        m = ql.Matrix(len(self.strikes), len(dates))
        for i in range(len(self.strikes)):
            for j in range(len(dates)):
                m[i][j] = float(self.vols[i, j])
        self.surface = ql.BlackVarianceSurface(
            env.ql_date,
            env.calendar,
            dates,
            list(map(float, self.strikes)),
            m,
            env.day_counter,
            ql.BlackVarianceSurface.ConstantExtrapolation,
            ql.BlackVarianceSurface.ConstantExtrapolation,
        )
        # bilinear = linear in total variance in time (bicubic would also be cubic in time and
        # overshoots with few, unevenly spaced expiries)
        self.surface.setInterpolation("bilinear")
        self.surface.enableExtrapolation()
        self.t_max = env.year_fraction(self.expiries[-1])
        self.t_min = env.year_fraction(self.expiries[0])

    def vol(self, t: float, strike: float) -> float:
        t = min(max(t, 1e-6), self.t_max)
        return float(self.surface.blackVol(t, float(strike)))

    def vols_at(self, t: float, strikes: np.ndarray) -> np.ndarray:
        return np.array([self.vol(t, k) for k in strikes])
