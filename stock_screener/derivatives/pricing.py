"""Valuation and payoff models for knock-out certificates and warrants.

Conventions
-----------
* Underlying prices ``S`` and strikes are in the underlying currency (USD for
  the screener's US stocks); product prices are in EUR. ``fx`` is EUR/USD, so
  ``value_eur = value_usd * ratio / fx``.
* Knock-out certificates (open-end): the strike accrues financing daily,
  ``K_t = K * exp(fin * t)`` for longs (``exp(-fin_short * t)`` for shorts) and
  the barrier moves proportionally. Value = intrinsic (the small issuer premium
  observed in the ask price is a cost paid at entry). On knock-out the holder
  receives ``recovery * (barrier_t - K_t)`` (longs) — zero for classic turbos
  where the barrier equals the strike.
* Warrants: Black-Scholes-Merton (QuantLib) with the warrant's own implied
  volatility, assumed unchanged over the holding period (limitation: vega risk
  of the *issuer's* vol quote is not modelled).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from ..analytics.qlpricing import MarketEnv, bs_price, implied_vol
from .models import Derivative


@dataclass
class MarketContext:
    """Everything about the underlying the derivatives module needs."""

    ticker: str
    as_of: dt.date
    spot: float  # underlying currency
    r: float
    q: float
    vol_fn: Callable[[float, float], float]  # (t_years, strike) -> Black vol
    surface_t_min: float = 0.0

    def vol(self, t: float, strike: float) -> float:
        return float(self.vol_fn(max(t, 1e-6), strike))


# --------------------------------------------------------------------------------------
# warrants
# --------------------------------------------------------------------------------------


def warrant_price_from_vol(
    ptype: str,
    S: float,
    K: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    ratio: float,
    fx: float,
) -> float:
    typ = "call" if ptype == "call_warrant" else "put"
    return bs_price(typ, S, K, t, r, q, sigma) * ratio / fx


def warrant_implied_vol(
    product: Derivative, ctx: MarketContext, fx: float, price: float | None = None
) -> float | None:
    """Implied vol of the warrant's mid (or given EUR price), solved with QuantLib."""
    if product.maturity is None:
        return None
    px_eur = product.mid if price is None else price
    px_usd = px_eur * fx / product.ratio
    typ = "call" if product.product_type == "call_warrant" else "put"
    env = MarketEnv(ctx.as_of, ctx.r)
    return implied_vol(env, typ, ctx.spot, product.strike, product.maturity, px_usd, q=ctx.q)


def warrant_values(
    product: Derivative,
    S_T: np.ndarray,
    t_elapsed: float,
    sigma: float,
    ctx: MarketContext,
    fx: float,
) -> np.ndarray:
    """EUR value per unit at the horizon for an array of underlying prices (vectorised BSM)."""
    if product.maturity is None:
        return np.zeros_like(S_T)
    T_total = (product.maturity - ctx.as_of).days / 365.0
    tau = T_total - t_elapsed
    K = product.strike
    S_T = np.asarray(S_T, dtype=float)
    if tau <= 1e-6:
        intrinsic = (
            np.maximum(S_T - K, 0)
            if product.product_type == "call_warrant"
            else np.maximum(K - S_T, 0)
        )
        return intrinsic * product.ratio / fx
    sq = sigma * math.sqrt(tau)
    d1 = (np.log(S_T / K) + (ctx.r - ctx.q + 0.5 * sigma * sigma) * tau) / sq
    d2 = d1 - sq
    if product.product_type == "call_warrant":
        v = S_T * math.exp(-ctx.q * tau) * norm.cdf(d1) - K * math.exp(-ctx.r * tau) * norm.cdf(d2)
    else:
        v = K * math.exp(-ctx.r * tau) * norm.cdf(-d2) - S_T * math.exp(-ctx.q * tau) * norm.cdf(
            -d1
        )
    return np.maximum(v, 0) * product.ratio / fx


# --------------------------------------------------------------------------------------
# knock-out certificates
# --------------------------------------------------------------------------------------


def ko_levels(
    product: Derivative, t_elapsed: float, fin_long: float, fin_short: float
) -> tuple[float, float]:
    """(strike_t, barrier_t) after financing accrual for open-end products."""
    K, B = product.strike, product.barrier if product.barrier is not None else product.strike
    if product.maturity is not None:
        return K, B
    fin = (
        product.financing_rate
        if product.financing_rate is not None
        else (fin_long if product.product_type == "ko_long" else fin_short)
    )
    factor = (
        math.exp(fin * t_elapsed)
        if product.product_type == "ko_long"
        else math.exp(-fin * t_elapsed)
    )
    return K * factor, B * factor


def ko_intrinsic(
    product: Derivative, S: np.ndarray | float, strike_t: float, fx: float
) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    intr = (
        np.maximum(S - strike_t, 0)
        if product.product_type == "ko_long"
        else np.maximum(strike_t - S, 0)
    )
    return intr * product.ratio / fx


def ko_residual(
    product: Derivative, strike_t: float, barrier_t: float, recovery: float, fx: float
) -> float:
    gap = (barrier_t - strike_t) if product.product_type == "ko_long" else (strike_t - barrier_t)
    return max(gap, 0.0) * recovery * product.ratio / fx


def knockout_probability(
    S0: float, S_T: np.ndarray, barrier: float, sigma: float, t: float, long: bool
) -> np.ndarray:
    """Brownian-bridge probability that the path crossed the barrier given start and end.

    For a long (barrier below): P(min S <= B | S0, S_T) = exp(-2 ln(S0/B) ln(S_T/B) / (σ² t)).
    For a short (barrier above): P(max S >= B | S0, S_T) = exp(-2 ln(B/S0) ln(B/S_T) / (σ² t)).
    Returns 1 where the end point is already beyond the barrier.
    """
    S_T = np.asarray(S_T, dtype=float)
    var = max(sigma * sigma * t, 1e-12)
    if long:
        a, b = math.log(S0 / barrier), np.log(S_T / barrier)
        beyond = S_T <= barrier
    else:
        a, b = math.log(barrier / S0), np.log(barrier / S_T)
        beyond = S_T >= barrier
    if a <= 0:
        return np.ones_like(S_T)
    p = np.exp(-2.0 * a * np.maximum(b, 0) / var)
    return np.where(beyond, 1.0, np.clip(p, 0, 1))


def ko_values(
    product: Derivative,
    S_T: np.ndarray,
    U: np.ndarray,
    t_elapsed: float,
    ctx: MarketContext,
    fx: float,
    recovery: float,
    fin_long: float,
    fin_short: float,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(EUR value per unit at horizon, knocked-out indicator) for sampled end prices.

    ``U`` are uniforms used to draw the knock-out event with the bridge probability, so
    that the same random numbers are shared across products (common random numbers).
    """
    strike_t, barrier_t = ko_levels(product, t_elapsed, fin_long, fin_short)
    long = product.product_type == "ko_long"
    # bridge vol: local vol near the barrier, from the underlying surface at the horizon
    sig = sigma if sigma is not None else ctx.vol(max(t_elapsed, 1e-6), barrier_t)
    p_ko = knockout_probability(ctx.spot, S_T, barrier_t, sig, t_elapsed, long)
    knocked = U < p_ko
    alive_value = ko_intrinsic(product, S_T, strike_t, fx)
    residual = ko_residual(product, strike_t, barrier_t, recovery, fx)
    return np.where(knocked, residual, alive_value), knocked


def ko_fair_value(product: Derivative, ctx: MarketContext, fx: float) -> float:
    return float(ko_intrinsic(product, ctx.spot, product.strike, fx))


def ko_payoff_curve(
    product: Derivative,
    S_grid: np.ndarray,
    t_elapsed: float,
    ctx: MarketContext,
    fx: float,
    recovery: float,
    fin_long: float,
    fin_short: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic payoff (no path knock-out) and knock-out-adjusted expected value along S_T."""
    strike_t, barrier_t = ko_levels(product, t_elapsed, fin_long, fin_short)
    long = product.product_type == "ko_long"
    sig = ctx.vol(max(t_elapsed, 1e-6), barrier_t)
    p_ko = knockout_probability(ctx.spot, S_grid, barrier_t, sig, t_elapsed, long)
    alive = ko_intrinsic(product, S_grid, strike_t, fx)
    residual = ko_residual(product, strike_t, barrier_t, recovery, fx)
    return alive, (1 - p_ko) * alive + p_ko * residual
