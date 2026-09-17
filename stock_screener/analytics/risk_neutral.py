"""Risk-neutral distribution via Breeden–Litzenberger.

Given a fitted :class:`VolSurface`, the horizon ``t`` and the implied forward,
we price European calls on a dense strike grid with QuantLib's Black formula,
differentiate twice in strike (f(K) = e^{rT} ∂²C/∂K²), clip negative mass and
renormalise. All statistics refer to the simple return R = S_T / S_0 − 1.

Interpretation: these are **risk-neutral** (Q-measure) quantities. They include
risk premia and are meant for cross-sectional comparison and hedging, not as
unbiased real-world forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .qlpricing import black_price_from_forward


class Surface(Protocol):
    def vol(self, t: float, strike: float) -> float: ...

    def vols_at(self, t: float, strikes: np.ndarray) -> np.ndarray: ...


@dataclass
class RNDResult:
    t: float
    forward: float
    spot: float
    strikes: np.ndarray
    density: np.ndarray  # density of S_T
    cdf: np.ndarray
    clipped_mass: float
    stats: dict


def _lognormal_tail_mass(spot, forward, t, sigma_lo, sigma_hi, k_lo, k_hi):
    """Mass outside [k_lo, k_hi] under lognormal with edge vols (used for renormalisation)."""
    from scipy.stats import norm

    def cdf_at(K, s):
        d2 = (math.log(forward / K) - 0.5 * s * s * t) / (s * math.sqrt(t))
        return 1 - norm.cdf(d2)  # P(S_T < K)

    return cdf_at(k_lo, sigma_lo), 1 - cdf_at(k_hi, sigma_hi)


def risk_neutral_density(
    surface: Surface,
    spot: float,
    forward: float,
    t: float,
    r: float,
    n_grid: int = 1201,
    width_sigmas: float = 6.0,
) -> RNDResult:
    sigma_ref = surface.vol(t, forward)
    w = width_sigmas * sigma_ref * math.sqrt(t)
    lo, hi = forward * math.exp(-w), forward * math.exp(w)
    strikes = np.exp(np.linspace(math.log(lo), math.log(hi), n_grid))
    disc = math.exp(-r * t)
    vols = surface.vols_at(t, strikes)
    calls = np.array(
        [
            black_price_from_forward("call", forward, k, t, disc, s)
            for k, s in zip(strikes, vols, strict=True)
        ]
    )
    # second derivative on a non-uniform grid
    dC = np.gradient(calls, strikes)
    d2C = np.gradient(dC, strikes)
    density = d2C / disc
    neg = density < 0
    clipped = float(np.trapezoid(np.where(neg, -density, 0.0), strikes))
    density = np.where(neg, 0.0, density)
    mass = float(np.trapezoid(density, strikes))
    # tails beyond the grid are tiny at 6 sigma; renormalise to unit mass
    density = density / mass if mass > 0 else density
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(strikes))])
    cdf = np.clip(cdf / cdf[-1], 0, 1) if cdf[-1] > 0 else cdf
    stats = _stats(strikes, density, cdf, spot, forward, t)
    stats["clipped_mass"] = clipped
    stats["rn_mean_check"] = stats["mean_price"] / forward - 1  # should be ~0
    return RNDResult(t, forward, spot, strikes, density, cdf, clipped, stats)


def _stats(
    K: np.ndarray, f: np.ndarray, cdf: np.ndarray, spot: float, forward: float, t: float
) -> dict:
    ret = K / spot - 1
    mean_price = float(np.trapezoid(K * f, K))
    mean_ret = mean_price / spot - 1
    var = float(np.trapezoid((ret - mean_ret) ** 2 * f, K))
    std = math.sqrt(max(var, 0))
    skew = float(np.trapezoid((ret - mean_ret) ** 3 * f, K)) / std**3 if std > 0 else float("nan")
    kurt = (
        float(np.trapezoid((ret - mean_ret) ** 4 * f, K)) / std**4 - 3 if std > 0 else float("nan")
    )
    down = np.where(ret < 0, ret, 0.0)
    semi_dev = math.sqrt(float(np.trapezoid(down**2 * f, K)))
    return {
        "mean_price": mean_price,
        "mean_ret": mean_ret,
        "std_ret": std,
        "skew": skew,
        "kurtosis": kurt,
        "semi_dev": semi_dev,
        "median_ret": float(quantile(K, cdf, 0.5) / spot - 1),
        "std_annualised": std / math.sqrt(t) if t > 0 else float("nan"),
    }


def quantile(K: np.ndarray, cdf: np.ndarray, p: float) -> float:
    return float(np.interp(p, cdf, K))


def prob_below(res: RNDResult, level: float) -> float:
    """P(S_T < level·S_0)."""
    return float(np.interp(level * res.spot, res.strikes, res.cdf))


def prob_above(res: RNDResult, level: float) -> float:
    return 1.0 - prob_below(res, level)


def expected_shortfall(res: RNDResult, alpha: float = 0.05) -> tuple[float, float]:
    """(VaR, ES) of the return at confidence ``alpha`` (left tail), as negative numbers."""
    var_price = quantile(res.strikes, res.cdf, alpha)
    mask = res.strikes <= var_price
    K, f = res.strikes[mask], res.density[mask]
    if len(K) < 2:
        return var_price / res.spot - 1, var_price / res.spot - 1
    tail_mass = float(np.trapezoid(f, K))
    es_price = float(np.trapezoid(K * f, K)) / tail_mass if tail_mass > 0 else var_price
    return var_price / res.spot - 1, es_price / res.spot - 1


def risk_metrics(
    res: RNDResult, loss_thresholds=(0.05, 0.10, 0.20), gain_thresholds=(0.05, 0.10)
) -> dict:
    m = {
        "rn_mean_ret": res.stats["mean_ret"],
        "rn_median_ret": res.stats["median_ret"],
        "rn_std": res.stats["std_ret"],
        "expected_move": res.stats["std_ret"],
        "rn_skew": res.stats["skew"],
        "rn_kurtosis": res.stats["kurtosis"],
        "rn_semi_dev": res.stats["semi_dev"],
        "rn_clipped_mass": res.clipped_mass,
    }
    for x in loss_thresholds:
        m[f"p_loss_{int(round(x * 100))}"] = prob_below(res, 1 - x)
    for x in gain_thresholds:
        m[f"p_gain_{int(round(x * 100))}"] = prob_above(res, 1 + x)
    var5, es5 = expected_shortfall(res, 0.05)
    m["var_5"] = var5
    m["es_5"] = es5
    if "p_loss_10" in m and "p_gain_10" in m:
        m["tail_asymmetry"] = m["p_loss_10"] - m["p_gain_10"]
    return m
