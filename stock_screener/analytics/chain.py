"""Option chain cleaning, implied forward, smile and term-structure fitting.

Pipeline for one ticker on one date:

1. :func:`clean_chain` — keep tradeable, fresh, sane contracts; attach the
   maturity ``t`` (years, Actual/365) and log-moneyness ``k = ln(K/F)``.
2. :func:`implied_forward` — per expiry, from put-call parity, else carry.
3. :func:`fit_smile` — per expiry, vega-weighted quadratic in ``k`` with a
   smooth (C1) flattening outside the observed strike range.
4. :class:`SmileSurface` — the fitted smiles as a surface (linear in total
   variance in time, smooth in strike); :func:`build_ql_surface` builds the
   equivalent QuantLib ``BlackVarianceSurface`` used as a consistency check.
5. :func:`chain_metrics` — ATM/constant-maturity IVs, skew, term slope,
   put/call ratios, max pain, liquidity.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .qlpricing import MarketEnv, VolSurface, bs_delta, implied_vol

MIN_CONTRACTS_PER_EXPIRY = 4


@dataclass
class SmileFit:
    expiry: dt.date
    t: float
    forward: float
    coef: tuple[float, float, float]  # a + b k + c k^2
    k_min: float
    k_max: float
    n: int
    rmse: float
    forward_source: str = "carry"
    lam: float = 0.15  # decay length of the slope outside the range

    def iv(self, k: float | np.ndarray) -> np.ndarray:
        a, b, c = self.coef
        k = np.asarray(k, dtype=float)
        out = a + b * k + c * k * k
        # C1 flattening outside [k_min, k_max]
        for edge, sign in ((self.k_max, 1.0), (self.k_min, -1.0)):
            slope = b + 2 * c * edge
            val = a + b * edge + c * edge * edge
            d = sign * (k - edge)
            mask = d > 0
            if np.any(mask):
                out = np.where(
                    mask, val + sign * slope * self.lam * (1 - np.exp(-d / self.lam)), out
                )
        return np.clip(out, 0.03, 4.0)

    @property
    def atm_iv(self) -> float:
        return float(self.iv(0.0))


class SmileSurface:
    """Implied-volatility surface defined by the fitted smiles.

    * strike dimension: each expiry's smooth (C1) quadratic smile in log-moneyness;
    * time dimension: linear interpolation of total variance σ²(k)·T between the
      bracketing expiries, flat vol before the first and after the last expiry.

    The same conventions as QuantLib's ``BlackVarianceSurface`` (bilinear), but
    smooth in strike, which the Breeden–Litzenberger second derivative needs.
    """

    def __init__(self, fits: list[SmileFit]):
        if not fits:
            raise ValueError("at least one fitted smile is required")
        self.fits = sorted(fits, key=lambda f: f.t)
        self.ts = np.array([f.t for f in self.fits])
        self.t_min, self.t_max = float(self.ts[0]), float(self.ts[-1])

    def _var(self, fit: SmileFit, strikes: np.ndarray) -> np.ndarray:
        iv = fit.iv(np.log(np.asarray(strikes, dtype=float) / fit.forward))
        return iv * iv * fit.t

    def vols_at(self, t: float, strikes: np.ndarray) -> np.ndarray:
        strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
        t = max(t, 1e-6)
        if t <= self.t_min:
            return np.sqrt(self._var(self.fits[0], strikes) / self.t_min)
        if t >= self.t_max:
            return np.sqrt(self._var(self.fits[-1], strikes) / self.t_max)
        j = int(np.searchsorted(self.ts, t, side="right"))
        f0, f1 = self.fits[j - 1], self.fits[j]
        w = (t - f0.t) / (f1.t - f0.t)
        var = (1 - w) * self._var(f0, strikes) + w * self._var(f1, strikes)
        return np.sqrt(np.maximum(var, 1e-10) / t)

    def vol(self, t: float, strike: float) -> float:
        return float(self.vols_at(t, np.array([strike]))[0])


@dataclass
class ChainAnalysis:
    spot: float
    q: float
    fits: list[SmileFit]
    surface: SmileSurface | None
    cleaned: pd.DataFrame
    ql_surface: VolSurface | None = None
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------------------


def clean_chain(
    chain: pd.DataFrame,
    env: MarketEnv,
    spot: float,
    q: float,
    min_dte: int = 7,
    max_dte: int = 400,
    solve_missing_iv: bool = True,
    american: bool = False,
    max_solve: int = 400,
) -> pd.DataFrame:
    """Return a cleaned copy with ``t``, ``dte``, ``k``, ``iv_src`` columns."""
    if chain is None or chain.empty:
        return pd.DataFrame()
    df = chain.copy()
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["dte"] = df["expiry"].map(lambda e: (e - env.as_of).days)
    df = df[(df["dte"] >= min_dte) & (df["dte"] <= max_dte)]
    df = df[df["strike"] > 0]
    df["t"] = df["expiry"].map(env.year_fraction)
    # freshness: contract must have been updated on the as-of date (when timestamps exist)
    if "last_updated" in df and df["last_updated"].notna().any():
        lu = pd.to_datetime(df["last_updated"])
        stale = lu.notna() & (lu.dt.date < env.as_of)
        df.loc[stale, "price"] = np.nan
    # liquidity: some open interest or volume
    df["open_interest"] = df["open_interest"].fillna(0)
    df["volume"] = df["volume"].fillna(0)
    df = df[(df["open_interest"] > 0) | (df["volume"] > 0)]
    if df.empty:
        return df
    # forward per expiry (carry only at this stage)
    df["fwd_carry"] = spot * np.exp((env.risk_free_rate - q) * df["t"])
    # intrinsic check on prices (with respect to forward, discounted)
    disc = np.exp(-env.risk_free_rate * df["t"])
    intrinsic = np.where(
        df["type"] == "call", df["fwd_carry"] - df["strike"], df["strike"] - df["fwd_carry"]
    )
    intrinsic = np.maximum(intrinsic, 0) * disc
    bad_price = df["price"].notna() & (df["price"] < intrinsic * 0.98)
    df.loc[bad_price, "price"] = np.nan
    df["iv_src"] = np.where(df["iv"].notna() & (df["iv"] > 0.01), "vendor", None)
    df.loc[df["iv"] <= 0.01, "iv"] = np.nan
    # solve IV from price with QuantLib where the vendor IV is missing
    if solve_missing_iv:
        need = df.index[df["iv"].isna() & df["price"].notna()]
        for idx in list(need)[:max_solve]:
            row = df.loc[idx]
            iv = implied_vol(
                env,
                row["type"],
                spot,
                float(row["strike"]),
                row["expiry"],
                float(row["price"]),
                q=q,
                american=american,
            )
            if iv is not None and 0.02 < iv < 4.0:
                df.loc[idx, "iv"] = iv
                df.loc[idx, "iv_src"] = "quantlib"
    df = df[df["iv"].notna()]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# forward, smile, surface
# --------------------------------------------------------------------------------------


def implied_forward(
    exp_df: pd.DataFrame, spot: float, t: float, r: float, q: float
) -> tuple[float, str]:
    """Put-call parity forward from the strikes nearest to spot; carry fallback."""
    carry = spot * math.exp((r - q) * t)
    px = exp_df.dropna(subset=["price"])
    calls = px[px["type"] == "call"].set_index("strike")["price"]
    puts = px[px["type"] == "put"].set_index("strike")["price"]
    common = calls.index.intersection(puts.index)
    if len(common) < 1:
        return carry, "carry"
    common = sorted(common, key=lambda k: abs(k - spot))[:3]
    fwds = [k + math.exp(r * t) * (calls[k] - puts[k]) for k in common]
    f = float(np.median(fwds))
    # sanity: within ±8 % of carry forward, else fall back
    if not (0.92 * carry <= f <= 1.08 * carry):
        return carry, "carry"
    return f, "parity"


def fit_smile(
    exp_df: pd.DataFrame,
    expiry: dt.date,
    t: float,
    forward: float,
    q: float,
    forward_source: str = "carry",
) -> SmileFit | None:
    """Vega-weighted quadratic fit of IV in log-moneyness, OTM contracts preferred."""
    df = exp_df.copy()
    df["k"] = np.log(df["strike"] / forward)
    otm = ((df["type"] == "call") & (df["k"] >= 0)) | ((df["type"] == "put") & (df["k"] < 0))
    use = df[otm]
    if len(use) < MIN_CONTRACTS_PER_EXPIRY:
        use = df
    if len(use) < 2:
        return None
    # adaptive moneyness band: |k| <= 2.5 sigma sqrt(t) around the median IV
    sig0 = float(use["iv"].median())
    band = 2.5 * sig0 * math.sqrt(t) + 0.02
    use = use[np.abs(use["k"]) <= band]
    if len(use) < 2:
        return None
    k = use["k"].to_numpy()
    iv = use["iv"].to_numpy()
    # vega weights (Black vega is largest ATM), plus liquidity weight
    d1 = (-k + 0.5 * iv**2 * t) / (iv * math.sqrt(t))
    vega_w = np.exp(-0.5 * d1**2)
    liq = np.log1p(use["open_interest"].to_numpy() + use["volume"].to_numpy())
    w = vega_w * (0.5 + liq / (liq.max() + 1e-9))
    if len(use) >= MIN_CONTRACTS_PER_EXPIRY and np.ptp(k) > 0.02:
        A = np.column_stack([np.ones_like(k), k, k * k]) * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, iv * np.sqrt(w), rcond=None)
        a, b, c = (float(x) for x in coef)
        if c < 0:  # keep the smile convex: refit linear
            A = np.column_stack([np.ones_like(k), k]) * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(A, iv * np.sqrt(w), rcond=None)
            a, b, c = float(coef[0]), float(coef[1]), 0.0
    else:
        a, b, c = float(np.average(iv, weights=w)), 0.0, 0.0
    fit = SmileFit(
        expiry=expiry,
        t=t,
        forward=forward,
        coef=(a, b, c),
        k_min=float(k.min()),
        k_max=float(k.max()),
        n=int(len(use)),
        rmse=0.0,
        forward_source=forward_source,
    )
    resid = fit.iv(k) - iv
    fit.rmse = float(np.sqrt(np.average(resid**2, weights=w)))
    return fit


def build_ql_surface(
    env: MarketEnv, fits: list[SmileFit], spot: float, n_strikes: int = 121, width: float = 2.5
) -> VolSurface | None:
    """QuantLib ``BlackVarianceSurface`` on a common strike grid (consistency check)."""
    if not fits:
        return None
    strikes = spot * np.exp(np.linspace(-width, width, n_strikes))
    vols = np.zeros((len(strikes), len(fits)))
    for j, f in enumerate(fits):
        vols[:, j] = f.iv(np.log(strikes / f.forward))
    return VolSurface(env, [f.expiry for f in fits], strikes, vols)


def interp_forward(fits: list[SmileFit], spot: float, t: float, r: float, q: float) -> float:
    """Forward at an arbitrary maturity: linear interpolation of ln(F/S) in t."""
    if not fits:
        return spot * math.exp((r - q) * t)
    ts = np.array([f.t for f in fits])
    lf = np.array([math.log(f.forward / spot) for f in fits])
    if t <= ts[0]:
        return spot * math.exp(lf[0] / ts[0] * t)
    if t >= ts[-1]:
        return spot * math.exp(lf[-1] / ts[-1] * t)
    return spot * math.exp(float(np.interp(t, ts, lf)))


def strike_for_delta(
    fit: SmileFit, target_delta: float, option_type: str, q: float
) -> float | None:
    """Strike where the Black-Scholes delta on the fitted smile equals ``target_delta``."""

    def g(k):
        K = fit.forward * math.exp(k)
        return bs_delta(option_type, fit.forward, K, fit.t, float(fit.iv(k)), q) - target_delta

    lo, hi = -3.0, 3.0
    try:
        if g(lo) * g(hi) > 0:
            return None
        k = brentq(g, lo, hi, xtol=1e-6)
    except (ValueError, RuntimeError):
        return None
    return fit.forward * math.exp(k)


def constant_maturity_iv(
    surface: SmileSurface | VolSurface,
    fits: list[SmileFit],
    spot: float,
    days: int,
    r: float,
    q: float,
) -> float | None:
    if surface is None:
        return None
    t = days / 365.0
    f = interp_forward(fits, spot, t, r, q)
    return surface.vol(t, f)


def _nearest_fit(fits: list[SmileFit], days: int) -> SmileFit | None:
    if not fits:
        return None
    return min(fits, key=lambda f: abs(f.t * 365 - days))


def max_pain(exp_df: pd.DataFrame) -> float | None:
    """Strike minimising the aggregate intrinsic value paid to option holders."""
    df = exp_df.dropna(subset=["open_interest"])
    if df.empty:
        return None
    strikes = np.sort(df["strike"].unique())
    calls = df[df["type"] == "call"]
    puts = df[df["type"] == "put"]
    cost = []
    for s in strikes:
        c = ((s - calls["strike"]).clip(lower=0) * calls["open_interest"]).sum()
        p = ((puts["strike"] - s).clip(lower=0) * puts["open_interest"]).sum()
        cost.append(c + p)
    return float(strikes[int(np.argmin(cost))])


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


def analyse_chain(
    chain: pd.DataFrame,
    env: MarketEnv,
    spot: float,
    q: float,
    horizons: tuple[int, ...] = (30, 60, 90),
    min_dte: int = 7,
    max_dte: int = 400,
    american: bool = False,
) -> ChainAnalysis:
    warnings: list[str] = []
    cleaned = clean_chain(chain, env, spot, q, min_dte=min_dte, max_dte=max_dte, american=american)
    r = env.risk_free_rate
    fits: list[SmileFit] = []
    if cleaned.empty:
        warnings.append("no usable option contracts")
        return ChainAnalysis(spot, q, fits, None, cleaned, None, {}, warnings)
    k_col = np.full(len(cleaned), np.nan)
    for expiry, exp_df in cleaned.groupby("expiry", sort=True):
        t = float(exp_df["t"].iloc[0])
        fwd, src = implied_forward(exp_df, spot, t, r, q)
        fit = fit_smile(exp_df, expiry, t, fwd, q, forward_source=src)
        if fit is None:
            continue
        fits.append(fit)
        k_col[exp_df.index] = np.log(exp_df["strike"] / fwd)
    cleaned["k"] = k_col
    if not fits:
        warnings.append("no expiry with enough contracts to fit a smile")
        return ChainAnalysis(spot, q, fits, None, cleaned, None, {}, warnings)
    fits.sort(key=lambda f: f.t)
    surface = SmileSurface(fits)
    ql_surface = build_ql_surface(env, fits, spot)
    m: dict = {}
    for d in horizons:
        m[f"iv_{d}"] = constant_maturity_iv(surface, fits, spot, d, r, q)
    # term structure and skew from the fits nearest to the primary horizon
    f30 = _nearest_fit(fits, horizons[0])
    f90 = _nearest_fit(fits, horizons[-1]) if len(horizons) > 1 else None
    if m.get(f"iv_{horizons[0]}") and m.get(f"iv_{horizons[-1]}"):
        m["term_slope"] = m[f"iv_{horizons[-1]}"] - m[f"iv_{horizons[0]}"]
    if f30 is not None:
        k25p = strike_for_delta(f30, -0.25, "put", q)
        k25c = strike_for_delta(f30, 0.25, "call", q)
        k10p = strike_for_delta(f30, -0.10, "put", q)
        if k25p and k25c:
            ivp = float(f30.iv(math.log(k25p / f30.forward)))
            ivc = float(f30.iv(math.log(k25c / f30.forward)))
            m["skew_25d"] = ivp - ivc
            m["skew_25d_rel"] = (ivp - ivc) / f30.atm_iv if f30.atm_iv else None
        if k10p:
            m["put_tail_10d"] = float(f30.iv(math.log(k10p / f30.forward))) - f30.atm_iv
        m["implied_forward_30"] = f30.forward
        m["implied_carry"] = math.log(f30.forward / spot) / f30.t - r if f30.t > 0 else None
        m["fit_rmse"] = f30.rmse
        m["forward_source"] = f30.forward_source
        m["nearest_expiry_dte"] = int(round(f30.t * 365))
        m["atm_straddle_move_30"] = _straddle_move(f30, r)
    if f90 is not None and f30 is not None and f90 is not f30:
        m["term_slope_fits"] = f90.atm_iv - f30.atm_iv
    # flows
    puts = cleaned[cleaned["type"] == "put"]
    calls = cleaned[cleaned["type"] == "call"]
    tot_oi_c, tot_oi_p = calls["open_interest"].sum(), puts["open_interest"].sum()
    tot_v_c, tot_v_p = calls["volume"].sum(), puts["volume"].sum()
    m["pc_oi_ratio"] = float(tot_oi_p / tot_oi_c) if tot_oi_c else None
    m["pc_volume_ratio"] = float(tot_v_p / tot_v_c) if tot_v_c else None
    m["option_volume"] = float(tot_v_c + tot_v_p)
    m["open_interest"] = float(tot_oi_c + tot_oi_p)
    m["n_contracts"] = int(len(cleaned))
    m["n_expiries"] = int(len(fits))
    m["iv_source_vendor_share"] = float((cleaned["iv_src"] == "vendor").mean())
    m["liquidity_score"] = float(np.log10(1 + tot_oi_c + tot_oi_p))
    # max pain on the nearest monthly-ish expiry (>= 14 dte, else nearest)
    cand = [f for f in fits if f.t * 365 >= 14] or fits
    mp_exp = min(cand, key=lambda f: f.t).expiry
    m["max_pain"] = max_pain(cleaned[cleaned["expiry"] == mp_exp])
    m["max_pain_dist"] = (m["max_pain"] / spot - 1) if m["max_pain"] else None
    # QuantLib cross-check: both surfaces must agree at the forward for the primary horizon
    if ql_surface is not None and m.get(f"iv_{horizons[0]}"):
        t0 = horizons[0] / 365.0
        f0 = interp_forward(fits, spot, t0, r, q)
        m["ql_surface_diff"] = abs(ql_surface.vol(t0, f0) - surface.vol(t0, f0))
    return ChainAnalysis(spot, q, fits, surface, cleaned, ql_surface, m, warnings)


def _straddle_move(fit: SmileFit, r: float) -> float | None:
    """Expected move from the ATM straddle: (C + P)/F at K = F (Black-Scholes, undiscounted)."""
    from .qlpricing import black_price_from_forward

    sigma = fit.atm_iv
    disc = math.exp(-r * fit.t)
    c = black_price_from_forward("call", fit.forward, fit.forward, fit.t, disc, sigma)
    p = black_price_from_forward("put", fit.forward, fit.forward, fit.t, disc, sigma)
    return float((c + p) / (fit.forward * disc))
