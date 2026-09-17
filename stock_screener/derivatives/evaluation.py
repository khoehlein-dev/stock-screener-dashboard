"""Probabilistic evaluation of strategies against the option-implied distribution.

For each underlying the terminal price at the horizon is sampled from the
risk-neutral density of the screener (inverse-CDF sampling with stratified
uniforms), optionally shifted by a configurable annual drift. Knock-out events
are drawn with the Brownian-bridge crossing probability (common random numbers
across products). Costs follow :class:`CostModel`. All metrics are relative to
the invested capital (including entry fees).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .costs import CostModel
from .models import Derivative
from .pricing import MarketContext, ko_payoff_curve, ko_values, warrant_implied_vol, warrant_values
from .strategies import STRATEGY_CLASSES, Strategy


@dataclass
class Samples:
    S_T: np.ndarray  # terminal underlying prices (underlying currency)
    U: np.ndarray  # uniforms for knock-out draws
    t: float  # horizon in years
    grid: np.ndarray  # density grid (prices)
    density: np.ndarray  # density of S_T on the grid


def sample_terminal(
    grid: np.ndarray, cdf: np.ndarray, n: int, t: float, drift_adj: float, seed: int
) -> Samples:
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + rng.uniform(0, 1, n)) / n  # stratified
    rng.shuffle(u)
    S_T = np.interp(u, cdf, grid) * math.exp(drift_adj * t)
    U = rng.uniform(0, 1, n)
    density = np.gradient(cdf, grid)
    return Samples(S_T=S_T, U=U, t=t, grid=grid * math.exp(drift_adj * t), density=density)


class Evaluator:
    def __init__(self, ctx: MarketContext, costs: CostModel, samples: Samples, cfg: dict):
        self.ctx = ctx
        self.costs = costs
        self.samples = samples
        self.cfg = cfg
        self.budget = float(cfg.get("budget_eur", 1000.0))
        self.alpha = float(cfg.get("es_alpha", 0.05))
        self.risk_aversion = float(cfg.get("risk_aversion", 0.5))
        self.fx = costs.fx_eur_usd
        ex = costs.exit
        self.spread_mult = float(ex.get("spread_multiplier", 1.0))
        self.min_half_spread = float(ex.get("min_half_spread", 0.0))
        self.recovery = float(ex.get("knockout_recovery", 0.0))
        idf = costs.issuer_defaults
        self.fin_long = float(idf.get("ko_financing_rate_long", 0.05))
        self.fin_short = float(idf.get("ko_financing_rate_short", 0.01))
        self._iv_cache: dict[str, float | None] = {}
        self._leg_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ---- leg valuation --------------------------------------------------------------
    def leg_sigma(self, leg: Derivative) -> float | None:
        if leg.isin not in self._iv_cache:
            iv = warrant_implied_vol(leg, self.ctx, self.fx)
            if iv is None:  # fall back to the underlying's surface vol
                T = (leg.maturity - self.ctx.as_of).days / 365.0 if leg.maturity else self.samples.t
                iv = self.ctx.vol(T, leg.strike)
            self._iv_cache[leg.isin] = iv
        return self._iv_cache[leg.isin]

    def leg_values(
        self, leg: Derivative, S_T: np.ndarray, U: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(value per unit in EUR at horizon, knocked-out indicator)."""
        shared = S_T is self.samples.S_T and U is None
        if shared and leg.isin in self._leg_cache:
            return self._leg_cache[leg.isin]
        t = self.samples.t
        if leg.is_ko:
            U = self.samples.U if U is None else U
            res = ko_values(
                leg, S_T, U, t, self.ctx, self.fx, self.recovery, self.fin_long, self.fin_short
            )
        else:
            res = (
                warrant_values(leg, S_T, t, self.leg_sigma(leg), self.ctx, self.fx),
                np.zeros(len(S_T), dtype=bool),
            )
        if shared:
            self._leg_cache[leg.isin] = res
        return res

    def exit_price(self, leg: Derivative, value: np.ndarray) -> np.ndarray:
        half = np.maximum(value * leg.spread_pct * self.spread_mult / 2, self.min_half_spread)
        return np.where(value > 0, np.maximum(value - half, 0.0), 0.0)

    # ---- strategy evaluation ------------------------------------------------------------
    def position(self, strategy: Strategy) -> tuple[list[int], float, float]:
        """Quantities, invested capital (incl. entry fees), entry fees."""
        qs, invested, fees = [], 0.0, 0.0
        for leg, w in zip(strategy.legs, strategy.weights, strict=True):
            q = int(math.floor(w * self.budget / leg.ask))
            qs.append(q)
            value = q * leg.ask
            fee = self.costs.order_fee(value, leg.issuer) if q > 0 else 0.0
            invested += value + fee
            fees += fee
        return qs, invested, fees

    def pnl(
        self, strategy: Strategy, S_T: np.ndarray | None = None, U: np.ndarray | None = None
    ) -> dict:
        S_T = self.samples.S_T if S_T is None else S_T
        qs, invested, entry_fees = self.position(strategy)
        if any(q <= 0 for q in qs) or invested <= 0:
            return {"feasible": False}
        proceeds = np.zeros(len(S_T))
        exit_fees = np.zeros(len(S_T))
        knocked_any = np.zeros(len(S_T), dtype=bool)
        for leg, q in zip(strategy.legs, qs, strict=True):
            value, knocked = self.leg_values(leg, S_T, U)
            px = self.exit_price(leg, value)
            gross = q * px
            proceeds += gross
            fee_vec = (
                np.array(
                    [
                        self.costs.order_fee(g, leg.issuer) if g > 0 else 0.0
                        for g in np.round(gross, 2)
                    ]
                )
                if len(gross) <= 4096
                else self._fee_fast(gross, leg.issuer)
            )
            exit_fees += fee_vec
            knocked_any |= knocked
        pnl = proceeds - exit_fees - invested
        tax = self.costs.tax_rate
        if tax > 0:
            pnl = np.where(pnl > 0, pnl * (1 - tax), pnl)
        return {
            "feasible": True,
            "pnl": pnl,
            "ret": pnl / invested,
            "invested": invested,
            "entry_fees": entry_fees,
            "exit_fees_mean": float(exit_fees.mean()),
            "quantities": qs,
            "knocked": knocked_any,
        }

    def _fee_fast(self, gross: np.ndarray, issuer: str) -> np.ndarray:
        """Vectorised order fee: piecewise on a few breakpoints (fees are step/linear functions)."""
        probe = (
            np.unique(np.round(np.quantile(gross[gross > 0], np.linspace(0, 1, 41)), 2))
            if (gross > 0).any()
            else np.array([0.0])
        )
        fees_probe = np.array([self.costs.order_fee(g, issuer) for g in probe])
        out = (
            np.interp(gross, probe, fees_probe)
            if len(probe) > 1
            else np.full_like(gross, fees_probe[0])
        )
        return np.where(gross > 0, out, 0.0)

    def metrics(self, strategy: Strategy) -> dict:
        res = self.pnl(strategy)
        cls = STRATEGY_CLASSES[strategy.strategy_class]
        base = {
            "id": strategy.id,
            "strategy_class": strategy.strategy_class,
            "class_label": cls.label,
            "underlying": strategy.underlying,
            "label": strategy.label,
            "n_legs": strategy.n_legs,
            "legs": "|".join(leg.isin for leg in strategy.legs),
            "param": strategy.param,
            "param_name": cls.param_name,
            "feasible": res["feasible"],
        }
        if not res["feasible"]:
            return base
        r = res["ret"]
        a = self.alpha
        var = float(np.quantile(r, a))
        tail = r[r <= var]
        es = float(tail.mean()) if len(tail) else var
        mean = float(r.mean())
        pos, neg = r[r > 0], r[r < 0]
        omega = float(pos.sum() / -neg.sum()) if len(neg) and neg.sum() < 0 else float("inf")
        # effective leverage: regression slope of return on underlying return around the centre
        u_ret = self.samples.S_T / self.ctx.spot - 1
        mask = np.abs(u_ret) < 0.1
        lev = float(np.polyfit(u_ret[mask], r[mask], 1)[0]) if mask.sum() > 50 else float("nan")
        be_up, be_down = self.break_even(strategy)
        base.update(
            {
                "invested": res["invested"],
                "entry_fees": res["entry_fees"],
                "exit_fees": res["exit_fees_mean"],
                "cost_drag": (res["entry_fees"] + res["exit_fees_mean"]) / res["invested"],
                "exp_return": mean,
                "median_return": float(np.median(r)),
                "std_return": float(r.std()),
                "p_profit": float((r > 0).mean()),
                "p_loss_50": float((r < -0.5).mean()),
                "p_total_loss": float((r <= -0.99).mean()),
                "p_knockout": float(res["knocked"].mean()),
                "var_5": var,
                "es_5": es,
                "gain_95": float(np.quantile(r, 0.95)),
                "gain_99": float(np.quantile(r, 0.99)),
                "omega": omega,
                "sharpe": mean / float(r.std()) if r.std() > 0 else float("nan"),
                "utility": mean - self.risk_aversion * abs(min(es, 0.0)),
                "effective_leverage": lev,
                "break_even_up": be_up,
                "break_even_down": be_down,
                "quantities": "|".join(str(q) for q in res["quantities"]),
            }
        )
        return base

    # ---- deterministic payoff along the terminal price --------------------------------
    def payoff_curve(self, strategy: Strategy, n: int = 241, width: float = 0.5) -> dict:
        S0 = self.ctx.spot
        grid = S0 * np.linspace(1 - width, 1 + width, n)
        qs, invested, _ = self.position(strategy)
        det = np.zeros(n)
        exp_val = np.zeros(n)
        for leg, q in zip(strategy.legs, qs, strict=True):
            if leg.is_ko:
                alive, adj = ko_payoff_curve(
                    leg,
                    grid,
                    self.samples.t,
                    self.ctx,
                    self.fx,
                    self.recovery,
                    self.fin_long,
                    self.fin_short,
                )
                det += q * self.exit_price(leg, alive)
                exp_val += q * self.exit_price(leg, adj)
            else:
                v = warrant_values(
                    leg, grid, self.samples.t, self.leg_sigma(leg), self.ctx, self.fx
                )
                px = q * self.exit_price(leg, v)
                det += px
                exp_val += px
        fees = np.array(
            [
                sum(self.costs.order_fee(x, leg.issuer) for leg in strategy.legs) if x > 0 else 0.0
                for x in det / max(len(strategy.legs), 1)
            ]
        )
        return {
            "grid": grid,
            "pnl": det - fees - invested,
            "pnl_ko_adjusted": exp_val - fees - invested,
            "invested": invested,
        }

    def break_even(self, strategy: Strategy) -> tuple[float | None, float | None]:
        c = self.payoff_curve(strategy)
        x = c["grid"] / self.ctx.spot - 1
        y = c["pnl"]
        up = down = None
        for i in range(len(x) - 1):
            if y[i] * y[i + 1] < 0:
                xb = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
                if xb >= 0 and up is None:
                    up = float(xb)
                if xb < 0:
                    down = float(xb)
        return up, down


# --------------------------------------------------------------------------------------
# recommendation and stability
# --------------------------------------------------------------------------------------


def recommend(rows: list[dict], cfg: dict) -> list[dict]:
    """Rank realizations within (class, underlying); flag the best and quantify stability."""
    import pandas as pd

    if not rows:
        return rows
    df = pd.DataFrame(rows)
    max_tl = float(cfg.get("max_total_loss_probability", 1.0))
    k = int(cfg.get("neighbours_for_stability", 5))
    df["eligible"] = df["feasible"] & (df.get("p_total_loss", 0).fillna(1) <= max_tl)
    df["rank"] = np.nan
    df["is_best"] = False
    df["stability"] = np.nan
    df["utility_gap"] = np.nan
    for (_, _), g in df.groupby(["underlying", "strategy_class"]):
        el = g[g["eligible"]].sort_values("utility", ascending=False)
        if el.empty:
            continue
        df.loc[el.index, "rank"] = np.arange(1, len(el) + 1)
        best = el.index[0]
        df.loc[best, "is_best"] = True
        if len(el) > 1:
            df.loc[best, "utility_gap"] = float(el["utility"].iloc[0] - el["utility"].iloc[1])
        # stability: dispersion of utility among the k nearest neighbours along the class parameter
        gp = g[g["feasible"]].sort_values("param")
        pos = int(np.where(gp.index == best)[0][0]) if best in gp.index else None
        if pos is not None and len(gp) > 1:
            lo, hi = max(0, pos - k), min(len(gp), pos + k + 1)
            neigh = gp.iloc[lo:hi]["utility"].to_numpy()
            scale = max(abs(float(gp.loc[best, "utility"])), float(np.abs(neigh).max()), 1e-3)
            df.loc[best, "stability"] = float(np.clip(1 - neigh.std() / scale, 0, 1))
    return df.to_dict("records")
