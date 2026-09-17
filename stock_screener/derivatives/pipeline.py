"""Orchestration: market context from the screener cache → products → strategies → parquet."""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..analytics.chain import SmileFit, SmileSurface, interp_forward
from ..analytics.risk_neutral import risk_neutral_density
from ..config import Settings
from ..data.cache import DataCache
from .costs import DEFAULT_COSTS_PATH, DEFAULT_DERIVATIVES_PATH, CostModel, load_yaml
from .evaluation import Evaluator, Samples, recommend, sample_terminal
from .models import Derivative, products_frame, products_from_frame
from .pricing import MarketContext, ko_fair_value, warrant_implied_vol
from .sources import make_source
from .strategies import (
    STRATEGY_CLASSES,
    Strategy,
    barrier_distance,
    bucket_products,
    filter_products,
    sample_realizations,
)

log = logging.getLogger(__name__)


def derivatives_dir(settings: Settings) -> Path:
    p = settings.data_dir / "derivatives"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# market context from the screener cache
# --------------------------------------------------------------------------------------


def load_fits(cache: DataCache, as_of: dt.date, ticker: str) -> list[SmileFit]:
    df = cache.get_frame(as_of, "fits", ticker)
    if df is None or df.empty:
        return []
    return [
        SmileFit(
            expiry=pd.Timestamp(r.expiry).date(),
            t=r.t,
            forward=r.forward,
            coef=(r.a, r.b, r.c),
            k_min=r.k_min,
            k_max=r.k_max,
            n=int(r.n),
            rmse=r.rmse,
            forward_source=r.forward_source,
        )
        for r in df.itertuples()
    ]


def market_context(
    cache: DataCache, as_of: dt.date, ticker: str, row: pd.Series | None, r: float
) -> MarketContext | None:
    bars = cache.get_frame(as_of, "bars", ticker)
    if bars is None or bars.empty:
        return None
    spot = float(bars["close"].iloc[-1])
    q = float(row.get("div_yield_used") or 0.0) if row is not None else 0.0
    fits = load_fits(cache, as_of, ticker)
    if fits:
        surf = SmileSurface(fits)
        return MarketContext(ticker, as_of, spot, r, q, surf.vol, surf.t_min)
    rv = float(row.get("rv_20") or 0.3) if row is not None else 0.3
    return MarketContext(ticker, as_of, spot, r, q, lambda t, k: rv)


def drift_adjustment(cfg_eval: dict, ctx: MarketContext, row: pd.Series | None) -> float:
    """Annualised drift added to the risk-neutral distribution.

    * ``risk_neutral``: 0 — pure market-implied distribution (default)
    * ``fixed``: ``drift_adjustment_annual`` from the config (e.g. an equity premium)
    * ``historical``: the underlying's annualised 6-month momentum replaces the
      risk-neutral drift (capped by ``max_abs_drift_annual``); a crude "trend
      continues" view, not a forecast.
    """
    mode = str(cfg_eval.get("drift_mode", "risk_neutral"))
    cap = float(cfg_eval.get("max_abs_drift_annual", 0.5))
    if mode == "fixed":
        adj = float(cfg_eval.get("drift_adjustment_annual", 0.0))
    elif (
        mode == "historical"
        and row is not None
        and row.get("ret_6m") is not None
        and not pd.isna(row.get("ret_6m"))
    ):
        hist = math.log1p(float(row["ret_6m"])) * 2.0
        adj = hist - (ctx.r - ctx.q)
    else:
        adj = 0.0
    return float(np.clip(adj, -cap, cap))


def terminal_samples(
    ctx: MarketContext, cache: DataCache, horizon_days: int, cfg_eval: dict, drift_adj: float = 0.0
) -> Samples:
    t = horizon_days / 365.0
    fits = load_fits(cache, ctx.as_of, ctx.ticker)
    if fits:
        surf = SmileSurface(fits)
        fwd = interp_forward(fits, ctx.spot, t, ctx.r, ctx.q)
        rnd = risk_neutral_density(surf, ctx.spot, fwd, t, ctx.r)
        grid, cdf = rnd.strikes, rnd.cdf
    else:  # lognormal fallback with the flat vol of the context
        sig = ctx.vol(t, ctx.spot)
        z = np.linspace(-6, 6, 1201)
        grid = ctx.spot * np.exp((ctx.r - ctx.q - 0.5 * sig**2) * t + sig * math.sqrt(t) * z)
        from scipy.stats import norm

        cdf = norm.cdf(z)
    return sample_terminal(
        grid,
        cdf,
        int(cfg_eval.get("n_samples", 20000)),
        t,
        drift_adj,
        int(cfg_eval.get("seed", 42)),
    )


# --------------------------------------------------------------------------------------
# product enrichment
# --------------------------------------------------------------------------------------


def enrich_products(products: list[Derivative], ctx: MarketContext, fx: float) -> pd.DataFrame:
    df = products_frame(products, ctx.as_of)
    if df.empty:
        return df
    fair, prem, iv, dist, delta = [], [], [], [], []
    for p in products:
        if p.is_ko:
            fv = ko_fair_value(p, ctx, fx)
            fair.append(fv)
            prem.append(p.ask / fv - 1 if fv > 0 else np.nan)
            iv.append(np.nan)
            dist.append((p.barrier / ctx.spot - 1) if p.barrier else np.nan)
            delta.append(1.0 if p.product_type == "ko_long" else -1.0)
        else:
            v = warrant_implied_vol(p, ctx, fx)
            iv.append(v if v is not None else np.nan)
            T = (p.maturity - ctx.as_of).days / 365.0
            surf_vol = ctx.vol(T, p.strike)
            from ..analytics.qlpricing import bs_delta, bs_price

            typ = "call" if p.product_type == "call_warrant" else "put"
            fv = bs_price(typ, ctx.spot, p.strike, T, ctx.r, ctx.q, surf_vol) * p.ratio / fx
            fair.append(fv)
            prem.append(p.ask / fv - 1 if fv > 0 else np.nan)
            dist.append(p.strike / ctx.spot - 1)
            fwd = ctx.spot * math.exp((ctx.r - ctx.q) * T)
            delta.append(bs_delta(typ, fwd, p.strike, T, v or surf_vol, ctx.q))
    df["spot"] = ctx.spot
    df["fair_value"] = fair
    df["premium_pct"] = prem
    df["implied_vol"] = iv
    df["surface_vol"] = [
        ctx.vol((p.maturity - ctx.as_of).days / 365.0, p.strike)
        if p.maturity and not p.is_ko
        else np.nan
        for p in products
    ]
    df["distance_pct"] = dist
    df["delta"] = delta
    df["barrier_distance"] = [
        barrier_distance(p, ctx.spot) if p.is_ko else np.nan for p in products
    ]
    return df


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------


class DerivativesRun:
    def __init__(
        self,
        settings: Settings,
        costs_path: Path | None = None,
        config_path: Path | None = None,
        source_kind: str | None = None,
    ):
        self.settings = settings
        self.costs = CostModel.load(costs_path or DEFAULT_COSTS_PATH)
        self.cfg = load_yaml(config_path or DEFAULT_DERIVATIVES_PATH)
        self.cache = DataCache(settings.data_dir)
        self.screener = self.cache.get_screener()
        if self.screener is None or self.screener.empty:
            raise RuntimeError("No screener table found — run `screener refresh` first")
        self.as_of = dt.date.fromisoformat(str(self.screener["as_of"].iloc[0]))
        self.source = make_source(
            self.cfg, self.costs.fx_eur_usd, self.as_of, self.costs.issuer_defaults, source_kind
        )
        self.horizon = int(self.cfg["evaluation"].get("horizon_days", 30))
        self.warnings: list[str] = []

    def row(self, ticker: str) -> pd.Series | None:
        m = self.screener[self.screener["ticker"] == ticker]
        return m.iloc[0] if len(m) else None

    def context(self, ticker: str) -> MarketContext | None:
        return market_context(
            self.cache, self.as_of, ticker, self.row(ticker), self.settings.risk_free_rate
        )

    def evaluator(self, ctx: MarketContext) -> Evaluator:
        adj = drift_adjustment(self.cfg["evaluation"], ctx, self.row(ctx.ticker))
        samples = terminal_samples(ctx, self.cache, self.horizon, self.cfg["evaluation"], adj)
        return Evaluator(ctx, self.costs, samples, self.cfg["evaluation"])

    def run(
        self, tickers: list[str] | None = None, progress=None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        tickers = tickers or self.screener["ticker"].tolist()
        product_frames, rows = [], []
        for i, t in enumerate(tickers, 1):
            ctx = self.context(t)
            if ctx is None:
                self.warnings.append(f"{t}: no market context (bars missing)")
                continue
            try:
                products = self.source.list_products(t, ctx)
            except Exception as exc:
                self.warnings.append(f"{t}: product query failed: {exc}")
                products = []
            if products:
                product_frames.append(enrich_products(products, ctx, self.costs.fx_eur_usd))
                usable = filter_products(products, ctx, self.cfg.get("filters", {}), self.horizon)
                buckets = bucket_products(usable, ctx, self.cfg.get("sampling", {}))
                reals = sample_realizations(buckets, ctx, self.cfg.get("sampling", {}))
                ev = self.evaluator(ctx)
                for s in reals:
                    rows.append(ev.metrics(s))
            if progress:
                progress(i, len(tickers), t, len(products))
        products_df = (
            pd.concat(product_frames, ignore_index=True) if product_frames else pd.DataFrame()
        )
        strategies_df = pd.DataFrame(recommend(rows, self.cfg["evaluation"]))
        out = derivatives_dir(self.settings)
        products_df.to_parquet(out / "products.parquet", index=False)
        strategies_df.to_parquet(out / "strategies.parquet", index=False)
        meta = {
            "as_of": self.as_of.isoformat(),
            "source": self.source.name,
            "horizon_days": self.horizon,
            "budget_eur": self.cfg["evaluation"].get("budget_eur"),
            "costs": self.costs.describe(),
            "drift_mode": self.cfg["evaluation"].get("drift_mode", "risk_neutral"),
            "drift_adjustment_annual": self.cfg["evaluation"].get("drift_adjustment_annual", 0.0),
            "risk_aversion": self.cfg["evaluation"].get("risk_aversion"),
            "warnings": self.warnings + list(getattr(self.source, "warnings", [])),
        }
        import json

        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        return products_df, strategies_df

    # ---- used by the dashboard detail view --------------------------------------------
    def rebuild_strategy(self, row: dict, products_df: pd.DataFrame) -> Strategy | None:
        isins = str(row["legs"]).split("|")
        sub = products_df[
            (products_df["underlying"] == row["underlying"]) & (products_df["isin"].isin(isins))
        ]
        legs_by_isin = {p.isin: p for p in products_from_frame(sub)}
        legs = [legs_by_isin[i] for i in isins if i in legs_by_isin]
        if len(legs) != len(isins):
            return None
        n = len(legs)
        cls = row["strategy_class"]
        weights = [0.7, 0.3] if cls in ("ko_long_put_hedge", "ko_short_call_hedge") else [1 / n] * n
        return Strategy(
            cls,
            row["underlying"],
            legs,
            weights,
            float(row["param"]),
            label=row.get("label", ""),
            id=row["id"],
        )


def class_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": c.key,
                "label": c.label,
                "description": c.description,
                "param": c.param_name,
                "max_legs": c.max_legs,
            }
            for c in STRATEGY_CLASSES.values()
        ]
    )
