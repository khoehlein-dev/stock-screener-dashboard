"""Derivatives tab: product shelves, strategy grid and strategy detail view.

Reads ``data_dir/derivatives/{products,strategies}.parquet`` written by
``screener derivatives refresh`` and re-evaluates a selected strategy on demand
for the detail charts (cheap: one strategy against the cached samples).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import dash_ag_grid as dag
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from plotly.subplots import make_subplots

from ..config import Settings
from ..derivatives.evaluation import Evaluator
from ..derivatives.pipeline import DerivativesRun, class_table
from ..derivatives.strategies import STRATEGY_CLASSES

PAL = {
    "s1": "#2a78d6",
    "s2": "#eb6834",
    "s3": "#1baf7a",
    "s4": "#eda100",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
    "good": "#006300",
    "bad": "#d03b3b",
    "neutral": "#f0efec",
}

F = {
    "pct": "params.value == null ? '' : d3.format('+.1%')(params.value)",
    "pct2": "params.value == null ? '' : d3.format('+.2%')(params.value)",
    "upct": "params.value == null ? '' : d3.format('.1%')(params.value)",
    "num2": "params.value == null ? '' : d3.format(',.2f')(params.value)",
    "num3": "params.value == null ? '' : d3.format(',.3f')(params.value)",
    "num1": "params.value == null ? '' : d3.format(',.1f')(params.value)",
    "int": "params.value == null ? '' : d3.format(',.0f')(params.value)",
    "eur": "params.value == null ? '' : d3.format(',.2f')(params.value) + ' €'",
    "text": "params.value == null ? '' : params.value",
}


def _col(field, label, fmt="num2", tooltip="", sign=0, width=None, pinned=None):
    d = {
        "field": field,
        "headerName": label,
        "headerTooltip": tooltip,
        "valueFormatter": {"function": F[fmt]},
        "filter": "agTextColumnFilter" if fmt == "text" else "agNumberColumnFilter",
        "sortable": True,
        "resizable": True,
    }
    if fmt != "text":
        d["type"] = "rightAligned"
    if width:
        d["width"] = width
    if pinned:
        d["pinned"] = pinned
    if sign:
        pos, neg = (PAL["good"], PAL["bad"]) if sign > 0 else (PAL["bad"], PAL["good"])
        d["cellStyle"] = {
            "styleConditions": [
                {"condition": "params.value > 0", "style": {"color": pos}},
                {"condition": "params.value < 0", "style": {"color": neg}},
            ]
        }
    return d


KO_COLUMNS = [
    _col("underlying", "Underlying", "text", pinned="left", width=100),
    _col("name", "Product", "text", width=230),
    _col("issuer", "Issuer", "text", width=120),
    _col("wkn", "WKN", "text", width=90),
    _col("isin", "ISIN", "text", width=130),
    _col("product_type", "Type", "text", width=90),
    _col("spot", "Spot", "num2", "Underlying price (USD)"),
    _col("strike", "Strike", "num2"),
    _col("barrier", "Barrier", "num2", "Knock-out level"),
    _col("distance_pct", "Barrier dist.", "pct", "Barrier / spot − 1"),
    _col("ratio", "Ratio", "num3"),
    _col("bid", "Bid", "num3"),
    _col("ask", "Ask", "num3"),
    _col("spread_pct", "Spread", "upct", "(ask − bid) / mid"),
    _col("fair_value", "Fair value", "num3", "Intrinsic value in EUR"),
    _col("premium_pct", "Premium", "upct", "Ask / fair value − 1"),
    _col("leverage", "Leverage", "num1"),
    _col("financing_rate", "Financing", "upct", "Annual financing rate (open-end)"),
    _col("maturity", "Maturity", "text", "Empty = open-end", width=110),
    _col("quote_time", "Quote", "text", width=150),
]
WARRANT_COLUMNS = [
    _col("underlying", "Underlying", "text", pinned="left", width=100),
    _col("name", "Product", "text", width=230),
    _col("issuer", "Issuer", "text", width=120),
    _col("wkn", "WKN", "text", width=90),
    _col("isin", "ISIN", "text", width=130),
    _col("product_type", "Type", "text", width=100),
    _col("spot", "Spot", "num2", "Underlying price (USD)"),
    _col("strike", "Strike", "num2"),
    _col("distance_pct", "Moneyness", "pct", "Strike / spot − 1"),
    _col("maturity", "Maturity", "text", width=110),
    _col("days_to_maturity", "DTM", "int", "Days to maturity"),
    _col("ratio", "Ratio", "num3"),
    _col("bid", "Bid", "num3"),
    _col("ask", "Ask", "num3"),
    _col("spread_pct", "Spread", "upct"),
    _col("implied_vol", "Impl. vol", "upct", "Implied vol of the warrant mid (QuantLib)"),
    _col(
        "surface_vol",
        "Stock IV",
        "upct",
        "Vol of the listed-option surface at the same strike/maturity",
    ),
    _col("premium_pct", "Vol premium", "upct", "Ask / fair value at stock IV − 1"),
    _col("delta", "Delta", "num2"),
    _col("leverage", "Leverage", "num1"),
]
STRATEGY_COLUMNS = [
    _col("underlying", "Underlying", "text", pinned="left", width=100),
    _col("class_label", "Strategy", "text", width=170),
    _col("label", "Instruments", "text", width=330),
    _col("n_legs", "Legs", "int", width=70),
    _col("param", "Param", "num3", "Class parameter (see tooltip of Strategy)"),
    _col("rank", "Rank", "int", "Rank within class and underlying by utility"),
    _col("invested", "Invested", "eur", "Capital incl. entry fees"),
    _col("exp_return", "E[return]", "pct", "Expected return on invested capital at the horizon", 1),
    _col("median_return", "Median", "pct", "", 1),
    _col("p_profit", "P(profit)", "upct"),
    _col("p_knockout", "P(KO)", "upct", "Probability that at least one leg is knocked out"),
    _col("p_total_loss", "P(total loss)", "upct"),
    _col("var_5", "VaR 5%", "pct", "5th percentile of return", 1),
    _col("es_5", "ES 5%", "pct", "Expected shortfall below the 5th percentile", 1),
    _col("gain_95", "95th pct", "pct", "", 1),
    _col("omega", "Omega", "num2", "E[gains] / E[losses]"),
    _col("utility", "Utility", "pct2", "E[return] − risk aversion × |ES|", 1),
    _col(
        "stability",
        "Stability",
        "upct",
        "1 − dispersion of utility among neighbouring realizations (best only)",
    ),
    _col("utility_gap", "Gap to 2nd", "pct2", "Utility advantage over the runner-up (best only)"),
    _col("cost_drag", "Cost drag", "upct", "(entry + expected exit fees) / invested"),
    _col(
        "effective_leverage",
        "Eff. lev.",
        "num2",
        "Slope of strategy return vs underlying return (±10 %)",
    ),
    _col("break_even_up", "BE up", "pct", "Underlying move at which P&L crosses zero (upside)"),
    _col(
        "break_even_down", "BE down", "pct", "Underlying move at which P&L crosses zero (downside)"
    ),
    _col("legs", "ISINs", "text", width=260),
]


class DerivativesStore:
    def __init__(self, settings: Settings, data_dir: Path):
        self.settings = settings
        self.dir = Path(data_dir) / "derivatives"
        self.products = pd.DataFrame()
        self.strategies = pd.DataFrame()
        self.meta: dict = {}
        self._run: DerivativesRun | None = None
        self._evaluators: dict[str, Evaluator] = {}
        self.reload()

    def reload(self) -> None:
        p, s, m = (
            self.dir / "products.parquet",
            self.dir / "strategies.parquet",
            self.dir / "meta.json",
        )
        self.products = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        self.strategies = pd.read_parquet(s) if s.exists() else pd.DataFrame()
        self.meta = json.loads(m.read_text()) if m.exists() else {}
        self._run = None
        self._evaluators = {}

    @property
    def run(self) -> DerivativesRun | None:
        if self._run is None:
            try:
                self._run = DerivativesRun(self.settings)
            except Exception:
                self._run = None
        return self._run

    def evaluator(self, ticker: str) -> Evaluator | None:
        if ticker not in self._evaluators and self.run is not None:
            ctx = self.run.context(ticker)
            self._evaluators[ticker] = self.run.evaluator(ctx) if ctx else None
        return self._evaluators.get(ticker)

    def records(self, df: pd.DataFrame) -> list[dict]:
        return df.replace({np.nan: None}).to_dict("records") if not df.empty else []


# --------------------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------------------


def _grid(id_, cols, height="52vh"):
    return dag.AgGrid(
        id=id_,
        rowData=[],
        columnDefs=cols,
        defaultColDef={
            "sortable": True,
            "filter": True,
            "resizable": True,
            "minWidth": 80,
            "wrapHeaderText": True,
            "autoHeaderHeight": True,
        },
        dashGridOptions={
            "rowSelection": "single",
            "animateRows": False,
            "tooltipShowDelay": 300,
            "enableCellTextSelection": True,
            "suppressMenuHide": True,
            "rowHeight": 30,
        },
        columnSize="autoSize",
        style={"height": height, "width": "100%"},
        csvExportParams={"fileName": f"{id_}.csv"},
    )


def layout(store: DerivativesStore) -> html.Div:
    m = store.meta
    status = (
        f"Derivatives as of {m.get('as_of')} · source: {m.get('source')} · horizon {m.get('horizon_days')} d · "
        f"budget {m.get('budget_eur')} € · drift: {m.get('drift_mode')} · {m.get('costs')}"
        if m
        else "No derivatives data — run `screener derivatives refresh`."
    )
    underlyings = (
        sorted(store.strategies["underlying"].unique()) if not store.strategies.empty else []
    )
    classes = class_table()
    return html.Div(
        [
            html.Div(status, className="status", id="d-status", style={"margin": "4px 0 8px"}),
            html.Div(
                "Derivative products are complex, leveraged instruments; knock-outs can lose their entire value. "
                "Nothing here is financial advice; the evaluation is model-based and uses unverified fee assumptions "
                "(config/costs.yaml). No orders are placed from this dashboard.",
                className="warn",
                style={"marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Label(
                        "Underlying",
                        className="flabel",
                        style={"display": "inline-block", "marginRight": "8px"},
                    ),
                    dcc.Dropdown(
                        id="d-underlying",
                        options=[{"label": u, "value": u} for u in underlyings],
                        multi=True,
                        placeholder="All underlyings",
                        style={
                            "minWidth": "320px",
                            "display": "inline-block",
                            "verticalAlign": "middle",
                        },
                    ),
                    dcc.Checklist(
                        id="d-show-all",
                        options=[
                            {
                                "label": " Show all realizations (not only the best per class)",
                                "value": "all",
                            }
                        ],
                        value=[],
                        style={"display": "inline-block", "marginLeft": "16px"},
                        labelStyle={"display": "flex", "alignItems": "center", "gap": "6px"},
                    ),
                    html.Button(
                        "Reload", id="d-reload", className="btn", style={"marginLeft": "12px"}
                    ),
                ],
                className="toolbar",
            ),
            dcc.Tabs(
                id="d-tabs",
                value="strategies",
                children=[
                    dcc.Tab(
                        label="Strategies",
                        value="strategies",
                        children=[
                            html.Details(
                                [
                                    html.Summary("Strategy classes"),
                                    html.Table(
                                        [
                                            html.Tr(
                                                [
                                                    html.Th("Class"),
                                                    html.Th("Description"),
                                                    html.Th("Parameter"),
                                                    html.Th("Max legs"),
                                                ]
                                            )
                                        ]
                                        + [
                                            html.Tr(
                                                [
                                                    html.Td(r.label),
                                                    html.Td(r.description),
                                                    html.Td(r.param),
                                                    html.Td(str(r.max_legs)),
                                                ]
                                            )
                                            for r in classes.itertuples()
                                        ],
                                        className="classes",
                                    ),
                                ],
                                style={"margin": "8px 0"},
                            ),
                            _grid("d-strategies", STRATEGY_COLUMNS, "48vh"),
                            html.Div(
                                id="d-detail",
                                className="detail",
                                children=html.Div(
                                    "Select a strategy to see its evaluation.", className="note"
                                ),
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Knock-out certificates",
                        value="ko",
                        children=[_grid("d-ko", KO_COLUMNS, "70vh")],
                    ),
                    dcc.Tab(
                        label="Warrants",
                        value="warrants",
                        children=[_grid("d-warrants", WARRANT_COLUMNS, "70vh")],
                    ),
                ],
            ),
        ]
    )


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def _base(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": PAL["ink"]}, "x": 0.01},
        height=height,
        margin={"l": 56, "r": 16, "t": 40, "b": 44},
        paper_bgcolor=PAL["surface"],
        plot_bgcolor=PAL["surface"],
        font={"color": PAL["ink2"], "size": 12},
        legend={"orientation": "h", "y": -0.2, "x": 0, "font": {"size": 11}},
    )
    fig.update_xaxes(gridcolor=PAL["grid"], linecolor=PAL["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=PAL["grid"], linecolor=PAL["axis"], zeroline=False)
    return fig


def payoff_figure(ev: Evaluator, strategy, title: str) -> go.Figure:
    c = ev.payoff_curve(strategy)
    x = c["grid"] / ev.ctx.spot - 1
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=c["pnl"],
            name="P&L if not knocked out",
            line={"color": PAL["s1"], "width": 2},
            hovertemplate="move %{x:+.1%}<br>P&L %{y:,.0f} €<extra></extra>",
        ),
        1,
        1,
    )
    if not np.allclose(c["pnl"], c["pnl_ko_adjusted"]):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=c["pnl_ko_adjusted"],
                name="expected P&L incl. knock-out probability",
                line={"color": PAL["s2"], "width": 2, "dash": "dash"},
                hovertemplate="move %{x:+.1%}<br>E[P&L] %{y:,.0f} €<extra></extra>",
            ),
            1,
            1,
        )
    fig.add_hline(y=0, line={"color": PAL["muted"], "width": 1}, row=1, col=1)
    fig.add_hline(
        y=-c["invested"],
        line={"color": PAL["bad"], "width": 1, "dash": "dot"},
        row=1,
        col=1,
        annotation_text="total loss",
        annotation_position="bottom right",
    )
    gx = ev.samples.grid / ev.ctx.spot - 1
    mask = (gx > x.min()) & (gx < x.max())
    fig.add_trace(
        go.Scatter(
            x=gx[mask],
            y=ev.samples.density[mask] * ev.ctx.spot,
            name="underlying distribution at horizon",
            fill="tozeroy",
            line={"color": PAL["s3"], "width": 1.5},
            fillcolor="rgba(27,175,122,0.18)",
            hovertemplate="move %{x:+.1%}<extra></extra>",
        ),
        2,
        1,
    )
    fig.add_vline(x=0, line={"color": PAL["muted"], "dash": "dot", "width": 1})
    _base(fig, title, 420)
    fig.update_xaxes(title_text="Underlying move at horizon", tickformat="+.0%", row=2, col=1)
    fig.update_yaxes(title_text="P&L (EUR)", row=1, col=1)
    fig.update_yaxes(title_text="Density", showticklabels=False, row=2, col=1)
    fig.update_layout(hovermode="x unified")
    return fig


def distribution_figure(res: dict, metrics: dict, title: str) -> go.Figure:
    r = res["ret"]
    lo, hi = float(np.quantile(r, 0.002)), float(np.quantile(r, 0.998))
    fig = go.Figure(
        go.Histogram(
            x=np.clip(r, lo, hi),
            nbinsx=80,
            name="return",
            marker_color=PAL["s1"],
            histnorm="probability",
            hovertemplate="return %{x:+.0%}: %{y:.1%}<extra></extra>",
        )
    )
    for key, label, col, pos in (
        ("es_5", "ES 5%", PAL["bad"], "bottom left"),
        ("var_5", "VaR 5%", PAL["s2"], "top left"),
        ("exp_return", "mean", PAL["ink"], "top right"),
    ):
        v = metrics.get(key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            fig.add_vline(
                x=v,
                line={"color": col, "width": 1.5, "dash": "dash"},
                annotation_text=f"{label} {v:+.1%}",
                annotation_position=pos,
                annotation_font={"size": 11, "color": col},
            )
    _base(fig, title, 340)
    fig.update_xaxes(title_text="Return on invested capital", tickformat="+.0%")
    fig.update_yaxes(title_text="Probability")
    fig.update_layout(bargap=0.05, showlegend=False)
    return fig


def tradeoff_figure(group: pd.DataFrame, best_id: str, title: str, param_name: str) -> go.Figure:
    g = group[group["feasible"] == True]  # noqa: E712
    fig = go.Figure()
    eligible = g[g["eligible"] == True]  # noqa: E712
    other = g[g["eligible"] != True]  # noqa: E712
    for sub, name, col, sym in (
        (eligible, "eligible", PAL["s1"], "circle"),
        (other, "excluded (loss limit)", PAL["muted"], "x"),
    ):
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=-sub["es_5"],
                y=sub["exp_return"],
                mode="markers",
                name=name,
                marker={"color": col, "size": 9, "symbol": sym, "opacity": 0.8},
                text=[
                    f"{lab}<br>{param_name}: {p:.3g}"
                    for lab, p in zip(sub["label"], sub["param"], strict=True)
                ],
                hovertemplate="%{text}<br>|ES| %{x:.1%} · E[ret] %{y:+.1%}<extra></extra>",
            )
        )
    b = g[g["id"] == best_id]
    if not b.empty:
        fig.add_trace(
            go.Scatter(
                x=-b["es_5"],
                y=b["exp_return"],
                mode="markers",
                name="recommended",
                marker={
                    "color": PAL["s2"],
                    "size": 16,
                    "symbol": "circle-open",
                    "line": {"width": 3},
                },
                hoverinfo="skip",
            )
        )
    _base(fig, title, 340)
    fig.update_xaxes(title_text="Risk: |expected shortfall 5%|", tickformat=".0%")
    fig.update_yaxes(title_text="Expected return", tickformat="+.0%")
    fig.update_layout(hovermode="closest")
    return fig


def stability_figure(group: pd.DataFrame, best_id: str, title: str, param_name: str) -> go.Figure:
    g = group[group["feasible"] == True].sort_values("param")  # noqa: E712
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g["param"],
            y=g["utility"],
            mode="lines+markers",
            name="utility",
            line={"color": PAL["s1"], "width": 2},
            hovertemplate=f"{param_name} %{{x:.3g}}<br>utility %{{y:+.1%}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g["param"],
            y=g["exp_return"],
            mode="lines+markers",
            name="E[return]",
            line={"color": PAL["s3"], "width": 1.5},
            hovertemplate="E[ret] %{y:+.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g["param"],
            y=g["es_5"],
            mode="lines+markers",
            name="ES 5%",
            line={"color": PAL["s2"], "width": 1.5},
            hovertemplate="ES %{y:+.1%}<extra></extra>",
        )
    )
    b = g[g["id"] == best_id]
    if not b.empty:
        fig.add_vline(
            x=float(b["param"].iloc[0]),
            line={"color": PAL["s2"], "width": 1, "dash": "dot"},
            annotation_text="recommended",
            annotation_position="top left",
        )
    _base(fig, title, 340)
    fig.update_xaxes(title_text=param_name)
    fig.update_yaxes(title_text="Return metrics", tickformat="+.0%")
    fig.update_layout(hovermode="x unified")
    return fig


def _tiles(m: dict, res: dict) -> html.Div:
    def f(k, fmt="{:+.1%}"):
        v = m.get(k)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "–"
        return fmt.format(v)

    items = [
        ("Invested (incl. fees)", f("invested", "{:,.2f} €")),
        ("Expected return", f("exp_return")),
        ("Median return", f("median_return")),
        ("P(profit)", f("p_profit", "{:.1%}")),
        ("P(knock-out)", f("p_knockout", "{:.1%}")),
        ("P(total loss)", f("p_total_loss", "{:.1%}")),
        ("VaR 5% / ES 5%", f"{f('var_5')} / {f('es_5')}"),
        ("95th percentile", f("gain_95")),
        ("Omega", f("omega", "{:.2f}")),
        ("Utility", f("utility", "{:+.2%}")),
        ("Stability", f("stability", "{:.0%}")),
        ("Cost drag", f("cost_drag", "{:.2%}")),
        ("Effective leverage", f("effective_leverage", "{:.2f}")),
        ("Break-even up / down", f"{f('break_even_up')} / {f('break_even_down')}"),
        ("Quantities", str(m.get("quantities", "–"))),
    ]
    return html.Div(
        [
            html.Div(
                [html.Div(k, className="tile-label"), html.Div(v, className="tile-value")],
                className="tile",
            )
            for k, v in items
        ],
        className="tiles",
    )


# --------------------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------------------


def register_callbacks(app, store: DerivativesStore) -> None:
    @app.callback(
        Output("d-strategies", "rowData"),
        Output("d-ko", "rowData"),
        Output("d-warrants", "rowData"),
        Output("d-status", "children"),
        Output("d-underlying", "options"),
        Input("d-underlying", "value"),
        Input("d-show-all", "value"),
        Input("d-reload", "n_clicks"),
    )
    def fill(underlyings, show_all, _n):
        from dash import ctx as dctx

        if dctx.triggered_id == "d-reload":
            store.reload()
        s, p = store.strategies, store.products
        if underlyings:
            s = s[s["underlying"].isin(underlyings)] if not s.empty else s
            p = p[p["underlying"].isin(underlyings)] if not p.empty else p
        if not s.empty and "all" not in (show_all or []):
            s = s[s["is_best"] == True]  # noqa: E712
        if not s.empty:
            s = s.sort_values(["underlying", "utility"], ascending=[True, False])
        ko = p[p["product_type"].isin(["ko_long", "ko_short"])] if not p.empty else p
        wr = p[p["product_type"].isin(["call_warrant", "put_warrant"])] if not p.empty else p
        m = store.meta
        status = (
            f"Derivatives as of {m.get('as_of')} · source: {m.get('source')} · horizon {m.get('horizon_days')} d · "
            f"budget {m.get('budget_eur')} € · drift: {m.get('drift_mode')} · {m.get('costs')} · "
            f"{len(store.strategies)} realizations, {len(store.products)} products"
            if m
            else "No derivatives data — run `screener derivatives refresh`."
        )
        opts = [
            {"label": u, "value": u}
            for u in (
                sorted(store.strategies["underlying"].unique())
                if not store.strategies.empty
                else []
            )
        ]
        return store.records(s), store.records(ko), store.records(wr), status, opts

    @app.callback(
        Output("d-detail", "children"),
        Input("d-strategies", "selectedRows"),
        State("d-show-all", "value"),
    )
    def detail(selected, _show_all):
        if not selected:
            return no_update
        row = selected[0]
        run = store.run
        ev = store.evaluator(row["underlying"])
        if run is None or ev is None:
            return html.Div(
                "Market context unavailable — refresh the screener and derivatives.",
                className="warn",
            )
        strategy = run.rebuild_strategy(row, store.products)
        if strategy is None:
            return html.Div(
                "Could not rebuild the strategy from the product table.", className="warn"
            )
        metrics = ev.metrics(strategy)
        for k in (
            "stability",
            "utility_gap",
            "rank",
        ):  # cross-sectional values live in the stored row
            metrics[k] = row.get(k)
        res = ev.pnl(strategy)
        if not res.get("feasible"):
            return html.Div("Strategy not feasible with the configured budget.", className="warn")
        cls = STRATEGY_CLASSES[row["strategy_class"]]
        group = store.strategies[
            (store.strategies["underlying"] == row["underlying"])
            & (store.strategies["strategy_class"] == row["strategy_class"])
        ]
        best_id = (
            row["id"]
            if row.get("is_best")
            else (
                group[group["is_best"].astype(bool)]["id"].iloc[0]
                if group["is_best"].astype(bool).any()
                else row["id"]
            )
        )  # noqa: E712
        legs_tbl = html.Table(
            [
                html.Tr(
                    [
                        html.Th(h)
                        for h in (
                            "Leg",
                            "Type",
                            "Issuer",
                            "Strike",
                            "Barrier",
                            "Maturity",
                            "Ask",
                            "Spread",
                            "Qty",
                        )
                    ]
                )
            ]
            + [
                html.Tr(
                    [
                        html.Td(leg.name or leg.isin),
                        html.Td(leg.product_type),
                        html.Td(leg.issuer),
                        html.Td(f"{leg.strike:,.2f}"),
                        html.Td(f"{leg.barrier:,.2f}" if leg.barrier else "–"),
                        html.Td(str(leg.maturity) if leg.maturity else "open-end"),
                        html.Td(f"{leg.ask:.3f} €"),
                        html.Td(f"{leg.spread_pct:.2%}"),
                        html.Td(str(q)),
                    ]
                )
                for leg, q in zip(strategy.legs, res["quantities"], strict=True)
            ],
            className="classes",
        )
        t = f"{row['underlying']} · {cls.label}"
        figs = [
            payoff_figure(ev, strategy, f"{t} — P&L at the horizon vs. underlying move"),
            distribution_figure(res, metrics, f"{t} — distribution of the return"),
            tradeoff_figure(
                group, best_id, f"{t} — risk/return of all sampled realizations", cls.param_name
            ),
            stability_figure(
                group, best_id, f"{t} — metrics along the class parameter", cls.param_name
            ),
        ]
        return html.Div(
            [
                html.H3(
                    f"{t}: {row['label']}" + (" · recommended" if row.get("is_best") else ""),
                    style={"margin": "0 0 6px"},
                ),
                html.Div(cls.description, className="note"),
                _tiles(metrics, res),
                legs_tbl,
                html.Div(
                    [dcc.Graph(figure=f, config={"displayModeBar": False}) for f in figs],
                    className="charts",
                ),
                html.Div(
                    "Terminal prices are sampled from the option-implied distribution of the underlying "
                    f"({store.meta.get('drift_mode', 'risk_neutral')} drift); knock-outs use the Brownian-bridge crossing "
                    "probability; warrants are re-priced with their own implied volatility held constant. Costs per "
                    "config/costs.yaml. Not financial advice.",
                    className="note",
                ),
            ]
        )
