"""Command line interface: ``screener refresh | serve | demo | columns``."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import typer

from .config import Settings

app = typer.Typer(help="Options-aware stock screener built on Massive, QuantLib and Dash.")


def _settings(provider: str | None, data_dir: Path | None) -> Settings:
    overrides = {}
    if provider:
        overrides["provider"] = provider
    if data_dir:
        overrides["data_dir"] = data_dir
    return Settings(**overrides)


def make_provider(settings: Settings, as_of: dt.date):
    name = settings.resolved_provider()
    if name == "massive":
        from .data.massive_provider import MassiveProvider

        return MassiveProvider(
            settings.massive_api_key,
            settings.massive_base_url,
            settings.requests_per_second,
            settings.has_fundamentals,
            settings.has_indices,
        )
    from .data.synthetic_provider import SyntheticProvider

    return SyntheticProvider(as_of=as_of, risk_free_rate=settings.risk_free_rate)


@app.command()
def refresh(
    as_of: str | None = typer.Option(
        None, help="Trading date YYYY-MM-DD (default: last NYSE session)"
    ),
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers (default: configured universe)"
    ),
    provider: str | None = typer.Option(None, help="massive | synthetic (default: auto)"),
    data_dir: Path | None = typer.Option(None, help="Data directory"),
    workers: int | None = typer.Option(None, help="Parallel workers"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Fetch data, compute all metrics and write the screener table."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from .analytics.pipeline import build_screener, default_as_of
    from .data.cache import DataCache

    settings = _settings(provider, data_dir)
    if workers:
        settings.max_workers = workers
    universe = [t.strip().upper() for t in tickers.split(",")] if tickers else settings.universe
    date = dt.date.fromisoformat(as_of) if as_of else default_as_of(settings.resolved_provider())
    prov = make_provider(settings, date)
    cache = DataCache(settings.data_dir)
    typer.echo(f"Refreshing {len(universe)} tickers for {date} with provider '{prov.name}'…")

    def progress(i, n, t):
        typer.echo(f"  [{i}/{n}] {t}")

    df = build_screener(prov, settings, date, cache, universe, progress)
    path = cache.put_screener(date, df)
    n_opt = int(df["iv_30"].notna().sum()) if "iv_30" in df else 0
    typer.echo(f"Wrote {path} ({len(df)} rows, {n_opt} with option analytics)")
    warn = getattr(prov, "warnings", [])
    if warn:
        typer.echo(f"{len(warn)} provider warnings (first 5):")
        for w in warn[:5]:
            typer.echo(f"  - {w}")


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    data_dir: Path | None = typer.Option(None),
    debug: bool = typer.Option(False),
):
    """Serve the dashboard from the last refreshed table."""
    from .app.dashboard import create_app

    settings = _settings(None, data_dir)
    dash_app = create_app(settings)
    dash_app.run(host=host or settings.host, port=port or settings.port, debug=debug)


@app.command()
def demo(
    port: int = typer.Option(8050),
    data_dir: Path = typer.Option(Path("./data-demo")),
    n: int = typer.Option(25, help="Number of synthetic tickers"),
):
    """Refresh with the synthetic provider and serve — no API key required."""
    from .analytics.pipeline import build_screener, default_as_of
    from .app.dashboard import create_app
    from .data.cache import DataCache
    from .data.synthetic_provider import SyntheticProvider

    logging.basicConfig(level=logging.INFO)
    settings = Settings(provider="synthetic", data_dir=data_dir)
    universe = (settings.universe + [f"SYN{i:02d}" for i in range(100)])[:n]
    date = default_as_of("synthetic")
    prov = SyntheticProvider(as_of=date, risk_free_rate=settings.risk_free_rate)
    cache = DataCache(data_dir)
    df = build_screener(prov, settings, date, cache, universe)
    cache.put_screener(date, df)
    typer.echo(
        f"Synthetic screener ready: {len(df)} tickers as of {date}. Serving on http://127.0.0.1:{port}"
    )
    create_app(settings, data_dir).run(host="127.0.0.1", port=port, debug=False)


@app.command()
def columns():
    """List all available screener columns."""
    from .app.columns import COLUMNS

    for c in COLUMNS:
        typer.echo(f"{c.key:24s} {c.group:24s} {c.label:14s} {c.tooltip}")


if __name__ == "__main__":
    app()
