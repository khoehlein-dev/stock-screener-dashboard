"""Application settings.

Values come from environment variables prefixed with ``SCREENER_`` or from a
``.env`` file in the working directory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "XOM",
    "UNH",
    "LLY",
    "V",
    "MA",
    "COST",
    "HD",
    "PG",
    "NFLX",
    "AMD",
    "CRM",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCREENER_", env_file=".env", extra="ignore")

    massive_api_key: str | None = Field(default=None, description="Massive API key")
    massive_base_url: str = "https://api.massive.com"
    universe: list[str] = Field(default_factory=lambda: list(DEFAULT_UNIVERSE))
    benchmark: str = "SPY"

    has_fundamentals: bool = True
    has_indices: bool = False
    has_futures: bool = False

    risk_free_rate: float = 0.04
    history_days: int = 400
    horizons_days: list[int] = Field(default_factory=lambda: [30, 60, 90])
    primary_horizon_days: int = 30
    loss_thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.20])
    gain_thresholds: list[float] = Field(default_factory=lambda: [0.05, 0.10])
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 400
    american_iv: bool = False

    max_workers: int = 8
    requests_per_second: float = 20.0
    data_dir: Path = Path("./data")
    provider: str = Field(default="auto", description="auto | massive | synthetic")
    host: str = "127.0.0.1"
    port: int = 8050

    @field_validator("universe", mode="before")
    @classmethod
    def _parse_universe(cls, value):
        if isinstance(value, str):
            path = Path(value)
            if path.exists():
                lines = path.read_text().splitlines()
                return [
                    ln.strip().split(",")[0].upper()
                    for ln in lines
                    if ln.strip() and not ln.startswith("#")
                ]
            return [t.strip().upper() for t in value.split(",") if t.strip()]
        return value

    @field_validator("horizons_days", "loss_thresholds", "gain_thresholds", mode="before")
    @classmethod
    def _parse_list(cls, value):
        if isinstance(value, str):
            return [float(v) if "." in v else int(v) for v in value.split(",") if v.strip()]
        return value

    def resolved_provider(self) -> str:
        if self.provider != "auto":
            return self.provider
        return "massive" if self.massive_api_key else "synthetic"


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
