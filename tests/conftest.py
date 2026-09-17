import datetime as dt

import pytest

from stock_screener.config import Settings
from stock_screener.data.synthetic_provider import SyntheticProvider

AS_OF = dt.date(2026, 9, 17)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        _env_file=None, universe=["AAPL", "MSFT", "XOM"], max_workers=2, provider="synthetic"
    )


@pytest.fixture(scope="session")
def provider() -> SyntheticProvider:
    return SyntheticProvider(as_of=AS_OF, risk_free_rate=0.04)
