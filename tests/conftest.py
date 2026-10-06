import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_prices  # noqa: E402


@pytest.fixture(scope="session")
def prices():
    return load_prices()


def make_ohlc(close, spread=0.5, volume=1_000.0):
    """OHLC sintético: open = close previo, high/low = max/min(open, close) ± spread."""
    close = pd.Series(close, index=pd.bdate_range("2020-01-01", periods=len(close)), dtype=float)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": volume,
    })
