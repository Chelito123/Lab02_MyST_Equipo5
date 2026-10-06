"""Carga, validación y auditoría de datos.

Los precios diarios se descargan UNA vez de Yahoo Finance (precios ajustados
por splits y dividendos) y se congelan en `data/<TICKER>.csv`. A partir de ahí
el proyecto solo lee esos archivos, de modo que los resultados son
reproducibles aunque Yahoo cambie su historia.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# 2n activos con n = 3 integrantes; cada integrante es responsable de 2.
TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"]
START = "2018-01-01"
END = "2026-10-01"          # exclusivo en yfinance -> último día 2026-09-30
TRAIN_END = "2023-12-31"    # train: 2018-2023, test: 2024-01 -> 2026-09

COLUMNS = ["open", "high", "low", "close", "volume"]


def download_prices(tickers=TICKERS, start=START, end=END, data_dir=DATA_DIR) -> None:
    """Descarga OHLCV diario ajustado y lo guarda como un CSV por activo."""
    import yfinance as yf

    data_dir.mkdir(parents=True, exist_ok=True)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="ticker")
    for t in tickers:
        df = raw[t].rename(columns=str.lower)[COLUMNS]
        df.index.name = "date"
        df.to_csv(data_dir / f"{t}.csv")


def load_prices(tickers=TICKERS, data_dir=DATA_DIR) -> dict:
    """Lee los CSV congelados y los alinea a las fechas comunes a todos los activos."""
    prices = {t: pd.read_csv(data_dir / f"{t}.csv", index_col="date", parse_dates=True)[COLUMNS]
              for t in tickers}
    common = sorted(set.intersection(*(set(df.index) for df in prices.values())))
    return {t: df.loc[common].astype(float) for t, df in prices.items()}


def audit_prices(prices: dict) -> pd.DataFrame:
    """Una fila por activo con las verificaciones de calidad de datos.

    - n_nan / n_duplicated: faltantes y fechas repetidas.
    - n_nonpositive: precios <= 0 o volumen < 0.
    - n_bad_ohlc: high < max(open, close) o low > min(open, close).
    - max_gap_days: mayor separación en días naturales entre barras consecutivas.
    - max_abs_return: mayor |retorno diario| de cierre (para detectar saltos espurios).
    """
    rows = []
    for t, df in prices.items():
        o, h, l, c, v = (df[k] for k in COLUMNS)
        rows.append({
            "ticker": t,
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "n_rows": len(df),
            "n_nan": int(df.isna().sum().sum()),
            "n_duplicated": int(df.index.duplicated().sum()),
            "n_nonpositive": int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (v < 0)).sum()),
            "n_bad_ohlc": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c))).sum()),
            "max_gap_days": int(df.index.to_series().diff().dt.days.max()),
            "max_abs_return": float(c.pct_change().abs().max()),
        })
    return pd.DataFrame(rows).set_index("ticker")


def validate_prices(prices: dict) -> None:
    """Lanza ValueError si algún activo no pasa la auditoría o si los activos no están traslapados."""
    audit = audit_prices(prices)
    bad = audit[(audit[["n_nan", "n_duplicated", "n_nonpositive", "n_bad_ohlc"]] > 0).any(axis=1)]
    if len(bad):
        raise ValueError(f"Datos inválidos:\n{bad}")
    if audit["n_rows"].nunique() != 1:
        raise ValueError("Los activos no tienen el mismo número de barras tras alinear fechas.")
    years = (audit["end"].iloc[0] - audit["start"].iloc[0]).days / 365.25
    if years < 6:
        raise ValueError(f"Se requieren al menos 6 años de datos, hay {years:.1f}.")


def is_train(index: pd.DatetimeIndex, train_end: str = TRAIN_END) -> np.ndarray:
    """Máscara booleana: True en el periodo de entrenamiento."""
    return np.asarray(index <= pd.Timestamp(train_end))
