import numpy as np
import pandas as pd

from conftest import make_ohlc
from src.backtest import COMMISSION, run_backtest, single_asset_backtest
from src.signals import DEFAULT_PARAMS, generate_signals


def test_accounting_real_data(prices):
    """Prueba 3: valor final = efectivo + posiciones y costos = comisión × nocional operado."""
    df = prices["JPM"]
    res = single_asset_backtest(df, generate_signals(df, DEFAULT_PARAMS), DEFAULT_PARAMS, "JPM")
    final_close = df["close"].iloc[-1]
    assert np.isclose(res["equity"].iloc[-1], res["cash"] + res["shares"]["JPM"] * final_close, rtol=0, atol=1e-6)
    assert np.isclose(res["fills"]["commission"].sum(), COMMISSION * res["fills"]["notional"].sum(), atol=1e-6)
    # cada operación cerrada tiene exactamente una ejecución de entrada y una de salida
    assert (res["fills"]["reason"] == "entry").sum() - len(res["trades"]) in (0, 1)
    assert len(res["fills"]) == len(res["trades"]) * 2 + int(res["shares"]["JPM"] != 0)


def test_costs_equal_number_of_trades_times_commission():
    """Prueba 3b: con precio constante, cada ejecución paga comisión × nocional y el equity solo pierde costos."""
    df = make_ohlc(np.full(12, 50.0), spread=0.0)
    direction = pd.DataFrame({"X": [1, 1, 0, 0, -1, -1, 0, 1, 0, 0, 0, 0]}, index=df.index, dtype=float)
    res = run_backtest({"X": df}, direction, 0.5, pd.DataFrame({"X": 1.0}, index=df.index), 3.0, 3.0)
    fills = res["fills"]
    assert len(fills) == 6  # 3 operaciones x (entrada + salida)
    np.testing.assert_allclose(fills["commission"], COMMISSION * fills["notional"])
    assert np.isclose(1_000_000 - res["equity"].iloc[-1], fills["commission"].sum())


def test_stop_loss_wins_when_both_levels_in_same_bar():
    df = make_ohlc([100, 100, 100, 100], spread=0.0)
    df.loc[df.index[2], ["high", "low"]] = [110.0, 90.0]  # toca SL y TP en la misma barra
    direction = pd.DataFrame({"X": [1, 1, 1, 1]}, index=df.index, dtype=float)
    res = run_backtest({"X": df}, direction, 1.0, pd.DataFrame({"X": 5.0}, index=df.index), 1.0, 1.0)
    first = res["trades"].iloc[0]
    assert first["exit_reason"] == "stop_loss" and first["exit_price"] == 95.0


def test_no_leverage_at_execution():
    """Pesos que suman 3 se recortan: el nocional bruto ejecutado no excede el equity."""
    tickers = ["A", "B", "C"]
    prices = {t: make_ohlc(np.full(5, 20.0 + k), spread=0.0) for k, t in enumerate(tickers)}
    index = prices["A"].index
    direction = pd.DataFrame(1.0, index=index, columns=tickers)
    res = run_backtest(prices, direction, 1.0, pd.DataFrame(1.0, index=index, columns=tickers), 50.0, 50.0)
    gross = res["positions"].abs().sum(axis=1)
    assert (gross <= res["equity"] + 1e-6).all()
    assert gross.iloc[-1] > 0.99 * res["equity"].iloc[-1]
