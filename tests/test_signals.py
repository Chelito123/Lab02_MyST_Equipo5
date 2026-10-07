import numpy as np
import pandas as pd
import pytest

from conftest import make_ohlc
from src.backtest import run_backtest
from src.signals import (DEFAULT_PARAMS, VOTE_COLUMNS, confirm, generate_signals, hold_direction,
                         max_vote_correlation, rsi_zone_votes, signal_strength)


@pytest.mark.parametrize("t", [60, 700, 1500, 2100])
def test_causality_truncated_series_gives_same_signal(prices, t):
    """Prueba 1: la señal en t con df.iloc[:t+1] es la misma que con la serie completa."""
    df = prices["AAPL"]
    full = generate_signals(df, DEFAULT_PARAMS)
    truncated = generate_signals(df.iloc[: t + 1], DEFAULT_PARAMS)
    pd.testing.assert_series_equal(full.iloc[t], truncated.iloc[t], check_names=False)


@pytest.mark.parametrize("votes, expected", [
    ((1, 0, 0, 0, 0, 0), 0), ((1, 1, 0, 0, 0, 0), 0), ((1, 1, -1, -1, 0, 0), 0),
    ((1, 1, 1, 0, 0, 0), 1), ((1, 1, 1, 1, -1, -1), 1), ((1, 1, 1, 1, 1, 1), 1),
    ((-1, -1, -1, 0, 0, 1), -1), ((-1, -1, -1, -1, 1, 1), -1),
    ((1, 1, 1, -1, -1, -1), 0),  # empate 3 contra 3: no abre
])
def test_confirmation_rule_table(votes, expected):
    """Prueba 2a: con menos de tres indicadores a favor no hay señal; con tres o más (y mayoría), sí."""
    v = pd.DataFrame([votes], columns=VOTE_COLUMNS)
    assert confirm(v).iloc[0] == expected


def test_confirmation_rule_opens_positions_only_with_three_votes():
    """Prueba 2b: en el backtest, dos votos a favor no abren posición y tres sí."""
    df = make_ohlc(np.full(10, 100.0))
    atr = pd.DataFrame({"X": 1.0}, index=df.index)
    for n_votes, should_trade in ((1, False), (2, False), (3, True)):
        votes = pd.DataFrame({c: int(i < n_votes) for i, c in enumerate(VOTE_COLUMNS)}, index=df.index)
        direction = pd.Series(hold_direction(confirm(votes), 100), index=df.index).to_frame("X")
        res = run_backtest({"X": df}, direction, 1.0, atr, 5.0, 5.0)
        assert (len(res["fills"]) > 0) == should_trade


def test_rsi_zone_needs_k_days_and_tolerates_small_pullbacks():
    """RSI: nivel 70 (y 30), tolerancia 5, k = 3. Un regreso a 68 no saca de la zona; bajar de 65 sí."""
    rsi_path = [50, 72, 68, 71, 66, 60, 50, 25, 28, 32, 26, 40]
    #           entra ↑ 68 y 71 siguen dentro → día 3 vota −1;  60 < 65 sale
    #                                   entra ↓ 28, 32 (< 35) siguen → día 3 vota +1;  40 > 35 sale
    expected = [0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 1, 0]
    np.testing.assert_array_equal(rsi_zone_votes(rsi_path, level=70, tol=5, k=3), expected)


def test_rsi_zone_without_tolerance_is_reset_by_a_small_pullback():
    """Sin tolerancia, el regreso a 68 reinicia el conteo y el RSI nunca llega a k = 3 días."""
    rsi_path = [50, 72, 68, 71, 66, 60]
    np.testing.assert_array_equal(rsi_zone_votes(rsi_path, level=70, tol=0, k=3), [0, 0, 0, 0, 0, 0])


def test_strength_is_net_votes_over_six():
    v = pd.DataFrame([(1, 1, 1, 0, 0, 0), (1, 1, 1, -1, 0, 0), (1, 1, 1, 1, 1, 1)], columns=VOTE_COLUMNS)
    np.testing.assert_allclose(signal_strength(v), [3 / 6, 2 / 6, 1.0])


def test_max_vote_correlation_ignores_constant_votes():
    v = pd.DataFrame({"a": [1, 0, -1, 1, 0], "b": [1, 0, -1, 1, 0], "c": [0, 0, 0, 0, 0]})
    assert max_vote_correlation(v) == pytest.approx(1.0)
    assert max_vote_correlation(v[["a", "c"]]) == 0.0


def test_hold_direction_keeps_side_until_opposite_or_max_holding():
    s = [1, 0, 0, 0, -1, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(hold_direction(s, 3), [1, 1, 1, 0, -1, -1, -1, 0, 0, 0])
