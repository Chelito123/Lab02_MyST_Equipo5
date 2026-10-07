import pandas as pd
import pytest

from src.regimes import HMMModel, causal_labels, market_index, regime_features, smooth_labels

# El HMM re-ajusta Baum-Welch cada mes; para que la prueba sea rápida se usa una muestra de 900 días.
SAMPLE_ROWS = {"rules": None, "kmeans": None, "hmm": 900}
CUTS = {"rules": [400, 1200, 1900], "kmeans": [400, 1200, 1900], "hmm": [450, 700]}


def _labels(prices, rows, method):
    sample = {k: df.iloc[:rows] for k, df in prices.items()}
    return smooth_labels(causal_labels(regime_features(market_index(sample)), method))


@pytest.fixture(scope="module")
def full_labels(prices):
    return {m: _labels(prices, n, m) for m, n in SAMPLE_ROWS.items()}


@pytest.mark.parametrize("method, t", [(m, t) for m, ts in CUTS.items() for t in ts])
def test_regime_label_does_not_change_with_future_data(prices, full_labels, method, t):
    """Prueba 4 (nivel B): la etiqueta en t no cambia al agregar datos posteriores a t."""
    truncated = _labels(prices, t + 1, method)
    date = truncated.index[-1]
    assert isinstance(full_labels[method][date], str)
    pd.testing.assert_series_equal(truncated, full_labels[method].loc[:date])


def test_viterbi_uses_future_but_filter_does_not(prices):
    """La etiqueta filtrada del HMM es causal; Viterbi re-etiqueta el pasado al ver más datos."""
    F = regime_features(market_index(prices))
    model = HMMModel().fit(F.iloc[:700])
    filtered_full, viterbi_full = model.predict(F), model.predict_viterbi(F)
    changed = 0
    for t in range(200, len(F), 150):
        assert model.predict(F.iloc[: t + 1]).iloc[-1] == filtered_full.iloc[t]
        changed += int((model.predict_viterbi(F.iloc[: t + 1]) != viterbi_full.iloc[: t + 1]).sum())
    assert changed > 0
