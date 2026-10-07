import numpy as np
import pandas as pd
import pytest

from src.portfolio import ledoit_wolf_cov, naive_risk_parity_weights, risk_contributions, risk_parity_weights

TOL = 1e-8  # tolerancia declarada sobre la diferencia máxima entre contribuciones relativas


def _check_equal_rc(cov):
    w = risk_parity_weights(cov)
    rc = risk_contributions(w, cov)
    share = rc / rc.sum()
    assert np.all(w > 0) and np.isclose(w.sum(), 1.0)
    assert np.isclose(rc.sum(), np.sqrt(w @ cov @ w))
    assert np.max(np.abs(share - 1 / len(w))) < TOL


def test_risk_parity_equal_contributions_real_data(prices):
    """Prueba 4 (nivel C): RC_i iguales con la covarianza Ledoit-Wolf de los 6 activos."""
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).dropna()
    _check_equal_rc(ledoit_wolf_cov(returns.iloc[-126:]))


def test_risk_parity_long_short_book(prices):
    """Con lados mixtos se usa Σ_d = DΣD y las contribuciones siguen siendo iguales."""
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).dropna()
    D = np.diag([1, -1, 1, -1, 1, -1])
    _check_equal_rc(D @ ledoit_wolf_cov(returns.iloc[-126:]) @ D)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_risk_parity_random_covariances(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    _check_equal_rc(np.cov(X, rowvar=False))


def _coordinate_descent_reference(cov, budgets=None, tol=1e-14, max_iter=100_000):
    """Solución de referencia independiente del mismo problema: descenso cíclico por coordenadas.

    Para cada i resuelve Σ_ii y_i² + a_i y_i - b_i = 0 (a_i = Σ_{j≠i} Σ_ij y_j) y toma la raíz positiva.
    """
    n = len(cov)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    y = 1.0 / np.sqrt(np.diag(cov))
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            a = cov[i] @ y - cov[i, i] * y[i]
            y[i] = (-a + np.sqrt(a * a + 4 * cov[i, i] * b[i])) / (2 * cov[i, i])
        if np.max(np.abs(y - y_old)) < tol * np.max(np.abs(y)):
            break
    return y / y.sum()


def _covariances(prices):
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).dropna()
    D = np.diag([1, -1, 1, -1, 1, -1])
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 4)) @ rng.normal(size=(4, 4))
    return {
        "ultimos_126": ledoit_wolf_cov(returns.iloc[-126:]),
        "covid": ledoit_wolf_cov(returns.loc["2019-10-01":"2020-03-31"]),
        "largo_corto": D @ ledoit_wolf_cov(returns.iloc[-126:]) @ D,
        "aleatoria": np.cov(X, rowvar=False),
    }


def test_scipy_minimize_matches_independent_solver(prices):
    """scipy.optimize.minimize y el descenso por coordenadas llegan al mismo mínimo global (problema convexo)."""
    for name, cov in _covariances(prices).items():
        w_scipy = risk_parity_weights(cov)
        w_ref = _coordinate_descent_reference(cov)
        assert np.max(np.abs(w_scipy - w_ref)) < 1e-8, name


def test_risk_parity_satisfies_first_order_condition(prices):
    """En el óptimo y_i (Σy)_i = b_i: con w normalizado, w_i (Σw)_i es igual para todo i."""
    for name, cov in _covariances(prices).items():
        w = risk_parity_weights(cov)
        marginal = w * (cov @ w)
        assert np.max(np.abs(marginal / marginal.mean() - 1)) < 1e-7, name


def test_minimize_differs_from_naive_when_correlations_differ(prices):
    """Con correlaciones distintas, naive (1/σ) no iguala el riesgo y Risk Parity por minimización sí."""
    cov = _covariances(prices)["covid"]
    w_naive, w_rp = naive_risk_parity_weights(cov), risk_parity_weights(cov)
    rc_naive = risk_contributions(w_naive, cov) / risk_contributions(w_naive, cov).sum()
    assert np.max(np.abs(w_naive - w_rp)) > 0.01
    assert np.ptp(rc_naive) > 0.02
    _check_equal_rc(cov)


def test_risk_parity_converges_on_many_real_books(prices):
    """Barrido de libros reales (subconjuntos de activos, lados mixtos, fechas distintas): siempre converge
    y deja contribuciones iguales. Incluye fechas donde L-BFGS-B termina con estado "ABNORMAL" en el óptimo."""
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()})
    rng = np.random.default_rng(0)
    for t in returns.index[200::40]:
        pos = returns.index.get_loc(t)
        cov = ledoit_wolf_cov(returns.iloc[max(pos - 125, 1): pos + 1])
        for m in (2, 3, 6):
            sel = rng.choice(6, m, replace=False)
            D = np.diag(rng.choice([-1, 1], m))
            C = D @ cov[np.ix_(sel, sel)] @ D
            w = risk_parity_weights(C)
            rc = risk_contributions(w, C)
            assert np.max(np.abs(rc / rc.sum() - 1 / m)) < 1e-6


def test_naive_risk_parity_is_inverse_volatility():
    """w_i ∝ 1/σ_i; con correlaciones nulas también iguala las contribuciones al riesgo."""
    sigma = np.array([0.1, 0.2, 0.4])
    cov = np.diag(sigma ** 2)
    w = naive_risk_parity_weights(cov)
    np.testing.assert_allclose(w, (1 / sigma) / (1 / sigma).sum())
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc / rc.sum(), 1 / 3, atol=TOL)
    np.testing.assert_allclose(w, risk_parity_weights(cov), atol=1e-10)


def test_risk_budgets_are_respected():
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.01]])
    b = np.array([0.5, 0.3, 0.2])
    w = risk_parity_weights(cov, b)
    rc = risk_contributions(w, cov)
    np.testing.assert_allclose(rc / rc.sum(), b, atol=TOL)
