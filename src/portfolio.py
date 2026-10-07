"""Risk Parity y agregación de señales del portafolio multi-activo.

Tres formas de repartir el capital entre los activos activos:

- Pesos iguales: w_i = 1/n.
- Risk Parity naive: w_i = (1/σ_i) / Σ_j (1/σ_j). Iguala el riesgo solo si
  las correlaciones son todas iguales.
- Risk Parity por minimización (la versión completa):

Risk Parity: cada activo aporta lo mismo al riesgo del portafolio,

    RC_i = w_i (Σw)_i / σ_p = σ_p / n,   σ_p = sqrt(wᵀΣw).

Se resuelve con `scipy.optimize.minimize` (L-BFGS-B, gradiente analítico)
sobre la formulación convexa de Spinu: minimizar ½ yᵀΣy - Σ b_i ln y_i (y > 0).
Al ser convexa tiene un único mínimo global. La condición de primer orden es
y_i (Σy)_i = b_i, es decir RC_i ∝ b_i; con b_i = 1/n las contribuciones son
iguales. Se normaliza w = y / Σy.

Estimador de covarianza: Ledoit-Wolf (encogimiento hacia una matriz escalada
de identidad) sobre los últimos 126 retornos diarios (6 meses, la misma
longitud que la ventana de entrenamiento). Con 126 observaciones y 6 activos
la covarianza muestral es ruidosa; el encogimiento reduce el error de
estimación sin imponer un modelo de factores.

Posiciones largas y cortas: el riesgo de un libro con lados d_i ∈ {-1, +1}
se calcula con la covarianza de los retornos con signo, Σ_d = D Σ D
(D = diag(d)). Risk Parity se aplica sobre Σ_d y el peso final con signo es
d_i · w_i.

Política de agregación (cada día, con información hasta el cierre):
1. Dirección por activo: la señal confirmada 3 de 6 mantenida
   (`signals.hold_direction`) con los parámetros del régimen vigente.
2. Fuerza: |f| = |N⁺ - N⁻| / 6 de la señal que abrió la dirección. Solo
   entran activos con |f| >= min_strength del régimen.
3. Conflictos: si dos activos activos tienen correlación > 0.6 y lados
   opuestos, se descarta el de menor |f|; si empatan, ambos.
4. Asignación base: Risk Parity sobre Σ_d, naive 1/σ_i, o 1/k en el
   benchmark de pesos iguales, con k = número de activos activos.
5. Peso final = base · |f| · multiplicador de riesgo del régimen. Lo que no se
   asigna queda en efectivo, así que la exposición bruta es <= 1.

Transiciones de régimen: una posición abierta conserva sus SL/TP y su holding
máximo. Si con el régimen nuevo su fuerza de entrada no alcanza el mínimo, la
dirección pasa a 0 y se cierra al open siguiente. Si sigue activa, se
re-dimensiona con el multiplicador nuevo (el cambio de régimen dispara un
rebalanceo).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

COV_WINDOW = 126
CORR_CONFLICT = 0.6
REGIME_RULES = {
    "trend":          {"min_strength": 1 / 6, "risk_mult": 1.00},   # cualquier señal confirmada
    "mean_reversion": {"min_strength": 2 / 6, "risk_mult": 0.75},   # al menos 2 votos netos
    "crisis":         {"min_strength": 3 / 6, "risk_mult": 0.50},   # al menos 3 votos netos
}
DEFAULT_REGIME = "trend"


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Covarianza de Ledoit-Wolf de los retornos (filas = días, columnas = activos)."""
    return LedoitWolf().fit(returns.to_numpy()).covariance_


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """RC_i = w_i (Σw)_i / σ_p. Suman σ_p."""
    w = np.asarray(w, float)
    sigma_p = np.sqrt(w @ cov @ w)
    return w * (cov @ w) / sigma_p


def risk_parity_weights(cov: np.ndarray, budgets=None) -> np.ndarray:
    """Pesos (positivos, suman 1) con RC_i ∝ b_i: minimiza ½ yᵀΣy - Σ b_i ln y_i y normaliza w = y / Σy.

    La covarianza se re-escala por su diagonal promedio para que el problema esté bien
    condicionado (los pesos normalizados no cambian con la escala de Σ).
    """
    cov = np.asarray(cov, float)
    n = len(cov)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    S = cov / np.mean(np.diag(cov))

    def objective(y):
        return 0.5 * y @ S @ y - b @ np.log(y)

    def gradient(y):
        return S @ y - b / y

    y0 = 1.0 / np.sqrt(np.diag(S))
    res = minimize(objective, y0 / y0.sum(), jac=gradient, method="L-BFGS-B", bounds=[(1e-12, None)] * n,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
    # Se valida con la condición de optimalidad y_i (Σy)_i = b_i y no con `res.success`: con tolerancias
    # tan estrictas L-BFGS-B a veces termina con "ABNORMAL" (la búsqueda de línea ya no mejora a
    # precisión de máquina) estando ya en el óptimo.
    y = res.x
    residual = np.max(np.abs(y * (S @ y) / b - 1))
    if residual > 1e-6:
        raise RuntimeError(f"Risk Parity no convergió (residuo {residual:.2e}): {res.message}")
    return y / y.sum()


def naive_risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Versión naive: w_i = (1/σ_i) / Σ_j (1/σ_j). Ignora las correlaciones."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    return inv_vol / inv_vol.sum()


def latched_strength(direction: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    """Fuerza de la señal que abrió cada dirección, mantenida mientras la dirección siga abierta."""
    start = (direction != 0) & (direction != direction.shift().fillna(0))
    return strength.where(start).ffill().where(direction != 0).fillna(0.0)


def target_weights(direction: pd.DataFrame, strength: pd.DataFrame, returns: pd.DataFrame,
                   labels: pd.Series, method: str = "rp", cov_window: int = COV_WINDOW) -> tuple:
    """Aplica la política de agregación día por día.

    direction: lados mantenidos por activo (fechas x activos).
    strength: fuerza f de la señal de cada día (se usa la del día de entrada).
    returns: retornos diarios de cierre de todos los activos (historia completa).
    labels: régimen vigente por fecha.
    method: "rp" (Risk Parity por minimización), "naive" (1/σ normalizado) o "ew" (1/n).
    labels None = modelo sin régimen (se usan las reglas de tendencia: sin filtro extra y multiplicador 1).

    Regresa (direction_adj, weights): lados tras la política y pesos absolutos.
    """
    f = latched_strength(direction, strength).abs()
    dir_adj = pd.DataFrame(0.0, index=direction.index, columns=direction.columns)
    weights = pd.DataFrame(0.0, index=direction.index, columns=direction.columns)
    ret_pos = returns.index.get_indexer(direction.index)

    for k, t in enumerate(direction.index):
        rule = REGIME_RULES.get(None if labels is None else labels.get(t), REGIME_RULES[DEFAULT_REGIME])
        d, s = direction.iloc[k].to_numpy(), f.iloc[k].to_numpy()
        active = (d != 0) & (s >= rule["min_strength"] - 1e-9)
        if not active.any():
            continue
        window = returns.iloc[max(ret_pos[k] - cov_window + 1, 1): ret_pos[k] + 1]
        cov = ledoit_wolf_cov(window)
        corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        idx = np.flatnonzero(active)
        drop = set()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if d[i] == -d[j] and corr[i, j] > CORR_CONFLICT:
                    if s[i] > s[j]:
                        drop.add(j)
                    elif s[j] > s[i]:
                        drop.add(i)
                    else:
                        drop.update((i, j))
        idx = np.array([i for i in idx if i not in drop], dtype=int)
        if len(idx) == 0:
            continue
        if method == "rp":
            D = np.diag(d[idx])
            base = risk_parity_weights(D @ cov[np.ix_(idx, idx)] @ D)
        elif method == "naive":
            base = naive_risk_parity_weights(cov[np.ix_(idx, idx)])
        else:
            base = np.full(len(idx), 1.0 / len(idx))
        dir_adj.iloc[k, idx] = d[idx]
        weights.iloc[k, idx] = base * s[idx] * rule["risk_mult"]
    return dir_adj, weights


def rebalance_schedule(index: pd.DatetimeIndex, every: int, labels: pd.Series = None) -> pd.Series:
    """Rebalanceo cada `every` días hábiles y además cuando cambia el régimen."""
    flag = pd.Series(np.arange(len(index)) % every == every - 1, index=index)
    if labels is not None:
        lab = labels.reindex(index)
        flag |= (lab != lab.shift()) & lab.shift().notna()
    return flag


def realized_risk_shares(positions: pd.DataFrame, equity: pd.Series, returns: pd.DataFrame,
                         cov_window: int = COV_WINDOW) -> pd.DataFrame:
    """Proporción de riesgo (RC_i / σ_p) de cada activo en el libro realizado, por mes.

    Usa los pesos con signo al cierre (valor de mercado / equity) del último día
    de cada mes y la covarianza Ledoit-Wolf de los 126 días previos.
    """
    w_all = positions.div(equity, axis=0)
    month_ends = w_all.groupby(w_all.index.to_period("M")).tail(1).index
    rows = {}
    for t in month_ends:
        w = w_all.loc[t].to_numpy()
        if np.abs(w).sum() == 0:
            continue
        pos = returns.index.get_loc(t)
        cov = ledoit_wolf_cov(returns.iloc[max(pos - cov_window + 1, 1): pos + 1])
        rc = risk_contributions(w, cov)
        rows[t] = rc / rc.sum()
    return pd.DataFrame(rows, index=positions.columns).T


def held_direction(oos: dict) -> pd.DataFrame:
    """Dirección mantenida por activo a partir de las señales fuera de muestra."""
    from .signals import hold_direction

    return pd.DataFrame({t: hold_direction(oos["signal"][t].fillna(0), oos["max_holding"][t].fillna(1))
                         for t in oos["signal"].columns}, index=oos["signal"].index)


def allocate(prices: dict, oos: dict, labels: pd.Series, method: str = "rp") -> tuple:
    """Direcciones y pesos objetivo del portafolio (ver `target_weights`) con las señales fuera de muestra."""
    tickers = list(oos["signal"].columns)
    returns = pd.DataFrame({t: prices[t]["close"].pct_change() for t in tickers})
    return target_weights(held_direction(oos), oos["strength"].fillna(0), returns,
                          None if labels is None else labels.reindex(oos["signal"].index), method)


def run_portfolio(prices: dict, oos: dict, labels: pd.Series, alloc: tuple, every: int = 5,
                  commission: float = None) -> dict:
    """Backtest del portafolio con las señales fuera de muestra del walk-forward.

    oos: salida de `optimize.oos_signals`. alloc: salida de `allocate`. every: días entre rebalanceos.
    """
    from .backtest import COMMISSION, run_backtest

    dir_adj, weights = alloc
    index = dir_adj.index
    res = run_backtest(prices, dir_adj, weights, oos["atr"], oos["sl_mult"], oos["tp_mult"],
                       rebalance=rebalance_schedule(index, every, None if labels is None else labels.reindex(index)),
                       commission=COMMISSION if commission is None else commission)
    res["weights"] = weights
    res["direction"] = dir_adj
    return res


def run_single_assets(prices: dict, oos: dict, commission: float = None) -> dict:
    """Backtest independiente de cada activo ($1,000,000 cada uno) con sus señales fuera de muestra."""
    from .backtest import COMMISSION, run_backtest

    direction = held_direction(oos)
    return {t: run_backtest({t: prices[t]}, direction[[t]], oos["position_frac"][[t]], oos["atr"][[t]],
                            oos["sl_mult"][[t]], oos["tp_mult"][[t]],
                            commission=COMMISSION if commission is None else commission)
            for t in direction.columns}


def buy_and_hold(prices: dict, index: pd.DatetimeIndex, commission: float = None) -> dict:
    """Benchmark pasivo: largo en todos los activos con pesos iguales y rebalanceo mensual."""
    from .backtest import COMMISSION, run_backtest

    tickers = list(prices)
    direction = pd.DataFrame(1.0, index=index, columns=tickers)
    atr = pd.DataFrame(1.0, index=index, columns=tickers)
    month = pd.Series(index.to_period("M"), index=index)
    month_end = month != month.shift(-1)
    # Stops inalcanzables: el benchmark nunca sale por SL/TP.
    return run_backtest(prices, direction, 1.0 / len(tickers), atr, 1e12, 1e12, rebalance=month_end,
                        commission=COMMISSION if commission is None else commission)
