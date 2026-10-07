"""Detección de régimen de mercado: reglas, K-means y HMM gaussiano (3 estados).

Serie base: índice de la canasta equiponderada de los activos del universo.
Variables, calculadas en una ventana móvil de 63 días hábiles (3 meses):

- vol: volatilidad anualizada de los retornos diarios.
- efficiency: razón de eficiencia de Kaufman |P_t - P_{t-63}| / Σ|ΔP| ∈ [0, 1]
  (1 = movimiento direccional limpio, 0 = ruido sin dirección).
- autocorr: autocorrelación de rezago 1 de los retornos (negativa = reversión).

(Se probó además el drawdown de la ventana; en train bajaba el silhouette de
K-means de 0.375 a 0.319 por ser redundante con la volatilidad, así que se descartó.)

Clasificadores (se ajustan con la historia disponible y etiquetan de forma causal):

1. Reglas: crisis si vol > q_0.90(vol); tendencia si no es crisis y
   efficiency > mediana(efficiency); reversión a la media en otro caso. Los
   umbrales se estiman con la historia del ajuste.
2. K-means sobre las variables estandarizadas.
3. HMM gaussiano (covarianza completa, Baum-Welch). Para operar se usa la
   etiqueta FILTRADA ŝ_t = argmax_j P(S_t = j | x_1..x_t) del algoritmo
   forward. Viterbi resuelve argmax P(S_1..S_T | x_1..x_T): la etiqueta en t
   usa observaciones posteriores a t (look-ahead), así que solo se usa como
   comparación visual, nunca para operar.

Nombres de los estados (K-means y HMM) por centroide/media: crisis = mayor
volatilidad; de los otros dos, tendencia = mayor eficiencia y reversión a la
media = el restante.

Causalidad: las variables solo usan datos <= t, y el modelo de cada mes se
ajusta con las variables observadas hasta el último día del mes anterior
(ventana expansiva; el HMM parte de los parámetros del mes anterior). La
etiqueta de t no cambia al agregar datos posteriores. La clasificación se
actualiza diariamente al cierre, con histéresis: el régimen vigente solo
cambia cuando la etiqueta nueva se repite `CONFIRM_DAYS` días seguidos.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REGIME_WINDOW = 63
REGIMES = ["trend", "mean_reversion", "crisis"]
METHODS = ["rules", "kmeans", "hmm"]
METHOD_NAMES = {"rules": "Reglas", "kmeans": "K-means", "hmm": "HMM (filtrada)"}
FEATURES = ["vol", "efficiency", "autocorr"]
CONFIRM_DAYS = 3
MIN_FIT_OBS = 126
SEED = 42


def market_index(prices: dict) -> pd.Series:
    """Índice (base 100) de la canasta equiponderada rebalanceada diariamente."""
    rets = pd.DataFrame({t: df["close"].pct_change() for t, df in prices.items()}).mean(axis=1).fillna(0.0)
    return (100 * (1 + rets).cumprod()).rename("market")


def regime_features(index: pd.Series, window: int = REGIME_WINDOW) -> pd.DataFrame:
    """Variables de régimen en ventana móvil de `window` días."""
    r = index.pct_change()
    path = index.diff().abs().rolling(window).sum()
    return pd.DataFrame({
        "vol": r.rolling(window).std() * np.sqrt(252),
        "efficiency": (index - index.shift(window)).abs() / path,
        "autocorr": r.rolling(window).corr(r.shift(1)),
    }).dropna()


def name_states(centers: pd.DataFrame) -> dict:
    """Nombre económico de cada estado a partir de su centro en unidades estandarizadas."""
    crisis = centers["vol"].idxmax()
    rest = centers.drop(index=crisis)
    trend = rest["efficiency"].idxmax()
    mean_rev = rest.drop(index=trend).index[0]
    return {trend: "trend", mean_rev: "mean_reversion", crisis: "crisis"}


class _Standardized:
    """Estandarización con la media y desviación de los datos del ajuste."""

    def _set_scale(self, F: pd.DataFrame):
        self.mean_, self.std_ = F.mean(), F.std()

    def standardize(self, F: pd.DataFrame) -> np.ndarray:
        return ((F[FEATURES] - self.mean_) / self.std_).to_numpy()


class RulesModel(_Standardized):
    """Crisis si vol > q_0.90; tendencia si efficiency > mediana; reversión a la media en otro caso."""

    def __init__(self, vol_quantile: float = 0.90, eff_quantile: float = 0.50, **_):
        self.vol_quantile = vol_quantile
        self.eff_quantile = eff_quantile

    def fit(self, F: pd.DataFrame, **_) -> "RulesModel":
        self._set_scale(F[FEATURES])
        self.vol_threshold_ = F["vol"].quantile(self.vol_quantile)
        self.eff_threshold_ = F["efficiency"].quantile(self.eff_quantile)
        return self

    def predict(self, F: pd.DataFrame) -> pd.Series:
        lab = np.where(F["vol"] > self.vol_threshold_, "crisis",
                       np.where(F["efficiency"] > self.eff_threshold_, "trend", "mean_reversion"))
        return pd.Series(lab, index=F.index, name="regime")


class KMeansModel(_Standardized):
    """K-means de 3 clusters sobre las variables estandarizadas."""

    def __init__(self, n_states: int = 3, seed: int = SEED):
        self.n_states = n_states
        self.seed = seed

    def fit(self, F: pd.DataFrame, **_) -> "KMeansModel":
        self._set_scale(F[FEATURES])
        self.km_ = KMeans(self.n_states, n_init=20, random_state=self.seed).fit(self.standardize(F))
        self.names_ = name_states(pd.DataFrame(self.km_.cluster_centers_, columns=FEATURES))
        return self

    def predict(self, F: pd.DataFrame) -> pd.Series:
        clusters = self.km_.predict(self.standardize(F))
        return pd.Series([self.names_[k] for k in clusters], index=F.index, name="regime")


class GaussianHMM:
    """HMM con emisiones gaussianas de covarianza completa.

    P(S_t = j | S_{t-1} = i) = A_ij,   x_t | S_t = j ~ N(μ_j, Σ_j).
    Ajuste por Baum-Welch (EM) en espacio logarítmico, inicializado con K-means
    o con los parámetros de un ajuste previo (`init`).
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4,
                 reg_covar: float = 1e-3, seed: int = SEED):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def _log_emissions(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([
            multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covars_[j], allow_singular=True)
            for j in range(self.n_states)])

    def _forward(self, logB: np.ndarray) -> np.ndarray:
        """log α_t(j) = log P(x_1..x_t, S_t = j)."""
        T, K = logB.shape
        log_alpha = np.empty((T, K))
        log_alpha[0] = np.log(self.startprob_) + logB[0]
        logA = np.log(self.transmat_)
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + logA, axis=0) + logB[t]
        return log_alpha

    def _backward(self, logB: np.ndarray) -> np.ndarray:
        T, K = logB.shape
        log_beta = np.zeros((T, K))
        logA = np.log(self.transmat_)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(logA + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def fit(self, X: np.ndarray, init: "GaussianHMM" = None) -> "GaussianHMM":
        K, d = self.n_states, X.shape[1]
        if init is None:
            km = KMeans(n_clusters=K, n_init=20, random_state=self.seed).fit(X)
            self.means_ = km.cluster_centers_.copy()
            self.covars_ = np.array([np.cov(X[km.labels_ == j].T) + self.reg_covar * np.eye(d) for j in range(K)])
            self.startprob_ = np.full(K, 1.0 / K)
            self.transmat_ = np.full((K, K), 0.05 / (K - 1))
            np.fill_diagonal(self.transmat_, 0.95)
        else:
            self.means_, self.covars_ = init.means_.copy(), init.covars_.copy()
            self.startprob_, self.transmat_ = init.startprob_.copy(), init.transmat_.copy()

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            logB = self._log_emissions(X)
            log_alpha = self._forward(logB)
            log_beta = self._backward(logB)
            ll = logsumexp(log_alpha[-1])
            gamma = np.exp(log_alpha + log_beta - ll)
            log_xi_sum = logsumexp(log_alpha[:-1, :, None] + np.log(self.transmat_)[None, :, :]
                                   + (logB[1:] + log_beta[1:])[:, None, :] - ll, axis=0)
            A = np.clip(np.exp(log_xi_sum), 1e-8, None)
            self.transmat_ = A / A.sum(axis=1, keepdims=True)
            self.startprob_ = np.clip(gamma[0], 1e-6, None)
            self.startprob_ /= self.startprob_.sum()
            w = gamma.sum(axis=0)
            self.means_ = (gamma.T @ X) / w[:, None]
            for j in range(K):
                diff = X - self.means_[j]
                self.covars_[j] = (gamma[:, j, None] * diff).T @ diff / w[j] + self.reg_covar * np.eye(d)
            if ll - prev_ll < self.tol:
                break
            prev_ll = ll
        return self

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """P(S_t | x_1..x_t) para cada t: causal, es lo que se usa para operar."""
        log_alpha = self._forward(self._log_emissions(X))
        return np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Camino más probable S_1..S_T dado x_1..x_T: usa el futuro, solo para comparar."""
        logB = self._log_emissions(X)
        T, K = logB.shape
        logA = np.log(self.transmat_)
        delta = np.empty((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.startprob_) + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path


class HMMModel(_Standardized):
    """HMM gaussiano sobre las variables estandarizadas; `predict` regresa la etiqueta filtrada."""

    def __init__(self, n_states: int = 3, seed: int = SEED):
        self.n_states = n_states
        self.seed = seed

    def fit(self, F: pd.DataFrame, init: "HMMModel" = None) -> "HMMModel":
        self._set_scale(F[FEATURES])
        self.hmm_ = GaussianHMM(self.n_states, seed=self.seed).fit(
            self.standardize(F), init=None if init is None else init.hmm_)
        self.names_ = name_states(pd.DataFrame(self.hmm_.means_, columns=FEATURES))
        return self

    def predict(self, F: pd.DataFrame) -> pd.Series:
        states = self.hmm_.filter_proba(self.standardize(F)).argmax(axis=1)
        return pd.Series([self.names_[k] for k in states], index=F.index, name="regime")

    def predict_viterbi(self, F: pd.DataFrame) -> pd.Series:
        states = self.hmm_.viterbi(self.standardize(F))
        return pd.Series([self.names_[k] for k in states], index=F.index, name="regime")

    def expected_durations(self) -> dict:
        """E[duración del estado j] = 1 / (1 - A_jj), en días."""
        d = 1.0 / (1.0 - np.diag(self.hmm_.transmat_))
        return {self.names_[j]: float(d[j]) for j in range(self.n_states)}


MODELS = {"rules": RulesModel, "kmeans": KMeansModel, "hmm": HMMModel}


def causal_labels(F: pd.DataFrame, method: str = "kmeans", min_fit_obs: int = MIN_FIT_OBS,
                  seed: int = SEED) -> pd.Series:
    """Etiquetas causales: el modelo de cada mes se ajusta con datos hasta el cierre del mes anterior.

    Reglas y K-means etiquetan punto a punto. El HMM filtra hacia adelante desde
    el inicio de la muestra hasta el fin del mes con los parámetros de ese mes.
    """
    labels = pd.Series(np.nan, index=F.index, dtype=object, name="regime")
    months = F.index.to_period("M")
    model = None
    for month in months.unique():
        in_month = months == month
        month_days = F.index[in_month]
        history = F.loc[F.index < month_days[0]]
        if len(history) < min_fit_obs:
            continue
        model = MODELS[method](seed=seed).fit(history, init=model if method == "hmm" else None)
        if method == "hmm":
            labels[in_month] = model.predict(F.loc[:month_days[-1]]).loc[month_days].to_numpy()
        else:
            labels[in_month] = model.predict(F.loc[in_month]).to_numpy()
    return labels


def smooth_labels(raw: pd.Series, confirm_days: int = CONFIRM_DAYS) -> pd.Series:
    """Histéresis causal: cambia de régimen solo tras `confirm_days` etiquetas nuevas consecutivas."""
    out = pd.Series(np.nan, index=raw.index, dtype=object, name="regime")
    current, candidate, streak = None, None, 0
    for t, lab in raw.items():
        if not isinstance(lab, str):
            continue
        if current is None:
            current = lab
        elif lab == current:
            candidate, streak = None, 0
        else:
            streak = streak + 1 if lab == candidate else 1
            candidate = lab
            if streak >= confirm_days:
                current, candidate, streak = lab, None, 0
        out[t] = current
    return out


def run_lengths(labels: pd.Series) -> pd.DataFrame:
    """Rachas consecutivas de cada régimen: régimen, inicio, fin y duración en días hábiles."""
    lab = labels.dropna()
    block = (lab != lab.shift()).cumsum()
    runs = lab.groupby(block).agg(["first", "size"])
    starts = lab.index.to_series().groupby(block).first()
    ends = lab.index.to_series().groupby(block).last()
    return pd.DataFrame({"regime": runs["first"].to_numpy(), "start": starts.to_numpy(),
                         "end": ends.to_numpy(), "days": runs["size"].to_numpy()})


def regime_validation(labels: pd.Series, train_mask: np.ndarray) -> dict:
    """Persistencia, transiciones y proporción de tiempo en train/test de unas etiquetas."""
    lab = labels.dropna()
    runs = run_lengths(lab)
    years = len(lab) / 252
    mask = pd.Series(train_mask, index=labels.index).reindex(lab.index)
    return {
        "avg_duration_days": float(runs["days"].mean()),
        "avg_duration_by_regime": runs.groupby("regime")["days"].mean().to_dict(),
        "n_transitions": int(len(runs) - 1),
        "transitions_per_year": float((len(runs) - 1) / years),
        "share_train": lab[mask].value_counts(normalize=True).to_dict(),
        "share_test": lab[~mask].value_counts(normalize=True).to_dict(),
        "transition_matrix": pd.crosstab(lab.shift().dropna(), lab.iloc[1:], normalize="index").round(4).to_dict(),
    }


def train_fit_stats(F_train: pd.DataFrame, method: str, seed: int = SEED) -> dict:
    """Ajuste de un método sobre todo el train: silhouette, persistencia y participación en muestra."""
    model = MODELS[method](seed=seed).fit(F_train)
    lab = model.predict(F_train)
    runs = run_lengths(lab)
    Z = model.standardize(F_train)
    out = {
        "silhouette": float(silhouette_score(Z, lab)) if lab.nunique() > 1 else np.nan,
        "avg_duration_days": float(runs["days"].mean()),
        "transitions_per_year": float((len(runs) - 1) / (len(lab) / 252)),
        **{f"share_{r}": float((lab == r).mean()) for r in REGIMES},
    }
    if method == "hmm":
        out["expected_duration_days"] = model.expected_durations()
        out["viterbi_disagreement"] = float((model.predict_viterbi(F_train) != lab).mean())
    return out
