"""Indicadores técnicos y regla de confirmación 3 de 6.

Seis indicadores (los cinco de las actividades anteriores más el RSI), cada
uno emite un voto v_k,t ∈ {-1, 0, +1} con información hasta el cierre de t:

- Tendencia — ADX con ±DI: v = sign(+DI - -DI) si ADX > adx_threshold; si no, 0.
- Momento (reversión) — RSI con zona de tolerancia: el RSI entra a la zona de
  sobrecompra al pasar del nivel L y se considera dentro mientras no baje de
  L - b (tolerancia, para que un regreso a L - 1 no lo saque). Si permanece
  k días seguidos en la zona vota -1 (corto, reversión), y sigue votando
  mientras siga en ella. Simétrico en sobreventa: entra bajo 100 - L, sigue
  dentro mientras no suba de 100 - L + b, y a los k días vota +1.
- Volumen — desbalance de OBV en `obv_window` barras:
      x_t = Σ sign(ΔC)·V / Σ V ∈ [-1, 1];  v = +1 si x > obv_threshold, -1 si x < -obv_threshold.
- Distribución — skew de los últimos `feat_window` retornos: +1 si > 0.5, -1 si < -0.5
  (0.5 ≈ un error estándar del skew con 20 observaciones).
- Régimen de autocorrelación — con ρ1 la autocorrelación de rezago 1 de los
  últimos `autocorr_window` retornos, vota solo si es significativa
  (|ρ1| > 1.96/√n) y los dos últimos retornos van en la misma dirección; con
  ρ1 > 0 sigue esa racha y con ρ1 < 0 va en contra. (La versión de las
  actividades anteriores exigía además que ρ2 y ρ3 tuvieran el signo de ρ1; en
  acciones diarias votaba solo 0.3% de los días, así que se relajó y su
  ventana se optimiza.)
- Volumen-momento — si el z-score del volumen de `feat_window` días es > 1,
  vota el signo del retorno del día.

Regla de confirmación (N⁺_t y N⁻_t = número de votos +1 y -1 en t):

      s_t = 1{N⁺_t ≥ 3 y N⁺_t > N⁻_t} - 1{N⁻_t ≥ 3 y N⁻_t > N⁺_t}

Un empate 3 contra 3 no abre posición.
Fuerza de la señal: f_t = (N⁺_t - N⁻_t) / 6.

Causalidad: todo se calcula con rolling, diff, el suavizado de Wilder (ewm con
adjust=False) o recursiones hacia adelante (zona del RSI), así que el valor en
t usa solo barras <= t. La señal de t se ejecuta al open de t+1 en el backtest.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

VOTE_COLUMNS = ["v_adx", "v_rsi", "v_obv", "v_skew", "v_autocorr", "v_volz"]
MIN_AGREE = 3
SKEW_THRESHOLD = 0.5
VOLZ_THRESHOLD = 1.0
AUTOCORR_Z = 1.96

DEFAULT_PARAMS = {
    "adx_window": 14, "adx_threshold": 25.0,
    "rsi_window": 14, "rsi_level": 70.0, "rsi_tol": 5.0, "rsi_k": 3,
    "obv_window": 20, "obv_threshold": 0.2,
    "feat_window": 20, "autocorr_window": 40,
    "atr_window": 14,
    "sl_mult": 2.0, "tp_mult": 3.0,
    "max_holding": 20,
    "position_frac": 1.0,
}


def _wilder(x: pd.Series, window: int) -> pd.Series:
    """Media de Wilder: EMA con alpha = 1/window."""
    return x.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Average True Range de Wilder."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"],
                            (df["high"] - prev_close).abs(),
                            (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return _wilder(true_range, window)


def adx(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """ADX, +DI y -DI de Wilder."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, window)
    plus_di = 100 * _wilder(plus_dm, window) / atr_
    minus_di = 100 * _wilder(minus_dm, window) / atr_
    di_sum = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0).where(plus_di.notna())
    return pd.DataFrame({"adx": _wilder(dx, window), "plus_di": plus_di, "minus_di": minus_di})


def rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI de Wilder en [0, 100]."""
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), window)
    avg_loss = _wilder((-delta).clip(lower=0), window)
    return 100 - 100 / (1 + avg_gain / avg_loss)


def obv_imbalance(df: pd.DataFrame, window: int) -> pd.Series:
    """Desbalance de On-Balance Volume: fracción neta del volumen con cierre al alza."""
    signed_volume = np.sign(df["close"].diff()).fillna(0.0) * df["volume"]
    return signed_volume.rolling(window).sum() / df["volume"].rolling(window).sum()


def _windows(x: pd.Series, window: int) -> np.ndarray:
    """Ventanas deslizantes (n x window) que terminan en cada t; las primeras window-1 filas son NaN."""
    arr = x.to_numpy(dtype=float)
    out = np.full((len(arr), window), np.nan)
    if len(arr) >= window:
        out[window - 1:] = sliding_window_view(arr, window)
    return out


def rolling_autocorr(ret: pd.Series, window: int, lag: int) -> pd.Series:
    """Autocorrelación de rezago `lag` dentro de cada ventana de `window` retornos."""
    w = _windows(ret, window)
    a, b = w[:, :-lag], w[:, lag:]
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (a_c * b_c).sum(axis=1) / np.sqrt((a_c ** 2).sum(axis=1) * (b_c ** 2).sum(axis=1))
    return pd.Series(rho, index=ret.index)


def compute_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Indicadores crudos más el ATR que usan los stops."""
    p = {**DEFAULT_PARAMS, **params}
    n = int(p["feat_window"])
    ret = df["close"].pct_change()
    ind = adx(df, int(p["adx_window"]))
    ind["rsi"] = rsi(df["close"], int(p["rsi_window"]))
    ind["obv_imb"] = obv_imbalance(df, int(p["obv_window"]))
    ind["skew"] = ret.rolling(n).skew()
    n_ac = int(p["autocorr_window"])
    ind["rho1"] = rolling_autocorr(ret, n_ac, 1)
    ind["rho_crit"] = AUTOCORR_Z / np.sqrt(n_ac)
    ind["ret"] = ret
    ind["ret_lag1"] = ret.shift(1)
    vol = df["volume"]
    ind["vol_z"] = (vol - vol.rolling(n).mean()) / vol.rolling(n).std()
    ind["atr"] = atr(df, int(p["atr_window"]))
    return ind


def rsi_zone_votes(rsi_values, level: float, tol: float, k: int) -> np.ndarray:
    """Voto del RSI con zona de tolerancia y permanencia mínima.

    Sobrecompra: entra con RSI > level, sigue dentro mientras RSI > level - tol;
    tras k días seguidos dentro vota -1. Sobreventa: entra con RSI < 100 - level,
    sigue dentro mientras RSI < 100 - level + tol; tras k días vota +1.
    """
    r = np.asarray(rsi_values, dtype=float)
    upper, lower = level, 100.0 - level
    votes = np.zeros(len(r))
    up_days = low_days = 0
    for t, x in enumerate(r):
        if np.isnan(x):
            up_days = low_days = 0
            continue
        if up_days > 0:
            up_days = up_days + 1 if x > upper - tol else 0
        elif x > upper:
            up_days = 1
        if low_days > 0:
            low_days = low_days + 1 if x < lower + tol else 0
        elif x < lower:
            low_days = 1
        if up_days >= k:
            votes[t] = -1.0
        elif low_days >= k:
            votes[t] = 1.0
    return votes


def max_vote_correlation(votes: pd.DataFrame) -> float:
    """Máximo |ρ| entre pares de votos (un voto constante no se correlaciona con nada)."""
    varying = votes.loc[:, votes.std() > 0]
    if varying.shape[1] < 2:
        return 0.0
    c = varying.corr().to_numpy()
    return float(np.nanmax(np.abs(c[np.triu_indices_from(c, k=1)])))


def compute_votes(ind: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Votos {-1, 0, +1} de cada indicador; un indicador sin historia suficiente (NaN) vota 0."""
    p = {**DEFAULT_PARAMS, **params}
    v = pd.DataFrame(index=ind.index)
    v["v_adx"] = np.where(ind["adx"] > p["adx_threshold"], np.sign(ind["plus_di"] - ind["minus_di"]), 0.0)
    v["v_rsi"] = rsi_zone_votes(ind["rsi"], p["rsi_level"], p["rsi_tol"], int(p["rsi_k"]))
    v["v_obv"] = np.select([ind["obv_imb"] > p["obv_threshold"], ind["obv_imb"] < -p["obv_threshold"]], [1.0, -1.0], 0.0)
    v["v_skew"] = np.select([ind["skew"] > SKEW_THRESHOLD, ind["skew"] < -SKEW_THRESHOLD], [1.0, -1.0], 0.0)
    significant = ind["rho1"].abs() > ind["rho_crit"]
    last, prev = np.sign(ind["ret"]), np.sign(ind["ret_lag1"])
    streak = (last == prev) & (last != 0)
    v["v_autocorr"] = np.where(significant & streak, np.sign(ind["rho1"]) * last, 0.0)
    v["v_volz"] = np.where(ind["vol_z"] > VOLZ_THRESHOLD, np.sign(ind["ret"]), 0.0)
    return v.fillna(0.0).astype(int)


def confirm(votes: pd.DataFrame, min_agree: int = MIN_AGREE) -> pd.Series:
    """Regla de confirmación: s_t = 1{N⁺ ≥ min_agree y N⁺ > N⁻} - 1{N⁻ ≥ min_agree y N⁻ > N⁺}."""
    n_long = (votes == 1).sum(axis=1)
    n_short = (votes == -1).sum(axis=1)
    long_ = (n_long >= min_agree) & (n_long > n_short)
    short = (n_short >= min_agree) & (n_short > n_long)
    return (long_.astype(int) - short.astype(int)).rename("signal")


def signal_strength(votes: pd.DataFrame) -> pd.Series:
    """f_t = (N⁺_t - N⁻_t) / número de indicadores."""
    return ((votes == 1).sum(axis=1) - (votes == -1).sum(axis=1)).div(votes.shape[1]).rename("strength")


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Votos, señal confirmada, fuerza y ATR para un activo."""
    ind = compute_indicators(df, params)
    votes = compute_votes(ind, params)
    out = votes.copy()
    out["signal"] = confirm(votes)
    out["strength"] = signal_strength(votes)
    out["atr"] = ind["atr"]
    return out


def hold_direction(signal, max_holding) -> np.ndarray:
    """Convierte señales de entrada en la dirección que se quiere mantener.

    Una señal no nula distinta de la dirección actual abre una nueva dirección
    (edad 0). Una señal nula o igual mantiene la dirección mientras la edad sea
    menor a `max_holding` barras; al cumplirse, la dirección vuelve a 0 una
    barra (salida por holding máximo). `max_holding` puede ser escalar o un
    arreglo por barra (se usa el valor de la barra en que se abrió la dirección).
    """
    s = np.asarray(signal, dtype=float)
    mh = np.broadcast_to(np.asarray(max_holding, dtype=float), s.shape)
    out = np.zeros(len(s))
    d, age, limit = 0.0, 0, 0.0
    for t, st in enumerate(s):
        if st != 0 and st != d:
            d, age, limit = st, 0, mh[t]
        elif d != 0:
            age += 1
            if age >= limit:
                d = 0.0
        out[t] = d
    return out
