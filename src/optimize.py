"""Optimización de hiperparámetros (Optuna TPE) y walk-forward analysis.

Walk-forward (nivel C): entrenamiento de 6 meses, prueba de 1 mes, paso
mensual, SOLO dentro del periodo de entrenamiento (meses de prueba 2019-01 a
2023-12). En cada ventana y para cada activo:

- θ global: 100 pruebas TPE maximizando el Calmar del activo en los 6 meses.
- θ por régimen: 100 pruebas TPE por régimen, evaluando solo entradas en los
  días de ese régimen (la señal se anula fuera de él). Si el régimen ocupa
  menos de MIN_REGIME_DAYS días de la ventana, o si ninguna configuración es
  factible, se usa el θ global de la ventana (fallback).

Restricciones de factibilidad (una configuración infactible nunca le gana a una válida):
- Mínimo de operaciones: si abre menos de N_MIN_TRADES (N_MIN_TRADES_REGIME en la
  optimización por régimen), su valor es -1000 + n_operaciones.
- Indicadores no redundantes: si algún par de votos tiene |ρ| ≥ MAX_VOTE_CORR en
  la ventana de entrenamiento, su valor es -500 - max|ρ|.

Periodo de prueba: el walk-forward deja un θ por ventana; para cada activo (y
cada régimen) se promedian esos θ (los enteros se redondean) y el θ promedio,
congelado, se usa sin cambios en todo el periodo de prueba (2024-01 a 2026-09).

Calmar de la optimización: con un piso de 1% en el MDD (MDD_FLOOR) para que
una ventana corta sin pérdidas no dé un Calmar infinito.

Semillas: cada estudio usa TPESampler(seed = SEED + 1000·ventana + 10·activo + régimen).

Defecto conocido del Calmar como objetivo: con retorno anualizado R < 0, Calmar = R / MDD
se acerca a 0 cuando el MDD crece, así que maximizarlo puede preferir perder más con un
drawdown mayor. Solo actúa cuando TODAS las configuraciones factibles de un estudio pierden.
Se conserva el Calmar porque el laboratorio lo exige; `audit_negative_calmar` mide cuántos
estudios caen en ese caso y qué eligió el optimizador (results/calmar_audit.csv).
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from .backtest import single_asset_backtest
from .metrics import annualized_return, calmar_ratio, max_drawdown
from .regimes import REGIMES
from .signals import DEFAULT_PARAMS, VOTE_COLUMNS, generate_signals, hold_direction, max_vote_correlation

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS = 100
N_MIN_TRADES = 4
N_MIN_TRADES_REGIME = 2
MIN_REGIME_DAYS = 21
MDD_FLOOR = 0.01
TRAIN_MONTHS = 6
TEST_MONTHS = 1
FIRST_TEST_MONTH = "2019-01"
LAST_WF_MONTH = "2023-12"   # el walk-forward termina con el periodo de entrenamiento
MAX_VOTE_CORR = 0.5   # 0.4 dejaba sin θ factible 7.6% de los estudios por correlación
STRATEGY_VERSION = "3de6-rsi-zona-autocorr-relajada-corr0.5-fallback-infactible"  # cambia la llave del caché si cambia la estrategia

PARAM_SPACE = {
    "adx_window": ("int", 7, 30),
    "adx_threshold": ("float", 15.0, 35.0),
    "rsi_window": ("int", 5, 30),
    "rsi_level": ("float", 65.0, 85.0),
    "rsi_tol": ("float", 0.0, 10.0),
    "rsi_k": ("int", 1, 10),
    "obv_window": ("int", 5, 40),
    "obv_threshold": ("float", 0.0, 0.4),
    "feat_window": ("int", 10, 40),
    "autocorr_window": ("int", 20, 90),
    "atr_window": ("int", 7, 30),
    "sl_mult": ("float", 1.0, 4.0),
    "tp_mult": ("float", 1.0, 6.0),
    "max_holding": ("int", 5, 60),
    "position_frac": ("float", 0.2, 1.0),
}
INT_PARAMS = [k for k, v in PARAM_SPACE.items() if v[0] == "int"]


def sample_params(trial: optuna.Trial) -> dict:
    """Muestrea un θ del espacio de búsqueda."""
    p = {}
    for name, (kind, lo, hi) in PARAM_SPACE.items():
        p[name] = trial.suggest_int(name, lo, hi) if kind == "int" else trial.suggest_float(name, lo, hi)
    return p


def evaluate(df: pd.DataFrame, params: dict, start, end, mask: pd.Series = None, ticker: str = "asset",
             signal_column: str = "signal") -> dict:
    """Backtest de un activo entre start y end con indicadores calculados sobre toda la historia hasta end.

    mask: Series booleana; las señales fuera de la máscara se anulan (optimización por régimen).
    signal_column: "signal" (regla 3 de 6) o un voto individual (p. ej. "v_adx").
    """
    p = {**DEFAULT_PARAMS, **params}
    sig = generate_signals(df.loc[:end], p).loc[start:end]
    raw = sig[signal_column]
    signal = raw if mask is None else raw.where(mask.reindex(sig.index, fill_value=False), 0)
    direction = pd.Series(hold_direction(signal, p["max_holding"]), index=sig.index)
    res = single_asset_backtest(df.loc[start:end], sig, p, ticker=ticker, direction=direction)
    res["n_entries"] = int((res["fills"]["reason"] == "entry").sum())
    res["calmar_opt"] = calmar_ratio(res["equity"], MDD_FLOOR)
    res["max_vote_corr"] = max_vote_correlation(sig[VOTE_COLUMNS])
    return res


def penalized_value(res: dict, n_min: int) -> float:
    """Calmar si la configuración es factible; si no, una penalización que nunca le gana a una factible."""
    if res["n_entries"] < n_min or not np.isfinite(res["calmar_opt"]):
        return -1000.0 + res["n_entries"]
    if res["max_vote_corr"] >= MAX_VOTE_CORR:
        return -500.0 - res["max_vote_corr"]
    return res["calmar_opt"]


def optimize_params(df: pd.DataFrame, start, end, mask: pd.Series = None, n_trials: int = N_TRIALS,
                    n_min: int = N_MIN_TRADES, seed: int = SEED) -> dict:
    """Estudio TPE que maximiza el Calmar con restricción de operaciones mínimas."""
    def objective(trial):
        res = evaluate(df, sample_params(trial), start, end, mask)
        trial.set_user_attr("n_entries", res["n_entries"])
        trial.set_user_attr("annual_return", annualized_return(res["equity"]))
        trial.set_user_attr("max_vote_corr", res["max_vote_corr"])
        return penalized_value(res, n_min)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial
    return {"params": {**DEFAULT_PARAMS, **best.params}, "is_calmar": best.value,
            "is_annual_return": best.user_attrs["annual_return"], "n_entries": best.user_attrs["n_entries"],
            "max_vote_corr": best.user_attrs["max_vote_corr"], "n_trials": len(study.trials)}


def make_windows(index: pd.DatetimeIndex, first_test_month: str = FIRST_TEST_MONTH,
                 last_test_month: str = LAST_WF_MONTH, train_months: int = TRAIN_MONTHS,
                 test_months: int = TEST_MONTHS) -> list:
    """Ventanas mensuales: train = `train_months` meses previos al mes de prueba."""
    months = index.to_period("M")
    windows = []
    for k, test_m in enumerate(pd.period_range(first_test_month, last_test_month, freq=f"{test_months}M")):
        train_dates = index[(months >= test_m - train_months) & (months < test_m)]
        test_dates = index[(months >= test_m) & (months < test_m + test_months)]
        if len(test_dates) == 0:
            continue
        windows.append({"window": k, "train_start": train_dates[0], "train_end": train_dates[-1],
                        "test_start": test_dates[0], "test_end": test_dates[-1]})
    return windows


METHOD_SEED_OFFSET = {"kmeans": 0, "rules": 100, "hmm": 200}


def _optimize_asset_window(a: int, ticker: str, df: pd.DataFrame, w: dict, labels_by_method: dict,
                           n_trials: int) -> list:
    """θ global (sin régimen) y θ por régimen de cada método de un activo en una ventana de entrenamiento."""
    base_seed = SEED + 1000 * w["window"] + 10 * a
    glob = optimize_params(df, w["train_start"], w["train_end"], n_trials=n_trials, seed=base_seed)
    records = [{"ticker": ticker, **w, "method": "none", "regime": "global", "fallback": False,
                "infeasible": glob["is_calmar"] <= -400, **glob}]
    for method, labels in labels_by_method.items():
        train_labels = labels.loc[w["train_start"]:w["train_end"]]
        for r_i, regime in enumerate(REGIMES, start=1):
            mask = train_labels == regime
            row = {"ticker": ticker, **w, "method": method, "regime": regime}
            if mask.sum() >= MIN_REGIME_DAYS:
                res = optimize_params(df, w["train_start"], w["train_end"], mask=mask, n_trials=n_trials,
                                      n_min=N_MIN_TRADES_REGIME,
                                      seed=base_seed + METHOD_SEED_OFFSET.get(method, 300) + r_i)
                if res["is_calmar"] > -400:
                    records.append({**row, "fallback": False, "infeasible": False, **res})
                else:  # ninguna configuración factible: se usa el θ global (se conserva el conteo de pruebas)
                    records.append({**row, "fallback": True, "infeasible": True, **glob, "n_trials": res["n_trials"]})
            else:
                records.append({**row, "fallback": True, "infeasible": False, **glob, "n_trials": 0})
    return records


def _cache_key(prices: dict, labels_by_method: dict, n_trials: int) -> str:
    """Huella de la configuración: si cambian activos, etiquetas o pruebas, no se reutiliza el caché."""
    parts = [STRATEGY_VERSION, str(n_trials), ",".join(prices),
             *[f"{m}:{lab.fillna('').str.cat()}" for m, lab in labels_by_method.items()]]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def run_walk_forward(prices: dict, labels_by_method: dict, windows: list, n_trials: int = N_TRIALS,
                     n_jobs: int = -1, cache_dir=None, batch: int = 8) -> tuple:
    """Optimiza todas las ventanas y activos en paralelo. Regresa (DataFrame de registros, segundos de cómputo).

    labels_by_method: {método: etiquetas de régimen}. El θ global (method = "none") se optimiza una
    sola vez por activo y ventana; los θ por régimen, una vez por método.
    cache_dir: si se da, cada bloque de `batch` ventanas se guarda ahí y una corrida interrumpida
    retoma desde el último bloque terminado. El caché se invalida si cambian activos, etiquetas,
    número de pruebas o STRATEGY_VERSION (subirla al cambiar la estrategia). Junto a cada
    bloque se guarda cuántos segundos tardó en calcularse, así que el tiempo reportado es el de
    cómputo aunque los bloques vengan del caché.
    """
    t0 = time.time()
    compute_seconds = 0.0
    key = _cache_key(prices, labels_by_method, n_trials)
    chunks = []
    for start in range(0, len(windows), batch):
        block = windows[start:start + batch]
        path = None if cache_dir is None else Path(cache_dir) / f"wf_{key}_{start:03d}.pkl"
        if path is not None and path.exists():
            chunks.append(pd.read_pickle(path))
            timing = path.with_suffix(".seconds")
            compute_seconds += float(timing.read_text()) if timing.exists() else 0.0
            continue
        t_block = time.time()
        jobs = [delayed(_optimize_asset_window)(a, t, prices[t], w, labels_by_method, n_trials)
                for a, t in enumerate(prices) for w in block]
        out = Parallel(n_jobs=n_jobs, verbose=0)(jobs)
        df = pd.DataFrame([r for chunk in out for r in chunk])
        compute_seconds += time.time() - t_block
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_pickle(path)
            path.with_suffix(".seconds").write_text(f"{time.time() - t_block:.1f}")
        chunks.append(df)
        print(f"  [walk-forward] ventanas {start + len(block)}/{len(windows)} ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    return pd.concat(chunks, ignore_index=True), compute_seconds


def oos_signals(prices: dict, labels: pd.Series, records: pd.DataFrame, windows: list,
                method: str = None) -> dict:
    """Señales fuera de muestra: en cada día de prueba se usa el θ del régimen vigente.

    method: método de régimen cuyos θ se usan; None = modelo sin régimen (θ global siempre).
    Regresa dict de DataFrames (fechas OOS x activos): signal, strength, un voto por
    indicador, atr, sl_mult, tp_mult, max_holding, position_frac.
    """
    fields = ["signal", "strength", *VOTE_COLUMNS, "atr"]
    param_fields = ["sl_mult", "tp_mult", "max_holding", "position_frac"]
    oos_index = pd.DatetimeIndex([d for w in windows for d in
                                  prices[next(iter(prices))].loc[w["test_start"]:w["test_end"]].index])
    out = {k: pd.DataFrame(np.nan, index=oos_index, columns=list(prices)) for k in fields + param_fields}
    rec = records[records["method"].isin(["none", method])].set_index(["ticker", "window", "regime"])
    for ticker, df in prices.items():
        for w in windows:
            test_dates = df.loc[w["test_start"]:w["test_end"]].index
            if method is None:
                regime_today = pd.Series("global", index=test_dates)
            else:
                regime_today = labels.reindex(test_dates).fillna("global")
            for regime in regime_today.unique():
                params = rec.loc[(ticker, w["window"], regime), "params"]
                days = test_dates[regime_today == regime]
                sig = generate_signals(df.loc[:w["test_end"]], params).loc[days]
                for k in fields:
                    out[k].loc[days, ticker] = sig[k]
                for k in param_fields:
                    out[k].loc[days, ticker] = params[k]
    return out


def average_params(param_list: list) -> dict:
    """Promedio de varios θ: media de cada parámetro; los enteros se redondean."""
    out = {}
    for name in PARAM_SPACE:
        value = float(np.mean([p[name] for p in param_list]))
        out[name] = int(round(value)) if name in INT_PARAMS else value
    return {**DEFAULT_PARAMS, **out}


def average_thetas(records: pd.DataFrame, method: str = None) -> dict:
    """θ promedio por activo a partir del walk-forward de entrenamiento.

    Regresa {(ticker, régimen): θ}. "global" promedia los θ globales factibles de todas las ventanas. Para cada
    régimen se promedian solo las ventanas donde ese régimen se optimizó (sin fallback); si nunca se
    optimizó, se usa el θ global promedio.
    """
    out = {}
    for ticker, g in records.groupby("ticker"):
        glob = g[g["method"] == "none"]
        feasible = glob[~glob["infeasible"]] if (~glob["infeasible"]).any() else glob
        out[(ticker, "global")] = average_params(list(feasible["params"]))
        if method is None:
            continue
        for regime in REGIMES:
            own = g[(g["method"] == method) & (g["regime"] == regime) & (~g["fallback"])]
            out[(ticker, regime)] = average_params(list(own["params"])) if len(own) else out[(ticker, "global")]
    return out


def frozen_signals(prices: dict, labels: pd.Series, thetas: dict, dates: pd.DatetimeIndex,
                   method: str = None) -> dict:
    """Señales del periodo de prueba con θ promedio congelados (mismas columnas que `oos_signals`).

    En cada día se usa el θ promedio del régimen vigente (o el global si method es None). Los
    indicadores se calculan con toda la historia disponible hasta el último día de `dates`.
    """
    fields = ["signal", "strength", *VOTE_COLUMNS, "atr"]
    param_fields = ["sl_mult", "tp_mult", "max_holding", "position_frac"]
    out = {k: pd.DataFrame(np.nan, index=dates, columns=list(prices)) for k in fields + param_fields}
    regime_today = (pd.Series("global", index=dates) if method is None
                    else labels.reindex(dates).fillna("global"))
    for ticker, df in prices.items():
        for regime in regime_today.unique():
            params = thetas[(ticker, regime)]
            days = dates[regime_today == regime]
            sig = generate_signals(df.loc[:dates[-1]], params).loc[days]
            for k in fields:
                out[k].loc[days, ticker] = sig[k]
            for k in param_fields:
                out[k].loc[days, ticker] = params[k]
    return out


def train_wf_then_frozen_test(prices: dict, labels: pd.Series, records: pd.DataFrame, windows: list,
                              test_dates: pd.DatetimeIndex, method: str = None) -> tuple:
    """Señales completas: walk-forward fuera de muestra en entrenamiento + θ promedio congelado en prueba.

    Regresa (señales, θ promedio usados en la prueba).
    """
    wf = oos_signals(prices, labels, records, windows, method)
    thetas = average_thetas(records, method)
    frozen = frozen_signals(prices, labels, thetas, test_dates, method)
    return {k: pd.concat([wf[k], frozen[k]]) for k in wf}, thetas


def vote_correlation(signals: dict, mask: np.ndarray = None) -> pd.DataFrame:
    """Matriz de correlación de los votos con los activos apilados (opcionalmente solo en `mask`)."""
    stacked = pd.DataFrame({c: signals[c].to_numpy()[mask if mask is not None else slice(None)].ravel()
                            for c in VOTE_COLUMNS}).fillna(0.0)
    return stacked.corr()


def sensitivity(df: pd.DataFrame, params: dict, start, end, deltas=(-0.2, -0.1, 0.0, 0.1, 0.2)) -> pd.DataFrame:
    """Calmar al variar cada parámetro del θ óptimo en ±10% y ±20%, uno a la vez."""
    rows = []
    for name in PARAM_SPACE:
        for delta in deltas:
            p = dict(params)
            value = params[name] * (1 + delta)
            p[name] = int(round(value)) if name in INT_PARAMS else value
            if name == "position_frac":
                p[name] = min(p[name], 1.0)
            res = evaluate(df, p, start, end)
            rows.append({"param": name, "delta": delta, "value": p[name],
                         "calmar": calmar_ratio(res["equity"]), "n_entries": res["n_entries"]})
    return pd.DataFrame(rows)


def _replay_study(df: pd.DataFrame, row: dict, mask: pd.Series, n_min: int, seed: int, n_trials: int) -> dict:
    """Repite un estudio del walk-forward con la misma semilla guardando retorno y MDD de cada prueba."""
    trials = []

    def objective(trial):
        res = evaluate(df, sample_params(trial), row["train_start"], row["train_end"], mask)
        value = penalized_value(res, n_min)
        feasible = value > -400
        trials.append((res["calmar_opt"], annualized_return(res["equity"]), max_drawdown(res["equity"]), feasible))
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    T = pd.DataFrame(trials, columns=["calmar", "R", "mdd", "feasible"])
    T = T[T["feasible"]]
    best = T.loc[T["calmar"].idxmax()]
    losers = T[T["R"] < 0]
    least_loss = losers.loc[losers["R"].idxmax()]
    return {
        "ticker": row["ticker"], "window": row["window"], "method": row["method"], "regime": row["regime"],
        "reproduces": bool(np.isclose(best["calmar"], row["is_calmar"])),
        "best_calmar": best["calmar"], "best_R": best["R"], "best_mdd": best["mdd"],
        "n_losing_trials": len(losers), "median_mdd_losers": losers["mdd"].median(),
        "mdd_percentile_of_best": float((losers["mdd"] < best["mdd"]).mean()),
        "corr_calmar_mdd_losers": losers[["calmar", "mdd"]].corr().iloc[0, 1],
        "least_loss_R": least_loss["R"], "least_loss_mdd": least_loss["mdd"],
        "picked_least_loss": bool(np.isclose(best["R"], least_loss["R"])),
    }


def audit_negative_calmar(prices: dict, records: pd.DataFrame, labels_by_method: dict,
                          n_trials: int = N_TRIALS, n_jobs: int = -1) -> pd.DataFrame:
    """Audita el defecto del Calmar con retorno negativo.

    Con R < 0, Calmar = R / MDD se acerca a 0 cuando el MDD crece: maximizarlo puede preferir
    configuraciones que pierden más con drawdowns mayores. Solo puede actuar en estudios donde
    TODAS las configuraciones factibles pierden (mejor Calmar < 0). Esta función repite cada uno de
    esos estudios con su semilla original y reporta si el elegido tiene un MDD alto entre las
    configuraciones perdedoras y si eligió la que menos perdía.
    """
    neg = records[(~records["fallback"]) & (records["is_calmar"] < 0) & (records["is_calmar"] > -400)]
    tickers = list(prices)
    jobs = []
    for _, row in neg.iterrows():
        a = tickers.index(row["ticker"])
        seed = SEED + 1000 * row["window"] + 10 * a
        mask, n_min = None, N_MIN_TRADES
        if row["method"] != "none":
            seed += METHOD_SEED_OFFSET.get(row["method"], 300) + REGIMES.index(row["regime"]) + 1
            lab = labels_by_method[row["method"]].loc[row["train_start"]:row["train_end"]]
            mask, n_min = lab == row["regime"], N_MIN_TRADES_REGIME
        jobs.append(delayed(_replay_study)(prices[row["ticker"]], row.to_dict(), mask, n_min, seed, n_trials))
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(jobs))


def confirmation_vs_single(df: pd.DataFrame, params: dict, periods: dict) -> pd.DataFrame:
    """Operaciones y Calmar de la regla 3 de 6 frente a cada indicador por sí solo, con el mismo θ."""
    rows = []
    for rule in ["signal", *VOTE_COLUMNS]:
        for name, (start, end) in periods.items():
            res = evaluate(df, params, start, end, signal_column=rule)
            rows.append({"rule": "3 de 6" if rule == "signal" else rule[2:].upper(), "period": name,
                         "n_entries": res["n_entries"], "calmar": calmar_ratio(res["equity"]),
                         "total_return": res["equity"].iloc[-1] / res["equity"].iloc[0] - 1})
    return pd.DataFrame(rows)
