"""Ejecuta el proyecto completo con un solo comando: python main.py

Etapas: datos -> regímenes (3 clasificadores) -> walk-forward -> comparación de
clasificadores y elección con train -> backtests del portafolio ->
análisis (preguntas del laboratorio) -> tablas y figuras en results/.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import data, metrics, optimize, plots, portfolio, regimes
from src.backtest import COMMISSION
from src.signals import VOTE_COLUMNS

SEED = 42
RESULTS = Path(__file__).resolve().parent / "results"
FIG = RESULTS / "figures"
REBALANCE_EVERY = 5
REBALANCE_SWEEP = [1, 5, 10, 21, 63]
COST_SWEEP = [0.0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.004]
# Método de régimen: None = se elige solo con train (mayor Calmar del portafolio RP en 2019-2023).
# Para forzar uno: "rules", "kmeans" o "hmm".
REGIME_METHOD = None
ALLOCATIONS = ["rp", "naive", "ew"]
ALLOC_NAMES = {"rp": "Risk Parity", "naive": "Risk Parity naive (1/σ)", "ew": "Pesos iguales"}


def save_json(obj, name):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Datos ---------------------------------------------------------------
    if not all((data.DATA_DIR / f"{t}.csv").exists() for t in data.TICKERS):
        data.download_prices()
    prices = data.load_prices()
    data.validate_prices(prices)
    data.audit_prices(prices).to_csv(RESULTS / "data_audit.csv")
    index = prices[data.TICKERS[0]].index
    split = index[~data.is_train(index)][0]
    print(f"[datos] {len(index)} días, {index[0].date()} -> {index[-1].date()}, prueba desde {split.date()}")

    # 2. Regímenes: tres clasificadores ------------------------------------------
    market = regimes.market_index(prices)
    F = regimes.regime_features(market)
    train_F = F[data.is_train(F.index)]
    raw_labels = {m: regimes.causal_labels(F, m) for m in regimes.METHODS}
    labels_by_method = {m: regimes.smooth_labels(raw) for m, raw in raw_labels.items()}
    regime_compare = {}
    for m in regimes.METHODS:
        regime_compare[m] = {
            "train_fit": regimes.train_fit_stats(train_F, m),
            "causal_raw": regimes.regime_validation(raw_labels[m], data.is_train(F.index)),
            "causal_operated": regimes.regime_validation(labels_by_method[m], data.is_train(F.index)),
        }
    hmm_train = regimes.HMMModel().fit(train_F)
    viterbi_train = hmm_train.predict_viterbi(train_F)
    print("[regímenes] silhouette train:", {m: round(v["train_fit"]["silhouette"], 3) for m, v in regime_compare.items()})

    # 3. Walk-forward --------------------------------------------------------
    windows = optimize.make_windows(index)
    records, wf_seconds = optimize.run_walk_forward(prices, labels_by_method, windows, cache_dir=RESULTS / "cache")
    params_cols = pd.DataFrame(list(records["params"])).add_prefix("p_")
    records.drop(columns="params").join(params_cols).to_csv(RESULTS / "wf_params.csv", index=False)
    print(f"[walk-forward] {len(windows)} ventanas, {int(records['n_trials'].sum())} pruebas, {wf_seconds / 60:.1f} min")

    # 4. Comparación de métodos de régimen (portafolio Risk Parity) -------------
    # Walk-forward solo en entrenamiento; en prueba, θ promedio por activo (y régimen) congelado.
    test_dates = index[index >= split]
    oos_by, thetas_by = {}, {}
    for m in regimes.METHODS + ["none"]:
        oos_by[m], thetas_by[m] = optimize.train_wf_then_frozen_test(
            prices, labels_by_method.get(m), records, windows, test_dates, None if m == "none" else m)
    oos_index = oos_by["none"]["signal"].index
    res_bh = portfolio.buy_and_hold(prices, oos_index)
    rp_by = {}
    for m in regimes.METHODS + ["none"]:
        lab = labels_by_method.get(m)
        rp_by[m] = portfolio.run_portfolio(prices, oos_by[m], lab, portfolio.allocate(prices, oos_by[m], lab, "rp"),
                                           REBALANCE_EVERY)
    comparison_names = {m: f"Régimen · {regimes.METHOD_NAMES[m]}" for m in regimes.METHODS}
    comparison_names["none"] = "Sin régimen"
    comp_perf = pd.concat({comparison_names[m]: metrics.split_summary(r, split) for m, r in rp_by.items()}
                          | {"Buy & hold equiponderado": metrics.split_summary(res_bh, split)},
                          names=["strategy", "period"])
    comp_perf.to_csv(RESULTS / "regime_method_comparison.csv")
    for period in ("train", "test"):
        curves_p = {comparison_names[m]: r["equity"] for m, r in rp_by.items()}
        curves_p["Buy & hold equiponderado"] = res_bh["equity"]
        if period == "train":
            curves_p = {k: v[v.index < split] for k, v in curves_p.items()}
        else:
            curves_p = {k: v.iloc[v.index.get_indexer([split])[0] - 1:] for k, v in curves_p.items()}
        plots.plot_strategy_comparison(
            curves_p, FIG / f"16_regime_methods_{period}.png",
            f"{period.upper()}: Risk Parity con θ por régimen (3 clasificadores) vs. sin régimen y buy & hold")

    # Elección del método SOLO con train: mayor Calmar del portafolio en el periodo de entrenamiento.
    train_calmar = {m: float(comp_perf.loc[(comparison_names[m], "train"), "calmar"]) for m in regimes.METHODS}
    method = REGIME_METHOD or max(train_calmar, key=train_calmar.get)
    labels, oos = labels_by_method[method], oos_by[method]
    validation = {"selected_method": method, "selection_rule": "max Calmar del portafolio RP en train (2019-2023)",
                  "train_calmar_by_method": train_calmar, "methods": regime_compare,
                  "hmm_train_viterbi_disagreement": float((viterbi_train != hmm_train.predict(train_F)).mean())}
    save_json(validation, "regime_validation.json")
    save_json({f"{t}|{r}": th for (t, r), th in thetas_by[method].items()}, "theta_averaged.json")
    print(f"[regímenes] método elegido con train: {method} (Calmar train {train_calmar})")

    # 5. Backtests del método elegido: tres formas de asignación ----------------
    alloc = {a: portfolio.allocate(prices, oos, labels, a) for a in ALLOCATIONS}
    res = {a: portfolio.run_portfolio(prices, oos, labels, alloc[a], REBALANCE_EVERY) for a in ALLOCATIONS}
    res_rp = res["rp"]
    singles = portfolio.run_single_assets(prices, oos)

    curves = {ALLOC_NAMES[a]: res[a]["equity"] for a in ALLOCATIONS}
    curves["Buy & hold equiponderado"] = res_bh["equity"]
    summaries = {ALLOC_NAMES[a]: metrics.split_summary(res[a], split) for a in ALLOCATIONS}
    summaries["Buy & hold equiponderado"] = metrics.split_summary(res_bh, split)
    summaries.update({f"Activo {t}": metrics.split_summary(r, split) for t, r in singles.items()})
    perf = pd.concat(summaries, names=["strategy", "period"])
    perf.to_csv(RESULTS / "performance.csv")
    res_rp["trades"].to_csv(RESULTS / "trades_rp.csv", index=False)
    print(perf.loc[list(curves), ["annual_return", "sharpe", "sortino", "calmar", "max_drawdown", "win_rate"]].round(3))

    # 6. Análisis --------------------------------------------------------------
    # 6.1 θ promedio global por activo (el que se congela en prueba en el modelo sin régimen):
    #     sensibilidad ±20%, 3 de 6 contra cada indicador solo y desempeño en train/test
    theta_avg = {t: thetas_by["none"][(t, "global")] for t in data.TICKERS}
    periods = {"train": (index[0], index[data.is_train(index)][-1]), "test": (split, index[-1])}
    sens, conf, frozen = [], [], []
    for t, th in theta_avg.items():
        sens.append(optimize.sensitivity(prices[t], th, *periods["train"]).assign(ticker=t))
        conf.append(optimize.confirmation_vs_single(prices[t], th, periods).assign(ticker=t))
        for name, (s, e) in periods.items():
            ev = optimize.evaluate(prices[t], th, s, e)
            frozen.append({"ticker": t, "period": name, **metrics.summary(ev["equity"], ev["trades"], ev["fills"])})
    sens = pd.concat(sens)
    conf = pd.concat(conf)
    sens.to_csv(RESULTS / "sensitivity.csv", index=False)
    conf.to_csv(RESULTS / "confirmation_vs_single.csv", index=False)
    pd.DataFrame(frozen).to_csv(RESULTS / "frozen_theta_performance.csv", index=False)

    # 6.2 Degradación IS -> OOS en cada ventana (θ global de la ventana vs. su mes de prueba)
    glob = records[records["regime"] == "global"]
    deg = []
    for _, row in glob.iterrows():
        eq = singles[row["ticker"]]["equity"]
        pos = eq.index.get_loc(row["test_start"])
        month = eq.iloc[max(pos - 1, 0): eq.index.get_loc(row["test_end"]) + 1]
        deg.append({"ticker": row["ticker"], "test_start": row["test_start"],
                    "is_calmar": row["is_calmar"], "is_annual_return": row["is_annual_return"],
                    "oos_calmar": metrics.calmar_ratio(month, optimize.MDD_FLOOR),
                    "oos_annual_return": metrics.annualized_return(month),
                    "period": "train" if row["test_start"] < split else "test"})
    deg = pd.DataFrame(deg)
    deg.to_csv(RESULTS / "wf_degradation.csv", index=False)

    # 6.3 Costos de transacción y frecuencia de rebalanceo
    cost_rows = []
    for c in COST_SWEEP:
        for name in ALLOCATIONS:
            r = portfolio.run_portfolio(prices, oos, labels, alloc[name], REBALANCE_EVERY, commission=c)
            s = metrics.split_summary(r, split)
            cost_rows.append({"commission": c, "method": name, "annual_return_total": s.loc["total", "annual_return"],
                              "annual_return_test": s.loc["test", "annual_return"],
                              "calmar_total": s.loc["total", "calmar"], "total_costs": s.loc["total", "total_costs"]})
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(RESULTS / "cost_sweep.csv", index=False)
    rebal_rows = []
    for every in REBALANCE_SWEEP:
        for name in ALLOCATIONS:
            r = portfolio.run_portfolio(prices, oos, labels, alloc[name], every)
            s = metrics.summary(r["equity"], r["trades"], r["fills"])
            gross = r["equity"].iloc[-1] + s["total_costs"]
            rebal_rows.append({"every": every, "method": name, "annual_return": s["annual_return"],
                               "calmar": s["calmar"], "turnover": s["turnover"], "total_costs": s["total_costs"],
                               "gross_return": gross / r["equity"].iloc[0] - 1, "net_return": s["total_return"]})
    rebal = pd.DataFrame(rebal_rows)
    rebal.to_csv(RESULTS / "rebalance_sweep.csv", index=False)

    # 6.4 Desempeño por régimen y transiciones
    by_regime = {}
    for name, r in list(res.items()) + [("bh", res_bh)]:
        table, pvalue = metrics.metrics_by_regime(r["equity"], labels)
        table.to_csv(RESULTS / f"metrics_by_regime_{name}.csv")
        by_regime[name] = pvalue
    exits = res_rp["trades"]["exit_reason"].value_counts().to_dict()
    lab_oos = labels.reindex(oos_index)
    transitions_oos = int(((lab_oos != lab_oos.shift()) & lab_oos.shift().notna()).sum())

    # 6.5 Tres formas de asignación: contribuciones al riesgo
    returns = pd.DataFrame({t: prices[t]["close"].pct_change() for t in data.TICKERS})
    rc = {ALLOC_NAMES[a]: portfolio.realized_risk_shares(res[a]["positions"], res[a]["equity"], returns)
          for a in ALLOCATIONS}
    rc_mean = pd.DataFrame({k: v.mean() for k, v in rc.items()})
    rc_mean.to_csv(RESULTS / "risk_contributions.csv")

    # 6.6 Impacto de mercado estimado (no incluido en el backtest)
    impact = metrics.estimate_market_impact(res_rp["fills"], prices)
    years = (len(oos_index) - 1) / 252
    impact_summary = {
        "median_participation": float(impact["participation"].median()),
        "max_participation": float(impact["participation"].max()),
        "impact_cost_total": float(impact["impact_cost"].sum()),
        "impact_cost_annual_pct_equity": float(impact["impact_cost"].sum() / res_rp["equity"].mean() / years),
        "commission_total": float(res_rp["fills"]["commission"].sum()),
    }

    # 6.7 Correlaciones por régimen y fuerza de señal
    lab_ret = labels.reindex(returns.index)
    corrs = {g: returns[lab_ret == g].corr() for g in regimes.REGIMES if (lab_ret == g).sum() > 20}
    direction = portfolio.held_direction(oos)
    strength_held = portfolio.latched_strength(direction, oos["strength"].fillna(0))
    strength_monthly = strength_held.resample("ME").mean().T
    strength_monthly.columns = strength_monthly.columns.strftime("%Y-%m")

    # 6.8 Defecto del Calmar con retorno negativo: repite los estudios donde todo perdía
    calmar_audit = optimize.audit_negative_calmar(prices, records, labels_by_method)
    calmar_audit.to_csv(RESULTS / "calmar_audit.csv", index=False)
    calmar_audit_summary = metrics.calmar_audit_summary(records, calmar_audit)

    # 6.9 Validación: correlación entre los votos de los indicadores (|ρ| < 0.4)
    is_tr = np.asarray(oos_index < split)
    vote_corr = {"train": optimize.vote_correlation(oos, is_tr), "test": optimize.vote_correlation(oos, ~is_tr)}
    for name, c in vote_corr.items():
        c.to_csv(RESULTS / f"vote_correlation_{name}.csv")
    off_diag = lambda c: float(np.nanmax(np.abs(c.to_numpy()[np.triu_indices(len(c), k=1)])))  # noqa: E731
    vote_corr_summary = {
        "threshold": optimize.MAX_VOTE_CORR,
        "max_abs_train": off_diag(vote_corr["train"]), "max_abs_test": off_diag(vote_corr["test"]),
        "chosen_theta_max_window_corr": float(records.loc[~records["fallback"], "max_vote_corr"].max()),
        "vote_frequency": {c: float((oos[c] != 0).to_numpy().mean()) for c in VOTE_COLUMNS},
        "signal_frequency": float((oos["signal"] != 0).to_numpy().mean()),
    }

    # 7. Figuras ---------------------------------------------------------------
    plots.plot_equity(curves, FIG / "01_equity.png", "Valor del portafolio fuera de muestra (walk-forward)", split)
    plots.plot_drawdown(curves, FIG / "02_drawdown.png", "Drawdown del portafolio", split)
    plots.plot_return_table(metrics.monthly_table(res_rp["equity"]), FIG / "03_monthly_returns_rp.png",
                            "Retornos mensuales y anuales (%) — Risk Parity")
    plots.plot_return_table(metrics.quarterly_table(res_rp["equity"]), FIG / "03b_quarterly_returns_rp.png",
                            "Retornos trimestrales (%) — Risk Parity")
    plots.plot_sensitivity(sens, FIG / "04_sensitivity.png", "Sensibilidad del Calmar (train) a ±10% y ±20% de cada parámetro")
    plots.plot_cost_curve(costs.pivot(index="commission", columns="method", values="annual_return_total")
                          .rename(columns=ALLOC_NAMES),
                          FIG / "05_cost_curve.png", "Retorno neto anualizado contra comisión", COMMISSION)
    plots.plot_regime_timeline(market, labels, FIG / "06_regime_timeline.png",
                               f"Regímenes del método elegido ({regimes.METHOD_NAMES[method]}) sobre el índice", split)
    plots.plot_regime_methods(market, {regimes.METHOD_NAMES[m]: lab for m, lab in labels_by_method.items()},
                              FIG / "06b_regime_methods.png", "Regímenes causales de los tres clasificadores", split)
    plots.plot_regime_methods(market.loc[train_F.index],
                              {"HMM — etiqueta filtrada (se opera)": hmm_train.predict(train_F),
                               "HMM — Viterbi (usa el futuro, solo comparación)": viterbi_train},
                              FIG / "06c_hmm_filtered_vs_viterbi.png",
                              "HMM ajustado en train: etiqueta filtrada contra Viterbi")
    plots.plot_regime_distributions(F, labels, FIG / "07_regime_distributions.png",
                                    "Distribución de las variables por régimen")
    plots.plot_equity_regimes({ALLOC_NAMES[a]: res[a]["equity"] for a in ALLOCATIONS}, labels,
                              FIG / "08_equity_regimes.png", "Valor del portafolio con regímenes superpuestos", split)
    plots.plot_risk_contributions(rc_mean, FIG / "09_risk_contributions.png",
                                  "Contribución promedio al riesgo por activo")
    plots.plot_heatmap(strength_monthly, FIG / "10_signal_heatmap.png",
                       "Fuerza de la señal mantenida por activo (promedio mensual)", "Mes", "Activo", "Fuerza (−1 a +1)")
    plots.plot_correlation_by_regime(corrs, FIG / "11_correlation_regimes.png",
                                     "Correlación de retornos diarios por régimen")
    plots.plot_lines(rebal.pivot(index="every", columns="method", values="net_return")
                     .rename(columns=ALLOC_NAMES),
                     FIG / "12_rebalance_sweep.png", "Retorno neto total contra frecuencia de rebalanceo",
                     "Días entre rebalanceos", "Retorno neto total (%)")
    plots.plot_equity({t: r["equity"] for t, r in singles.items()} | {"Portafolio Risk Parity": res_rp["equity"]},
                      FIG / "13_assets_vs_portfolio.png", "Portafolio contra la estrategia de cada activo", split)
    labels_votes = {"v_adx": "ADX", "v_rsi": "RSI", "v_obv": "OBV", "v_skew": "Skew", "v_autocorr": "Autocorr",
                    "v_volz": "Vol-z"}
    plots.plot_correlation_by_regime({k: c.rename(index=labels_votes, columns=labels_votes) for k, c in vote_corr.items()},
                                     FIG / "17_vote_correlation.png",
                                     f"Correlación entre los votos de los indicadores (umbral |ρ| < {optimize.MAX_VOTE_CORR})",
                                     axis_label="Indicador")
    plots.plot_bars(conf[conf["period"] == "train"].groupby("rule")[["n_entries"]].mean()
                    .rename(columns={"n_entries": "Operaciones promedio por activo (train)"}),
                    FIG / "14_confirmation_trades.png", "Operaciones: regla 3 de 6 contra un solo indicador",
                    "Regla", "Operaciones")
    plots.plot_bars(conf.groupby(["rule", "period"])["calmar"].median().unstack(),
                    FIG / "15_confirmation_calmar.png", "Calmar mediano por activo: 3 de 6 contra un solo indicador",
                    "Regla", "Calmar")

    # 8. Metadatos -------------------------------------------------------------
    n_configs = int(records["n_trials"].sum())
    meta = {
        "seed": SEED, "tickers": data.TICKERS, "commission": COMMISSION, "initial_cash": 1_000_000,
        "split_date": split, "n_windows": len(windows), "oos_start": oos_index[0], "oos_end": oos_index[-1],
        "n_trials_per_study": optimize.N_TRIALS, "wf_minutes": wf_seconds / 60,
        "n_configs_evaluated": n_configs, "test_protocol": "theta promedio del walk-forward de train, congelado",
        "vote_correlation": vote_corr_summary,
        "n_studies_regime_fallback": int(records["fallback"].sum()),
        "n_studies_infeasible": int(records["infeasible"].sum()),
        "n_studies_total": int(len(records[(records["method"] == "none") | (records["n_trials"] > 0)])),
        "total_minutes": (time.time() - t_start) / 60,
        "regime_method": method,
        "avg_gross_exposure": {name: float((r["positions"].abs().sum(axis=1) / r["equity"]).mean())
                               for name, r in res.items()},
        "max_gross_exposure_close": {name: float((r["positions"].abs().sum(axis=1) / r["equity"]).max())
                                     for name, r in res.items()},
        "kruskal_pvalue_by_regime": by_regime, "rp_exit_reasons": exits,
        "regime_transitions_oos": transitions_oos, "market_impact": impact_summary,
        "calmar_negative_audit": calmar_audit_summary,
        "breakeven_commission": {m: float(np.interp(0, g["annual_return_total"].to_numpy()[::-1],
                                                    g["commission"].to_numpy()[::-1]))
                                 if (g["annual_return_total"] < 0).any() and (g["annual_return_total"] > 0).any()
                                 else None
                                 for m, g in costs.groupby("method")},
        "degradation": {p: {"median_is_calmar": float(g["is_calmar"].median()),
                            "median_oos_calmar": float(g["oos_calmar"].median()),
                            "mean_is_annual_return": float(g["is_annual_return"].mean()),
                            "mean_oos_annual_return": float(g["oos_annual_return"].mean())}
                        for p, g in deg.groupby("period")},
    }
    save_json(meta, "run_meta.json")
    print(f"[fin] {meta['total_minutes']:.1f} min, {n_configs} configuraciones evaluadas. Resultados en {RESULTS}")


if __name__ == "__main__":
    main()
