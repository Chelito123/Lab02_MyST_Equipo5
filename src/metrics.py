"""Métricas de desempeño sobre curvas de equity diarias.

Anualización con 252 días hábiles. Retorno anualizado geométrico (CAGR).
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def annualized_return(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return np.nan
    growth = equity.iloc[-1] / equity.iloc[0]
    return growth ** (periods_per_year / n) - 1 if growth > 0 else -1.0


def sharpe_ratio(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Sharpe con tasa libre de riesgo 0: media / desviación de retornos diarios × √252."""
    r = daily_returns(equity)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def sortino_ratio(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Sortino: media / downside deviation (raíz del promedio de min(r, 0)²) × √252."""
    r = daily_returns(equity)
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown relativo al máximo previo (≤ 0)."""
    return equity / equity.cummax() - 1


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown como número positivo (0.25 = -25%)."""
    return float(-drawdown(equity).min())


def calmar_ratio(equity: pd.Series, mdd_floor: float = 0.0,
                 periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Calmar = retorno anualizado / MDD. `mdd_floor` evita dividir entre ~0 en ventanas cortas."""
    mdd = max(max_drawdown(equity), mdd_floor)
    return float(annualized_return(equity, periods_per_year) / mdd) if mdd > 0 else np.nan


def win_rate(trades: pd.DataFrame) -> float:
    """Proporción de operaciones cerradas con P&L neto (después de comisiones) positivo."""
    return float((trades["pnl"] > 0).mean()) if len(trades) else np.nan


def turnover(fills: pd.DataFrame, equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Rotación anualizada: nocional operado total / equity promedio, por año."""
    if len(fills) == 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    return float(fills["notional"].sum() / equity.mean() / years)


def summary(equity: pd.Series, trades: pd.DataFrame = None, fills: pd.DataFrame = None) -> dict:
    """Métricas obligatorias del laboratorio más retorno total, rotación y costos."""
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annual_return": annualized_return(equity),
        "annual_vol": float(daily_returns(equity).std() * np.sqrt(PERIODS_PER_YEAR)),
        "sharpe": sharpe_ratio(equity),
        "sortino": sortino_ratio(equity),
        "calmar": calmar_ratio(equity),
        "max_drawdown": max_drawdown(equity),
    }
    if trades is not None:
        out["win_rate"] = win_rate(trades)
        out["n_trades"] = int(len(trades))
    if fills is not None:
        out["turnover"] = turnover(fills, equity)
        out["total_costs"] = float(fills["commission"].sum())
    return out


def period_returns(equity: pd.Series) -> dict:
    """Retornos compuestos mensuales, trimestrales y anuales."""
    r = daily_returns(equity)
    agg = lambda freq: (1 + r).resample(freq).prod() - 1  # noqa: E731
    return {"monthly": agg("ME"), "quarterly": agg("QE"), "annual": agg("YE")}


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Tabla año x mes de retornos mensuales, con el retorno anual en la última columna."""
    pr = period_returns(equity)
    m = pr["monthly"]
    table = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    table = table.pivot(index="year", columns="month", values="ret")
    table.columns = [MONTHS[k - 1] for k in table.columns]
    table["Año"] = pr["annual"].set_axis(pr["annual"].index.year)
    return table


def quarterly_table(equity: pd.Series) -> pd.DataFrame:
    """Tabla año x trimestre de retornos trimestrales."""
    q = period_returns(equity)["quarterly"]
    table = pd.DataFrame({"year": q.index.year, "q": "T" + q.index.quarter.astype(str), "ret": q.values})
    return table.pivot(index="year", columns="q", values="ret")


def split_summary(result: dict, split_date) -> pd.DataFrame:
    """Métricas por separado antes (train) y después (test) de `split_date` para un resultado de backtest."""
    eq, trades, fills = result["equity"], result["trades"], result["fills"]
    split = pd.Timestamp(split_date)
    before = eq.index < split
    parts = {
        "train": (eq[before], trades["exit_date"] < split, fills["date"] < split),
        # el tramo de prueba arranca desde el último cierre de train
        "test": (eq.iloc[max(before.sum() - 1, 0):], trades["exit_date"] >= split, fills["date"] >= split),
    }
    rows = {name: summary(e, trades[tm], fills[fm]) for name, (e, tm, fm) in parts.items()}
    rows["total"] = summary(eq, trades, fills)
    return pd.DataFrame(rows).T


def metrics_by_regime(equity: pd.Series, labels: pd.Series) -> tuple:
    """Métricas de los retornos diarios agrupados por el régimen vigente al cierre anterior.

    Regresa (tabla, p-value de Kruskal-Wallis de igualdad de distribuciones entre regímenes).
    """
    from scipy.stats import kruskal

    r = daily_returns(equity)
    lab = labels.shift(1).reindex(r.index)
    rows, groups = {}, []
    for regime, g in r.groupby(lab):
        groups.append(g.to_numpy())
        sd = g.std()
        downside = np.sqrt((np.minimum(g, 0) ** 2).mean())
        rows[regime] = {
            "days": len(g),
            "mean_daily": g.mean(),
            "annual_vol": sd * np.sqrt(PERIODS_PER_YEAR),
            "sharpe": g.mean() / sd * np.sqrt(PERIODS_PER_YEAR) if sd > 0 else np.nan,
            "sortino": g.mean() / downside * np.sqrt(PERIODS_PER_YEAR) if downside > 0 else np.nan,
            "cum_return": (1 + g).prod() - 1,
            "hit_rate": (g > 0).mean(),
        }
    pvalue = kruskal(*groups).pvalue if len(groups) > 1 else np.nan
    return pd.DataFrame(rows).T, float(pvalue)


def estimate_market_impact(fills: pd.DataFrame, prices: dict, y: float = 1.0) -> pd.DataFrame:
    """Impacto estimado con la ley de raíz cuadrada: Y · σ_diaria · sqrt(Q / V) · nocional, por ejecución.

    σ_diaria: desviación de 20 días de los retornos del activo; V: volumen del día de la ejecución.
    El backtest NO cobra este costo; sirve para dimensionar la limitación.
    """
    rows = []
    for t, f in fills.groupby("ticker"):
        df = prices[t]
        sigma = df["close"].pct_change().rolling(20).std().reindex(f["date"]).to_numpy()
        participation = f["qty"].abs().to_numpy() / df["volume"].reindex(f["date"]).to_numpy()
        rows.append(pd.DataFrame({"date": f["date"].to_numpy(), "ticker": t, "participation": participation,
                                  "impact_cost": y * sigma * np.sqrt(participation) * f["notional"].to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def calmar_audit_summary(records: pd.DataFrame, audit: pd.DataFrame) -> dict:
    """Resumen de `optimize.audit_negative_calmar`: alcance y efecto del defecto del Calmar con R < 0."""
    studies = records[~records["fallback"]]
    affected = audit[~audit["picked_least_loss"]]
    return {
        "n_studies": int(len(studies)),
        "n_best_calmar_negative": int(len(audit)),
        "share_best_calmar_negative": float(len(audit) / len(studies)),
        "all_reproduce": bool(audit["reproduces"].all()),
        "mean_mdd_percentile_of_best_among_losers": float(audit["mdd_percentile_of_best"].mean()),
        "mean_corr_calmar_mdd_among_losers": float(audit["corr_calmar_mdd_losers"].mean()),
        "n_picked_least_loss": int(audit["picked_least_loss"].sum()),
        "n_picked_bigger_loss": int(len(affected)),
        "share_of_all_studies_affected": float(len(affected) / len(studies)),
        "median_extra_loss_when_affected": float((affected["least_loss_R"] - affected["best_R"]).median()) if len(affected) else 0.0,
        "median_extra_mdd_when_affected": float((affected["best_mdd"] - affected["least_loss_mdd"]).median()) if len(affected) else 0.0,
    }
