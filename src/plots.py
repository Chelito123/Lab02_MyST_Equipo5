"""Figuras del laboratorio. Cada función recibe datos ya calculados y guarda un PNG.

Todas las figuras llevan título, ejes etiquetados y leyenda.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import drawdown  # noqa: E402

REGIME_COLORS = {"trend": "#2a9d8f", "mean_reversion": "#457b9d", "crisis": "#e63946"}
REGIME_NAMES = {"trend": "Tendencia", "mean_reversion": "Reversión a la media", "crisis": "Crisis"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _split_line(ax, split_date):
    if split_date is not None:
        ax.axvline(pd.Timestamp(split_date), color="k", ls="--", lw=1, label="Inicio de prueba")


def plot_equity(curves: dict, path, title: str, split_date=None, ylabel: str = "Valor del portafolio (USD)"):
    """Curvas de valor del portafolio (dict nombre -> Series)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, lw=1.3)
    _split_line(ax, split_date)
    ax.set(title=title, xlabel="Fecha", ylabel=ylabel)
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_drawdown(curves: dict, path, title: str, split_date=None):
    """Curvas de drawdown en porcentaje."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, eq in curves.items():
        dd = drawdown(eq) * 100
        ax.plot(dd.index, dd.values, label=name, lw=1.1)
    _split_line(ax, split_date)
    ax.set(title=title, xlabel="Fecha", ylabel="Drawdown (%)")
    ax.legend(loc="lower left")
    _save(fig, path)


def plot_return_table(table: pd.DataFrame, path, title: str):
    """Mapa de calor de una tabla de retornos (año x periodo) con valores en %."""
    data = table.to_numpy(dtype=float) * 100
    lim = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 2, 0.45 * table.shape[0] + 1.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set(title=title, xlabel="Periodo", ylabel="Año")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Retorno (%)")
    _save(fig, path)


def plot_sensitivity(sens: pd.DataFrame, path, title: str):
    """Calmar ante variaciones de ±10% y ±20% de cada parámetro, una línea por activo."""
    params = list(dict.fromkeys(sens["param"]))
    ncols = 4
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)
    for ax, name in zip(axes.flat, params):
        sub = sens[sens["param"] == name]
        for ticker, g in sub.groupby("ticker"):
            ax.plot(g["delta"] * 100, g["calmar"], marker="o", ms=3, lw=1, label=ticker)
        ax.set(title=name, xlabel="Variación del parámetro (%)", ylabel="Calmar")
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=3, title="Activo")
    fig.suptitle(title)
    _save(fig, path)


def plot_cost_curve(curve: pd.DataFrame, path, title: str, base_cost: float):
    """Retorno neto anualizado contra comisión por operación (una columna por estrategia)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in curve.columns:
        ax.plot(curve.index * 100, curve[col] * 100, marker="o", label=col)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(base_cost * 100, color="gray", ls="--", label=f"Comisión del laboratorio ({base_cost:.3%})")
    ax.set(title=title, xlabel="Comisión por operación (%)", ylabel="Retorno neto anualizado (%)")
    ax.legend()
    _save(fig, path)


def _shade_regimes(ax, labels: pd.Series):
    lab = labels.dropna()
    block = (lab != lab.shift()).cumsum()
    seen = set()
    for _, seg in lab.groupby(block):
        regime = seg.iloc[0]
        ax.axvspan(seg.index[0], seg.index[-1], color=REGIME_COLORS[regime], alpha=0.18, lw=0,
                   label=REGIME_NAMES[regime] if regime not in seen else None)
        seen.add(regime)


def plot_regime_timeline(index: pd.Series, labels: pd.Series, path, title: str, split_date=None):
    """Índice de mercado con los regímenes sombreados."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, labels)
    ax.plot(index.index, index.values, color="k", lw=1, label="Índice equiponderado")
    _split_line(ax, split_date)
    ax.set(title=title, xlabel="Fecha", ylabel="Índice (base 100)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_equity_regimes(curves: dict, labels: pd.Series, path, title: str, split_date=None):
    """Valor del portafolio con regímenes superpuestos."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_regimes(ax, labels.reindex(next(iter(curves.values())).index))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, lw=1.3, label=name)
    _split_line(ax, split_date)
    ax.set(title=title, xlabel="Fecha", ylabel="Valor del portafolio (USD)")
    ax.legend(loc="upper left")
    _save(fig, path)


COMPARISON_STYLE = {
    "Régimen · Reglas": dict(color="#264653"),
    "Régimen · K-means": dict(color="#f4a261"),
    "Régimen · HMM (filtrada)": dict(color="#2a9d8f"),
    "Sin régimen": dict(color="black", linestyle="--"),
    "Buy & hold equiponderado": dict(color="#8d99ae", linestyle=":"),
}


def plot_strategy_comparison(curves: dict, path, title: str, initial: float = 1_000_000):
    """Curvas de un mismo periodo re-escaladas al capital inicial, con el valor final en la leyenda."""
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for name, eq in curves.items():
        eq = eq / eq.iloc[0] * initial
        ax.plot(eq.index, eq.values, lw=1.4, label=f"{name}  (final ${eq.iloc[-1]:,.0f})",
                **COMPARISON_STYLE.get(name, {}))
    ax.axhline(initial, color="gray", lw=0.8, ls=":")
    ax.set(title=title, xlabel="Fecha", ylabel="Valor del portafolio (USD)")
    ax.legend(loc="upper left")
    _save(fig, path)


def plot_regime_methods(index: pd.Series, labels: dict, path, title: str, split_date=None):
    """Un panel por clasificador con el índice y los regímenes sombreados."""
    fig, axes = plt.subplots(len(labels), 1, figsize=(13, 3.2 * len(labels)), sharex=True)
    for ax, (name, lab) in zip(np.atleast_1d(axes), labels.items()):
        _shade_regimes(ax, lab)
        ax.plot(index.index, index.values, color="k", lw=0.9, label="Índice equiponderado")
        _split_line(ax, split_date)
        ax.set(title=name, ylabel="Índice (base 100)")
        ax.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Fecha")
    fig.suptitle(title)
    _save(fig, path)


def plot_regime_distributions(F: pd.DataFrame, labels: pd.Series, path, title: str):
    """Histogramas de cada variable de régimen por estado."""
    lab = labels.reindex(F.index)
    fig, axes = plt.subplots(1, F.shape[1], figsize=(4.5 * F.shape[1], 4))
    for ax, col in zip(np.atleast_1d(axes), F.columns):
        for regime, color in REGIME_COLORS.items():
            x = F.loc[lab == regime, col]
            if len(x):
                ax.hist(x, bins=40, alpha=0.5, color=color, density=True, label=REGIME_NAMES[regime])
        ax.set(title=col, xlabel=col, ylabel="Densidad")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, path)


def plot_risk_contributions(shares: pd.DataFrame, path, title: str):
    """Barras de contribución promedio al riesgo por activo (columnas = método)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(shares.index))
    width = 0.8 / shares.shape[1]
    for k, col in enumerate(shares.columns):
        ax.bar(x + k * width - 0.4 + width / 2, shares[col] * 100, width, label=col)
    ax.axhline(100 / len(shares.index), color="k", ls="--", lw=1, label="Contribución igual (1/n)")
    ax.set_xticks(x, shares.index)
    ax.set(title=title, xlabel="Activo", ylabel="Contribución al riesgo (%)")
    ax.legend()
    _save(fig, path)


def plot_heatmap(data: pd.DataFrame, path, title: str, xlabel: str, ylabel: str, cbar: str,
                 cmap: str = "RdBu_r", vlim: float = 1.0):
    """Mapa de calor genérico (filas = data.index, columnas = data.columns)."""
    fig, ax = plt.subplots(figsize=(13, 0.5 * data.shape[0] + 2))
    im = ax.imshow(data.to_numpy(float), cmap=cmap, vmin=-vlim, vmax=vlim, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(data.shape[0]), data.index)
    step = max(1, data.shape[1] // 16)
    ax.set_xticks(range(0, data.shape[1], step), [str(c) for c in data.columns[::step]], rotation=45, ha="right")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label=cbar)
    _save(fig, path)


def plot_correlation_by_regime(corrs: dict, path, title: str, axis_label: str = "Activo"):
    """Matrices de correlación (dict régimen -> DataFrame) lado a lado."""
    fig, axes = plt.subplots(1, len(corrs), figsize=(5 * len(corrs), 4.5))
    for ax, (regime, c) in zip(np.atleast_1d(axes), corrs.items()):
        im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(c.shape[1]), c.columns, rotation=45)
        ax.set_yticks(range(c.shape[0]), c.index)
        ax.set(title=REGIME_NAMES.get(regime, regime), xlabel=axis_label, ylabel=axis_label)
        ax.grid(False)
    fig.subplots_adjust(wspace=0.35)
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label="Correlación")
    fig.suptitle(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_lines(df: pd.DataFrame, path, title: str, xlabel: str, ylabel: str, scale: float = 100.0,
               marker: str = "o", hline: float = None):
    """Gráfica de líneas genérica (índice = eje x, una línea por columna)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in df.columns:
        ax.plot(df.index, df[col] * scale, marker=marker, label=col)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    _save(fig, path)


def plot_bars(df: pd.DataFrame, path, title: str, xlabel: str, ylabel: str):
    """Barras agrupadas (índice = categorías, columnas = series)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot.bar(ax=ax, rot=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    _save(fig, path)
