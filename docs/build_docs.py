"""Genera docs/reporte.pdf y docs/presentacion.pdf a partir de results/ (correr después de main.py).

    python docs/build_docs.py

Las tablas y casi todas las cifras del texto se leen de results/. Si se vuelve a correr main.py,
revisar las frases cualitativas (p. ej. qué clasificador gana en prueba).
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
DOCS = ROOT / "docs"

EQUIPO = "Equipo 5"
TEAM = [("Adrián Marcelo Ballesteros Herrera", "AAPL, MSFT"),
        ("Erik del Castillo Román", "JPM, XOM"),
        ("Jesús Emmanuel Flores Cortés", "JNJ, WMT")]
REPO = "github.com/Chelito123/Lab02_MyST_Equipo5"
N_TESTS = 43
FECHA = "Octubre 2026"

# DejaVu viene con matplotlib (portátil) y trae todos los símbolos que usa el texto (σ, θ, ∈, ∓, ≥, ...).
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")

INK = colors.HexColor("#1d2b36")
ACCENT = colors.HexColor("#2a9d8f")
MUTED = colors.HexColor("#5c6b77")
LIGHT = colors.HexColor("#eef3f5")
WARN = colors.HexColor("#e76f51")

meta = json.loads((RES / "run_meta.json").read_text(encoding="utf-8"))
val = json.loads((RES / "regime_validation.json").read_text(encoding="utf-8"))
perf = pd.read_csv(RES / "performance.csv", index_col=[0, 1])
comp = pd.read_csv(RES / "regime_method_comparison.csv", index_col=[0, 1])
audit = meta["calmar_negative_audit"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def num(x, d=2):
    return f"{x:.{d}f}"


# ----------------------------------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------------------------------
S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=22, leading=27, textColor=INK, spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=12, leading=16, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=12, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=9.6, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=9.6, leading=13.2, leftIndent=12, bulletIndent=2, spaceAfter=2),
    "formula": ParagraphStyle("formula", fontName="Sans", fontSize=10.5, leading=15, alignment=TA_CENTER,
                              backColor=LIGHT, borderPadding=6, spaceBefore=4, spaceAfter=8),
    "caption": ParagraphStyle("caption", fontName="Sans-Italic", fontSize=8.2, leading=10.5, textColor=MUTED,
                              alignment=TA_CENTER, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10, textColor=colors.white),
    "note": ParagraphStyle("note", fontName="Sans", fontSize=9.2, leading=12.8, backColor=colors.HexColor("#fdf0ec"),
                           borderPadding=6, spaceBefore=4, spaceAfter=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths=None, header=True):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3d9")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), INK))
    t.setStyle(TableStyle(style))
    return t


def figure(name, caption, width=16.5 * cm):
    path = FIG / name
    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, "caption")])


def metrics_rows(names):
    rows = [["Estrategia", "Periodo", "Ret. anual", "Sharpe", "Sortino", "Calmar", "MDD", "Win rate", "Núm. oper."]]
    for n in names:
        for period in ("train", "test"):
            r = perf.loc[(n, period)]
            wr = "—" if pd.isna(r["win_rate"]) else pct(r["win_rate"])
            rows.append([n, period, pct(r["annual_return"]), num(r["sharpe"]), num(r["sortino"]), num(r["calmar"]),
                         pct(r["max_drawdown"]), wr, int(r["n_trades"])])
    return rows


def sensitivity_summary():
    """Cambio absoluto del Calmar de train ante ±20% de cada parámetro (mediana y peor caso entre activos)."""
    s = pd.read_csv(RES / "sensitivity.csv")
    base = s[s["delta"] == 0].set_index(["ticker", "param"])["calmar"]
    s["d_calmar"] = s["calmar"] - s.set_index(["ticker", "param"]).index.map(base)
    x = s[s["delta"].abs() == 0.2]
    by_param = x.groupby("param")["d_calmar"].agg(med=lambda v: v.abs().median(), worst=lambda v: v.abs().max())
    flips = (np.sign(x["calmar"]) != np.sign(x.set_index(["ticker", "param"]).index.map(base))).mean()
    return by_param.sort_values("med", ascending=False), float(flips), base.groupby("ticker").first()


def build_report():
    rp_tot, ew_tot, nv_tot = (perf.loc[(n, "total")] for n in ("Risk Parity", "Pesos iguales", "Risk Parity naive (1/σ)"))
    rp_tr, rp_te = perf.loc[("Risk Parity", "train")], perf.loc[("Risk Parity", "test")]
    bh_tot = perf.loc[("Buy & hold equiponderado", "total")]
    hmm = val["methods"]["hmm"]
    deg = meta["degradation"]["train"]
    wfe = deg["mean_oos_annual_return"] / deg["mean_is_annual_return"]
    imp = meta["market_impact"]
    be = meta["breakeven_commission"]
    vc = meta["vote_correlation"]
    costs = pd.read_csv(RES / "cost_sweep.csv")
    zero_cost = costs[(costs["commission"] == 0) & (costs["method"] == "rp")]["annual_return_total"].iloc[0]
    rebal = pd.read_csv(RES / "rebalance_sweep.csv")
    rp_reb = rebal[rebal["method"] == "rp"].set_index("every")
    sens, flips, sens_base = sensitivity_summary()
    frozen = pd.read_csv(RES / "frozen_theta_performance.csv")
    frozen_test = frozen[frozen["period"] == "test"].set_index("ticker")
    n_frozen_losers = int((frozen_test["annual_return"] < 0).sum())
    s = []

    # Portada
    s += [Spacer(1, 3 * cm), P("Laboratorio 02 — Estrategias de trading con análisis técnico", "title"),
          P(f"Reporte ejecutivo · Nivel C · {EQUIPO} · Microestructuras y Sistemas de Trading, ITESO", "subtitle"),
          Spacer(1, 0.6 * cm),
          table([["Integrante", "Activos a cargo"]] + [list(t) for t in TEAM], widths=[8 * cm, 6 * cm]),
          Spacer(1, 0.6 * cm),
          P(f"{FECHA}. Repositorio: {REPO}. Todos los resultados se reproducen con <b>python main.py</b> "
            "(semilla 42)."),
          PageBreak()]

    # 1. Resumen
    s += [P("1. Resumen ejecutivo", "h1"),
          P("Construimos un sistema de trading sistemático sobre seis acciones líquidas de EE. UU. Cada activo opera con "
            "seis indicadores técnicos de familias distintas y abre posición cuando al menos tres coinciden en "
            "dirección (regla 3 de 6, autorizada por el profesor con mínimo de tres indicadores). Un motor event-driven "
            "con comisión de 0.125% simula la ejecución sin apalancamiento. Los parámetros se optimizan con Optuna en un "
            "walk-forward de 6 meses de entrenamiento y 1 mes de prueba dentro del periodo de entrenamiento "
            "(2019–2023), por activo y por régimen de mercado; para el periodo de prueba (2024 → 2026-09) se promedian "
            "los θ de cada activo y se congelan. Las señales se agregan en un portafolio de Risk Parity con $1,000,000."),
          P(f"<b>Resultado.</b> El portafolio Risk Parity rinde {pct(rp_tr['annual_return'])} anual en el walk-forward "
            f"de entrenamiento y {pct(rp_te['annual_return'])} anual en prueba con el θ congelado (Sharpe "
            f"{num(rp_te['sharpe'])}, Calmar {num(rp_te['calmar'])}, MDD {pct(rp_te['max_drawdown'])}); en todo el "
            f"periodo, {pct(rp_tot['annual_return'])} anual contra {pct(bh_tot['annual_return'])} del buy &amp; hold "
            f"equiponderado. La regla 3 de 6 bajó la actividad: la señal está activa {pct(vc['signal_frequency'])} de los "
            f"días y la comisión de equilibrio es {be['rp'] * 100:.2f}% por operación, arriba del 0.125% especificado."),
          P("<b>Cautela.</b> La ganancia de prueba depende de un solo clasificador de régimen (el HMM; con K-means, "
            "reglas o sin régimen el portafolio no gana en prueba), la ventaja in-sample casi no sobrevive en el "
            f"walk-forward (eficiencia {wfe:.2f}) y esta versión de la estrategia se definió después de ver resultados "
            "de prueba de la versión anterior (sección 10). No la consideramos una ventaja demostrada.", "note"),
          table([["Indicador (todo el periodo 2019-01 → 2026-09)", "Risk Parity", "RP naive (1/σ)", "Pesos iguales", "Buy & hold"],
                 ["Retorno anual", pct(rp_tot["annual_return"]), pct(nv_tot["annual_return"]), pct(ew_tot["annual_return"]), pct(bh_tot["annual_return"])],
                 ["Sharpe", num(rp_tot["sharpe"]), num(nv_tot["sharpe"]), num(ew_tot["sharpe"]), num(bh_tot["sharpe"])],
                 ["Calmar", num(rp_tot["calmar"]), num(nv_tot["calmar"]), num(ew_tot["calmar"]), num(bh_tot["calmar"])],
                 ["Máximo drawdown", pct(rp_tot["max_drawdown"]), pct(nv_tot["max_drawdown"]), pct(ew_tot["max_drawdown"]), pct(bh_tot["max_drawdown"])],
                 ["Comisiones pagadas", f"${rp_tot['total_costs']:,.0f}", f"${nv_tot['total_costs']:,.0f}", f"${ew_tot['total_costs']:,.0f}", f"${bh_tot['total_costs']:,.0f}"]],
                widths=[5.6 * cm, 2.7 * cm, 2.8 * cm, 2.7 * cm, 2.7 * cm]),
          Spacer(1, 6),
          P("<b>Advertencia de interpretación.</b> El backtest asume ejecución completa al precio modelado (open de "
            "t+1 o el nivel de stop) y no incorpora impacto de mercado, spread, costo de préstamo en cortos ni fallas "
            f"de ejecución. Con la ley de raíz cuadrada estimamos el impacto omitido en ≈{pct(imp['impact_cost_annual_pct_equity'], 2)} "
            "del capital por año (sección 9).", "note")]

    # 2. Datos
    s += [P("2. Datos y universo", "h1"),
          P("Universo de 2n = 6 acciones (n = 3 integrantes), dos por integrante y de sectores distintos para tener "
            "correlaciones variadas: AAPL y MSFT (tecnología), JPM y XOM (banca y energía), JNJ y WMT (salud y consumo). "
            "Precios diarios OHLCV ajustados por splits y dividendos de Yahoo Finance del 2018-01-02 al 2026-09-30 "
            "(2,198 días hábiles, 8.75 años), congelados en <i>data/</i>."),
          P("La auditoría (<i>results/data_audit.csv</i>) no encontró faltantes, fechas duplicadas, precios no positivos "
            "ni barras OHLC inconsistentes; el mayor hueco es de 4 días naturales y el mayor salto diario es 18% (JPM, "
            "marzo de 2020). División: <b>entrenamiento</b> 2018–2023 (el walk-forward produce resultados fuera de muestra "
            "desde 2019-01, porque la primera ventana de entrenamiento son los 6 meses previos) y <b>prueba</b> "
            "2024-01-02 → 2026-09-30."),
          P("<b>Sesgo de supervivencia.</b> Los activos se eligieron en 2026 entre empresas que siguen siendo grandes y "
            "líquidas; un inversionista en 2018 no podía saberlo.", "note")]

    # 3. Estrategia
    vf = vc["vote_frequency"]
    s += [P("3. Estrategia por activo", "h1"),
          P("Seis indicadores: los cinco de las actividades anteriores más el RSI. Cada uno emite un voto "
            "v<sub>k,t</sub> ∈ {−1, 0, +1} con información hasta el cierre de t:"),
          table([["Familia", "Indicador", "Voto", "% de días que vota"],
                 ["Tendencia", "ADX con ±DI (Wilder)", "sign(+DI − −DI) si ADX > umbral", pct(vf["v_adx"], 0)],
                 ["Momento (reversión)", "RSI con zona de tolerancia", "Ver abajo: −1 en sobrecompra sostenida, +1 en sobreventa sostenida", pct(vf["v_rsi"], 0)],
                 ["Volumen", "Desbalance de OBV: Σ sign(ΔC)·V / Σ V", "±1 si pasa de ±umbral", pct(vf["v_obv"], 0)],
                 ["Distribución", "Skew de los retornos", "±1 si pasa de ±0.5 (≈ 1 error estándar con n = 20)", pct(vf["v_skew"], 0)],
                 ["Régimen", "Autocorrelación de rezago 1", "Si |ρ1| > 1.96/√n y hay racha de 2 días: sigue la racha (ρ1 > 0) o va en contra (ρ1 < 0)", pct(vf["v_autocorr"], 0)],
                 ["Volumen-momento", "z-score del volumen", "sign(retorno) si z > 1", pct(vf["v_volz"], 0)]],
                widths=[2.7 * cm, 4.4 * cm, 7.0 * cm, 2.5 * cm]),
          P("Frecuencia de voto en el periodo fuera de muestra con los θ realmente usados.", "caption"),
          P("<b>RSI con zona de tolerancia.</b> El RSI entra a la zona de sobrecompra al pasar del nivel L y se considera "
            "dentro mientras no baje de L − b, de modo que un retroceso pequeño (por ejemplo de 80 a 79) no lo saca. Si "
            "permanece k días seguidos en la zona vota −1 (reversión) y sigue votando mientras siga dentro. Simétrico en "
            "sobreventa con el nivel 100 − L y tolerancia hacia arriba: +1. L ∈ [65, 85], b ∈ [0, 10] y k ∈ [1, 10] "
            "los elige Optuna."),
          P("<b>Autocorrelación.</b> La versión de las actividades anteriores exigía además que ρ2 y ρ3 tuvieran el signo de "
            "ρ1; en acciones diarias votaba en 0.3% de los días, así que se quitó esa condición y su ventana (20–90 días) se "
            "optimiza."),
          P("Con N<super>+</super><sub>t</sub> y N<super>−</super><sub>t</sub> el número de votos a favor y en contra, la "
            "<b>regla de confirmación 3 de 6</b> y la fuerza de la señal son:"),
          P("s<sub>t</sub> = 1{N<super>+</super> ≥ 3 y N<super>+</super> &gt; N<super>−</super>} − "
            "1{N<super>−</super> ≥ 3 y N<super>−</super> &gt; N<super>+</super>}"
            "&nbsp;&nbsp;&nbsp;&nbsp; f<sub>t</sub> = (N<super>+</super> − N<super>−</super>) / 6", "formula"),
          P("Un empate 3 contra 3 no abre posición. La señal de t se ejecuta al open de t+1. Una posición se mantiene hasta: "
            "stop-loss = P ∓ k<sub>SL</sub>·ATR, take-profit = P ± k<sub>TP</sub>·ATR, señal opuesta u holding máximo. Si SL "
            "y TP caen en la misma barra se ejecuta primero el stop-loss; si el open abre más allá de un nivel, la salida es "
            "al open. Tamaño por activo: fracción f ∈ [0.2, 1] del capital."),
          P("<b>Validación de que los indicadores no son redundantes.</b> El profesor pidió verificar con una matriz de "
            "correlación que los indicadores no sean muy similares. Se impuso como restricción de la optimización: un θ "
            f"cuyos votos tengan |ρ| ≥ {vc['threshold']} en su ventana de entrenamiento es infactible (0.4 dejaba sin θ "
            "factible 7.6% de los estudios, así que se usó 0.5). Fuera de muestra, la correlación máxima entre votos es "
            f"{vc['max_abs_train']:.2f} en entrenamiento y {vc['max_abs_test']:.2f} en prueba; el par más correlacionado es "
            "ADX–OBV (ambos siguen la tendencia). Los valores crudos de ADX, RSI y OBV sí están muy correlacionados "
            "(0.63–0.83) porque los tres miden la dirección del precio; los votos lo están mucho menos porque cada uno usa "
            "umbrales distintos y el RSI vota en reversión."),
          figure("17_vote_correlation.png", "Figura 1. Correlación entre los votos de los seis indicadores, fuera de muestra.", width=15 * cm)]

    # 4. Motor
    s += [P("4. Motor de backtesting", "h1"),
          P("Motor event-driven barra por barra (<i>src/backtest.py</i>) con estado explícito: efectivo, acciones por activo "
            "(negativas en cortos) y valor del portafolio = efectivo + Σ acciones·precio. El mismo motor simula un activo o "
            "el portafolio de seis."),
          *bullets([
              "<b>Costos:</b> comisión de 0.125% sobre el nocional de cada ejecución: apertura, cierre y ajustes de rebalanceo.",
              "<b>Sin apalancamiento:</b> el nocional bruto ejecutado nunca excede equity / (1 + comisión). Exposición bruta "
              f"al cierre: promedio {pct(meta['avg_gross_exposure']['rp'])}, máximo {pct(meta['max_gross_exposure_close']['rp'])}.",
              f"<b>Pruebas (pytest, {N_TESTS} casos):</b> causalidad de la señal recalculando sobre df.iloc[:t+1]; regla 3 de 6 "
              "(con 2 votos no se abre posición y con 3 sí; empate 3–3 no abre); zona del RSI; contabilidad (valor final = "
              "efectivo + posiciones y costos = comisión × nocional); SL primero; no apalancamiento; etiqueta de régimen "
              "invariante a datos futuros; contribuciones al riesgo iguales en Risk Parity y misma solución que un método "
              "independiente."])]

    # 5. Optimización
    s += [P("5. Optimización, walk-forward y θ congelado", "h1"),
          table([["Elemento", "Especificación"],
                 ["Método", "Optuna TPE (bayesiano), 100 pruebas por estudio"],
                 ["Walk-forward", f"6 meses de entrenamiento, 1 mes de prueba, paso mensual, solo dentro del entrenamiento: {meta['n_windows']} ventanas (2019-01 → 2023-12)"],
                 ["Estudios por activo y ventana", "1 θ global (modelo sin régimen) + 1 θ por régimen para cada uno de los 3 clasificadores"],
                 ["Espacio θ (15 dims.)", "ventanas de ADX, RSI, OBV, skew/vol-z, autocorrelación y ATR; umbral de ADX; nivel L, tolerancia b y días k del RSI; umbral de OBV; k_SL, k_TP; holding máximo; fracción de capital"],
                 ["Restricciones", f"mínimo 4 operaciones por ventana (2 por régimen); |ρ| entre votos < {vc['threshold']}"],
                 ["Objetivo", "Calmar = retorno anualizado / MDD, con piso de 1% en el MDD"],
                 ["Prueba", "para cada activo y régimen se promedian los θ del walk-forward (enteros redondeados) y se congelan para todo 2024-01 → 2026-09"],
                 ["Cómputo", f"{meta['n_configs_evaluated']:,} configuraciones evaluadas; ≈{meta['wf_minutes']:.0f} min de cómputo en 8 procesos paralelos (joblib)"]],
                widths=[4.2 * cm, 12.4 * cm]),
          Spacer(1, 6),
          P(f"Un estudio por régimen usa el θ global de su ventana si el régimen ocupa menos de 21 días o si ninguna "
            f"configuración es factible: {meta['n_studies_regime_fallback']:,} registros de régimen usaron el θ global; "
            f"{meta['n_studies_infeasible']} de {meta['n_studies_total']:,} estudios optimizados no encontraron un θ factible."),
          P("<b>Defecto del Calmar con retorno negativo.</b> Con R &lt; 0, Calmar = R/MDD se acerca a 0 cuando el MDD crece: "
            "maximizarlo puede preferir perder más con más drawdown. Repetimos con su semilla original los "
            f"{audit['n_best_calmar_negative']} estudios (de {audit['n_studies']:,}) donde todas las configuraciones perdían: "
            f"en {audit['n_picked_bigger_loss']} ({pct(audit['share_of_all_studies_affected'], 1)} del total) el optimizador "
            f"eligió una que perdía más (mediana +{audit['median_extra_loss_when_affected'] * 100:.1f} pts de pérdida anual y "
            f"+{audit['median_extra_mdd_when_affected'] * 100:.1f} pts de MDD). Se conserva el Calmar porque el laboratorio lo "
            "exige (detalle en <i>docs/hallazgo_calmar.md</i>).", "note")]

    # 6. Régimen
    rows = [["Clasificador", "Silhouette (train)", "Duración media (días)", "Transiciones / año", "% tendencia / reversión / crisis (test)"]]
    for m, name in (("rules", "Reglas"), ("kmeans", "K-means"), ("hmm", "HMM (filtrada)")):
        v = val["methods"][m]
        sh = v["causal_operated"]["share_test"]
        rows.append([name, num(v["train_fit"]["silhouette"], 3), num(v["causal_operated"]["avg_duration_days"], 1),
                     num(v["causal_operated"]["transitions_per_year"], 1),
                     " / ".join(pct(sh.get(r, 0.0), 0) for r in ("trend", "mean_reversion", "crisis"))])
    crow = [["Estrategia (Risk Parity)", "Calmar train", "Ret. anual train", "Ret. anual test", "Calmar test", "MDD total"]]
    for n in ["Régimen · Reglas", "Régimen · K-means", "Régimen · HMM (filtrada)", "Sin régimen", "Buy & hold equiponderado"]:
        crow.append([n, num(comp.loc[(n, "train"), "calmar"], 3), pct(comp.loc[(n, "train"), "annual_return"]),
                     pct(comp.loc[(n, "test"), "annual_return"]), num(comp.loc[(n, "test"), "calmar"], 3),
                     pct(comp.loc[(n, "total"), "max_drawdown"])])
    tm = hmm["causal_operated"]["transition_matrix"]
    s += [PageBreak(), P("6. Análisis de régimen", "h1"),
          P("<b>Variables</b> sobre el índice de la canasta equiponderada en una ventana móvil de 63 días hábiles (3 meses): "
            "<b>volatilidad</b> anualizada, <b>razón de eficiencia</b> de Kaufman |P<sub>t</sub> − P<sub>t−63</sub>| / Σ|ΔP| "
            "(fuerza de tendencia) y <b>autocorrelación de rezago 1</b> (reversión). El drawdown de la ventana se descartó "
            "por redundante con la volatilidad (bajaba el silhouette de K-means de 0.375 a 0.319)."),
          P("<b>Clasificadores.</b> (1) <b>Reglas</b>: crisis si vol &gt; percentil 90; tendencia si eficiencia &gt; mediana; "
            "reversión en otro caso. (2) <b>K-means</b> de 3 clusters. (3) <b>HMM gaussiano</b> (Baum-Welch en numpy) operado "
            "con la etiqueta <b>filtrada</b> ŝ<sub>t</sub> = argmax<sub>j</sub> P(S<sub>t</sub> = j | x<sub>1</sub>, …, x<sub>t</sub>) "
            "del algoritmo forward. Viterbi re-etiqueta el pasado con información futura, así que solo se muestra como "
            f"comparación (en train difiere de la filtrada en {pct(val['hmm_train_viterbi_disagreement'])} de los días). Los tres "
            "se re-ajustan cada mes solo con historia previa y se actualizan diariamente con histéresis de 3 días."),
          table(rows, widths=[3.2 * cm, 2.6 * cm, 3.0 * cm, 2.8 * cm, 5.0 * cm]),
          P("Ningún clasificador alcanza el silhouette objetivo de 0.4.", "caption"),
          P("<b>Elección con entrenamiento.</b> Se opera el clasificador con mayor Calmar del portafolio en el walk-forward de "
            "entrenamiento: el HMM filtrado. La elección no se revisó tras ver la prueba."),
          table(crow, widths=[5.0 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.4 * cm]),
          Spacer(1, 6),
          figure("16_regime_methods_train.png", "Figura 2. Entrenamiento (walk-forward 2019–2023): θ por régimen con los tres clasificadores, sin régimen y buy & hold."),
          figure("16_regime_methods_test.png", "Figura 3. Prueba (θ promedio congelado, 2024-01 → 2026-09): mismas estrategias."),
          P("En entrenamiento los tres clasificadores pierden menos que el modelo sin régimen; en prueba solo el HMM gana. "
            "Con un único periodo de prueba no es posible distinguir si eso es robusto o suerte."),
          figure("06b_regime_methods.png", "Figura 4. Línea de tiempo de regímenes causales de los tres clasificadores."),
          figure("06c_hmm_filtered_vs_viterbi.png", "Figura 5. HMM ajustado en entrenamiento: etiqueta filtrada (operable) contra Viterbi (usa el futuro).", width=15 * cm),
          figure("07_regime_distributions.png", "Figura 6. Distribución de las variables por régimen (HMM)."),
          P("<b>Validación del HMM.</b> Duración media de "
            f"{num(hmm['causal_operated']['avg_duration_days'], 1)} días, {num(hmm['causal_operated']['transitions_per_year'], 1)} "
            f"transiciones al año y probabilidad de permanencia diaria ≥ {min(tm[r][r] for r in tm):.3f}. <b>Estabilidad fuera "
            f"de muestra:</b> en entrenamiento el tiempo se reparte {pct(hmm['causal_operated']['share_train']['trend'], 0)} "
            f"tendencia / {pct(hmm['causal_operated']['share_train']['mean_reversion'], 0)} reversión / "
            f"{pct(hmm['causal_operated']['share_train']['crisis'], 0)} crisis, y en prueba "
            f"{pct(hmm['causal_operated']['share_test']['trend'], 0)} / {pct(hmm['causal_operated']['share_test']['mean_reversion'], 0)} / "
            f"{pct(hmm['causal_operated']['share_test']['crisis'], 0)}."),
          P("<b>Reglas de transición.</b> Una posición abierta conserva sus SL/TP y su holding máximo. Al cambiar de régimen: "
            "(1) los θ del nuevo régimen generan la señal y una señal opuesta cierra la posición; (2) si la fuerza con que se abrió "
            "no alcanza el mínimo del nuevo régimen, se cierra al open siguiente; (3) el cambio dispara un rebalanceo con el "
            f"multiplicador nuevo. Hubo {meta['regime_transitions_oos']} transiciones en el periodo fuera de muestra."),
          figure("08_equity_regimes.png", "Figura 7. Valor del portafolio con los regímenes del HMM superpuestos.")]
    br = pd.read_csv(RES / "metrics_by_regime_rp.csv", index_col=0)
    names = {"trend": "Tendencia", "mean_reversion": "Reversión", "crisis": "Crisis"}
    brow = [["Régimen", "Días", "Ret. acumulado", "Vol. anual", "Sharpe", "Sortino", "% días positivos"]]
    for r in ("trend", "mean_reversion", "crisis"):
        x = br.loc[r]
        brow.append([names[r], int(x["days"]), pct(x["cum_return"]), pct(x["annual_vol"]), num(x["sharpe"]),
                     num(x["sortino"]), pct(x["hit_rate"], 0)])
    s += [P("<b>Desempeño por régimen (Risk Parity, todo el periodo).</b>", "body"),
          table(brow, widths=[2.6 * cm, 1.6 * cm, 2.6 * cm, 2.2 * cm, 2.0 * cm, 2.0 * cm, 3.0 * cm]),
          P(f"Kruskal-Wallis de igualdad de distribuciones entre regímenes: p = {meta['kruskal_pvalue_by_regime']['rp']:.2f} "
            f"(Risk Parity), {meta['kruskal_pvalue_by_regime']['bh']:.2f} (buy &amp; hold).", "caption")]

    # 7. Portafolio
    rc = pd.read_csv(RES / "risk_contributions.csv", index_col=0)
    s += [PageBreak(), P("7. Portafolio multi-activo", "h1"),
          P("<b>Tres formas de asignar</b> sobre los activos con posición, con las mismas señales, costos y rebalanceo:"),
          *bullets(["<b>Pesos iguales</b> w<sub>i</sub> = 1/n (benchmark).",
                    "<b>Risk Parity naive</b> w<sub>i</sub> = (1/σ<sub>i</sub>) / Σ<sub>j</sub>(1/σ<sub>j</sub>): iguala riesgos solo si las correlaciones son iguales.",
                    "<b>Risk Parity por minimización</b>: <i>scipy.optimize.minimize</i> (L-BFGS-B) sobre la formulación convexa "
                    "min<sub>y&gt;0</sub> ½yᵀΣy − Σ b<sub>i</sub> ln y<sub>i</sub>, w = y/Σy; en el óptimo y<sub>i</sub>(Σy)<sub>i</sub> = b<sub>i</sub>, "
                    "es decir RC<sub>i</sub> = w<sub>i</sub>(Σw)<sub>i</sub>/σ<sub>p</sub> iguales (verificado en pruebas contra un método "
                    "independiente por coordenadas). Para un libro largo/corto se usa Σ<sub>d</sub> = DΣD."]),
          P("<b>Estimador de covarianza:</b> Ledoit-Wolf sobre los últimos 126 retornos diarios; con 126 observaciones y 6 "
            "activos la covarianza muestral es ruidosa y el encogimiento reduce el error de estimación. <b>Por qué Risk "
            "Parity:</b> asigna riesgo, no retorno: no requiere estimar retornos esperados y evita que el activo más volátil "
            "domine el riesgo del libro."),
          P("<b>Agregación de señales</b> (cada día al cierre): (1) dirección por activo = señal 3 de 6 mantenida con los θ del "
            "régimen vigente; (2) <b>fuerza</b> |f| de la señal que abrió la posición, con mínimo por régimen; (3) "
            "<b>conflictos</b>: si dos activos con correlación &gt; 0.6 tienen lados opuestos se descarta el de menor |f|; (4) "
            "asignación base; (5) peso final = base · |f| · multiplicador del régimen; lo no asignado queda en efectivo."),
          table([["Régimen", "Fuerza mínima para entrar", "Multiplicador de riesgo"],
                 ["Tendencia", "1/6 (cualquier señal confirmada)", "1.00"],
                 ["Reversión a la media", "2/6 (al menos 2 votos netos)", "0.75"],
                 ["Crisis", "3/6 (al menos 3 votos netos)", "0.50"],
                 ["Sin régimen", "1/6", "1.00"]], widths=[4 * cm, 6.5 * cm, 4.5 * cm]),
          Spacer(1, 6),
          P("<b>Rebalanceo:</b> cada 5 días hábiles (fijado por diseño, no optimizado) y además cuando cambia el régimen; "
            "entradas y salidas siempre al open siguiente; ajustes menores a 1% del capital no se ejecutan."),
          table(metrics_rows(["Risk Parity", "Risk Parity naive (1/σ)", "Pesos iguales", "Buy & hold equiponderado"]),
                widths=[3.0 * cm, 1.75 * cm, 1.6 * cm, 1.75 * cm, 1.8 * cm, 1.75 * cm, 1.45 * cm, 1.6 * cm, 2.0 * cm]),
          P("Entrenamiento: walk-forward 2019–2023. Prueba: θ promedio congelado 2024-01 → 2026-09. Win rate sobre operaciones "
            "cerradas con P&amp;L neto de comisiones.", "caption"),
          figure("01_equity.png", "Figura 8. Valor del portafolio con sus benchmarks."),
          figure("02_drawdown.png", "Figura 9. Curva de drawdown."),
          figure("03_monthly_returns_rp.png", "Figura 10. Retornos mensuales y anuales (Risk Parity).", width=15 * cm),
          figure("03b_quarterly_returns_rp.png", "Figura 11. Retornos trimestrales (Risk Parity).", width=9 * cm),
          table([["Activo"] + list(rc.columns)] + [[t] + [pct(x) for x in rc.loc[t]] for t in rc.index],
                widths=[3 * cm, 4 * cm, 4.5 * cm, 4 * cm]),
          P("Contribución promedio al riesgo del libro realizado (fin de cada mes). La contribución igual sería 16.7%.", "caption"),
          figure("09_risk_contributions.png", "Figura 12. Contribuciones al riesgo por activo.", width=13 * cm),
          figure("13_assets_vs_portfolio.png", "Figura 13. Portafolio contra la estrategia individual de cada activo ($1M cada una)."),
          figure("10_signal_heatmap.png", "Figura 14. Mapa de calor de la fuerza de la señal mantenida por activo."),
          figure("11_correlation_regimes.png", "Figura 15. Matriz de correlación de retornos por régimen (HMM)."),
          P(f"<b>Costos de transacción.</b> La rotación anual es {rp_tot['turnover']:.1f} veces el capital y las comisiones suman "
            f"${rp_tot['total_costs']:,.0f}, contra un retorno bruto (antes de comisiones) de {pct(rp_reb.loc[5, 'gross_return'])} "
            f"acumulado; el neto es {pct(rp_reb.loc[5, 'net_return'])}."),
          figure("12_rebalance_sweep.png", "Figura 16. Retorno neto total contra frecuencia de rebalanceo.", width=13 * cm),
          figure("05_cost_curve.png", "Figura 17. Retorno neto anualizado contra nivel de comisión.", width=13 * cm)]

    # 8. Preguntas
    cf = pd.read_csv(RES / "confirmation_vs_single.csv").groupby(["rule", "period"])[["n_entries", "calmar"]].median()
    rule_keys = ["3 de 6", "ADX", "RSI", "OBV", "SKEW", "AUTOCORR", "VOLZ"]
    qrow = [["Regla", "Operaciones train", "Calmar train", "Operaciones test", "Calmar test"]]
    for r in rule_keys:
        qrow.append([r, num(cf.loc[(r, "train"), "n_entries"], 1), num(cf.loc[(r, "train"), "calmar"]),
                     num(cf.loc[(r, "test"), "n_entries"], 1), num(cf.loc[(r, "test"), "calmar"])])
    singles_tr = cf.xs("train", level="period").drop("3 de 6")
    n_confirm = cf.loc[("3 de 6", "train"), "n_entries"]
    busier = singles_tr[singles_tr["n_entries"] > n_confirm]
    quieter = singles_tr[singles_tr["n_entries"] <= n_confirm]
    top_sens = sens.head(3)
    s += [PageBreak(), P("8. Preguntas de análisis", "h1"),
          P("1. ¿Qué aporta la confirmación 3 de 6 frente a un solo indicador?", "h2"),
          P("Con el θ promedio de cada activo se repitió el backtest usando cada voto por sí solo como señal (mediana de los "
            "seis activos):"),
          table(qrow, widths=[3 * cm, 3.4 * cm, 3 * cm, 3.4 * cm, 3 * cm]),
          Spacer(1, 4),
          P(f"La confirmación opera menos que la mayoría de los indicadores sueltos: {num(cf.loc[('3 de 6', 'train'), 'n_entries'], 0)} operaciones medianas por "
            f"activo en entrenamiento contra {num(busier['n_entries'].min(), 0)}–{num(busier['n_entries'].max(), 0)} de "
            f"{', '.join(busier.index)} por sí solos ({', '.join(quieter.index)} operan menos porque votan pocos días). "
            "Pero a nivel de cada activo no mejora el Calmar: "
            f"{num(cf.loc[('3 de 6', 'train'), 'calmar'])} en entrenamiento y {num(cf.loc[('3 de 6', 'test'), 'calmar'])} en "
            "prueba. Su efecto principal es bajar la rotación y los costos; la mejora del portafolio viene de la capa de "
            "régimen y de la asignación."),
          figure("15_confirmation_calmar.png", "Figura 18. Calmar mediano por activo: regla 3 de 6 contra un solo indicador.", width=12 * cm),
          P("2. ¿Cuánto se degrada el desempeño entre entrenamiento y prueba en el walk-forward?", "h2"),
          P(f"En cada ventana el θ global obtiene en sus 6 meses un retorno anualizado promedio de "
            f"{pct(deg['mean_is_annual_return'])} (Calmar mediano {num(deg['median_is_calmar'], 1)}); en el mes siguiente obtiene "
            f"{pct(deg['mean_oos_annual_return'])} (Calmar mediano {num(deg['median_oos_calmar'], 1)}, muchos meses sin "
            f"operaciones). <b>Sobrevive {wfe:.0%} de la ventaja</b>, debajo del umbral de 50% que distingue ventaja real de "
            f"ruido ajustado. Con el θ promedio congelado, {n_frozen_losers} de 6 activos pierden en prueba de forma individual."),
          P("3. ¿Qué tan sensible es la estrategia a ±20% en sus parámetros?", "h2"),
          P("Se varió cada parámetro del θ promedio de cada activo, uno a la vez, y se midió el Calmar de entrenamiento. "
            f"Los parámetros más sensibles (mediana de |ΔCalmar| ante ±20%) son {', '.join(f'{p} ({v:.2f})' for p, v in top_sens['med'].items())}. "
            f"En {flips:.0%} de los casos el Calmar cambia de signo. Los parámetros del RSI casi no mueven el resultado "
            "porque el RSI vota en pocos días. Como el Calmar base ya es bajo o negativo en la mayoría de los activos "
            f"({int((sens_base < 0).sum())} de 6), la superficie es más bien plana alrededor de un valor pobre: no hay un pico "
            "aislado que defender, pero tampoco una meseta rentable."),
          figure("04_sensitivity.png", "Figura 19. Calmar de entrenamiento ante variaciones de ±10% y ±20% de cada parámetro."),
          P("4. ¿A qué costo deja de ser rentable?", "h2"),
          P(f"Sin comisiones el portafolio Risk Parity rinde {pct(zero_cost, 2)} anual; el equilibrio está en "
            f"{be['rp'] * 100:.3f}% por operación (naive {be['naive'] * 100:.3f}%, pesos iguales {be['ew'] * 100:.3f}%). Frente "
            f"al 0.125% especificado el margen de seguridad es de {be['rp'] / 0.00125:.1f} veces. Con la regla anterior (3 "
            "indicadores, 2 de 3) el equilibrio estaba en 0.008%: reducir la actividad fue lo que hizo viable la estrategia "
            "frente a costos. En el barrido de rebalanceo, rebalancear cada 5 días (el valor fijado) da el mayor retorno neto."),
          P("5. ¿El desempeño difiere entre regímenes? ¿Qué aporta la capa de régimen?", "h2"),
          P(f"El Sharpe por régimen va de {br['sharpe'].min():.2f} a {br['sharpe'].max():.2f}, pero la prueba de Kruskal-Wallis no "
            f"rechaza igualdad (p = {meta['kruskal_pvalue_by_regime']['rp']:.2f}). La capa de régimen sí cambia el resultado del "
            "portafolio: en entrenamiento los tres clasificadores pierden menos que el modelo sin régimen y en prueba el HMM gana "
            "mientras el modelo sin régimen pierde. Su efecto viene sobre todo de controlar la exposición (entradas más "
            "estrictas y multiplicadores &lt; 1), y depende del clasificador."),
          P("6. ¿Risk Parity mejora el Calmar frente a pesos iguales?", "h2"),
          P(f"Ligeramente: Calmar total {num(rp_tot['calmar'], 3)} con Risk Parity, {num(nv_tot['calmar'], 3)} con la versión naive y "
            f"{num(ew_tot['calmar'], 3)} con pesos iguales; MDD {pct(rp_tot['max_drawdown'])}, {pct(nv_tot['max_drawdown'])} y "
            f"{pct(ew_tot['max_drawdown'])}. Risk Parity reparte mejor el riesgo entre activos (contribuciones de "
            f"{pct(rc['Risk Parity'].min(), 0)} a {pct(rc['Risk Parity'].max(), 0)} contra {pct(rc['Pesos iguales'].min(), 0)} a "
            f"{pct(rc['Pesos iguales'].max(), 0)}), a costa de algo más de rotación ({rp_tot['turnover']:.1f} contra "
            f"{ew_tot['turnover']:.1f} veces el capital al año). Con exposición promedio de "
            f"{pct(meta['avg_gross_exposure']['rp'], 0)} la diferencia entre asignaciones es pequeña."),
          P("7. Tres limitaciones para operar con capital real", "h2"),
          *bullets([
              "<b>Robustez:</b> la ganancia de prueba depende de un clasificador de régimen; la eficiencia del walk-forward es "
              f"{wfe:.2f}; y la estrategia se rediseñó después de ver resultados de prueba (sección 10).",
              "<b>Ejecución:</b> se asume llenado completo al open o al nivel del stop (en un gap el stop puede ejecutarse peor), "
              "disponibilidad ilimitada de cortos y sin costo de préstamo ni spread.",
              "<b>Selección del universo:</b> sesgo de supervivencia en los 6 activos y un único periodo dominado por un mercado "
              "alcista; el buy &amp; hold supera ampliamente a la estrategia."])]

    # 9. Ejecución, sesgos y conclusiones
    s += [P("9. Limitación de ejecución: impacto de mercado", "h1"),
          P(f"Estimamos el impacto omitido con la ley de raíz cuadrada, impacto ≈ σ<sub>diaria</sub>·√(Q/V)·nocional por "
            f"ejecución. La participación mediana es {imp['median_participation'] * 100:.4f}% del volumen diario (máxima "
            f"{imp['max_participation'] * 100:.3f}%). El costo estimado es ${imp['impact_cost_total']:,.0f} en todo el periodo "
            f"(≈{pct(imp['impact_cost_annual_pct_equity'], 2)} del capital por año), "
            f"{imp['impact_cost_total'] / imp['commission_total']:.0%} de las comisiones pagadas. Como proporción del capital crece "
            "con la raíz del tamaño de las órdenes."),
          P("10. Auditoría de sesgos y desviaciones declaradas", "h1"),
          table([["Sesgo / desviación", "Evidencia en este proyecto"],
                 ["Look-ahead", "Señal en t con datos ≤ t y ejecución en t+1 (prueba de truncamiento); régimen re-ajustado mes a mes con historia previa y HMM con etiqueta filtrada, no Viterbi (prueba de invariancia a datos futuros)."],
                 ["Data snooping sobre la prueba", "La versión anterior (3 indicadores, regla 2 de 3) perdía en prueba (−1.9% anual). Después de verlo se rediseñó la estrategia (6 indicadores, 3 de 6, RSI con zona de tolerancia, autocorrelación relajada) y se adoptó el protocolo de θ promedio congelado indicado por el profesor. El resultado de prueba de esta versión no es una evaluación ciega."],
                 ["Supervivencia", "Universo elegido en 2026 entre empresas que siguen siendo grandes."],
                 ["Sobreajuste", f"{meta['n_configs_evaluated']:,} configuraciones; eficiencia OOS/IS {wfe:.2f}; el clasificador de régimen se eligió con entrenamiento."],
                 ["Ejecución optimista", f"Llenado completo sin spread, préstamo ni impacto; impacto estimado ≈{pct(imp['impact_cost_annual_pct_equity'], 2)} anual."],
                 ["Desviaciones del enunciado", "Regla 3 de 6 en lugar de 2 de 3 (autorizada por el profesor, mínimo 3 indicadores); régimen actualizado diario (datos diarios, no de 5 minutos)."]],
                widths=[3.8 * cm, 12.8 * cm]),
          P("11. Conclusiones", "h1"),
          P("La implementación cumple la estructura y las pruebas del laboratorio. Reducir la actividad con la regla 3 de 6 "
            f"bajó las comisiones y llevó la comisión de equilibrio a {be['rp'] * 100:.2f}%, por encima del costo del laboratorio, "
            f"y el portafolio con régimen HMM obtuvo {pct(rp_te['annual_return'])} anual en prueba con drawdown bajo. Aun así, la "
            "ventaja no está demostrada: depende del clasificador, la eficiencia del walk-forward es baja, el diseño se ajustó "
            "después de ver la prueba y el buy &amp; hold rinde mucho más en este periodo alcista. El siguiente paso sería "
            "evaluar la estrategia congelada en un periodo nuevo no observado.")]

    doc = SimpleDocTemplate(str(DOCS / "reporte.pdf"), pagesize=letter, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="Lab 02 — Reporte ejecutivo", author=EQUIPO)

    def footer(c, d):
        c.saveState()
        c.setFont("Sans", 8)
        c.setFillColor(MUTED)
        c.drawString(2.2 * cm, 1.2 * cm, f"Lab02_MyST_{EQUIPO.replace(' ', '')} · Reporte ejecutivo")
        c.drawRightString(letter[0] - 2.2 * cm, 1.2 * cm, f"{d.page}")
        c.restoreState()

    doc.build(s, onFirstPage=lambda c, d: None, onLaterPages=footer)


# ----------------------------------------------------------------------------------------------
# Presentación (16:9)
# ----------------------------------------------------------------------------------------------
W, H = 33.867 * cm, 19.05 * cm


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H))
        self.c.setTitle("Lab 02 — Presentación")
        self.c.setAuthor(EQUIPO)
        self.n = 0

    def _bg(self, dark=False):
        self.c.setFillColor(INK if dark else colors.white)
        self.c.rect(0, 0, W, H, stroke=0, fill=1)

    def cover(self, title, subtitle, lines):
        self._bg(dark=True)
        c = self.c
        c.setFillColor(ACCENT)
        c.rect(2 * cm, 11.2 * cm, 3 * cm, 0.25 * cm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Sans-Bold", 31)
        y = 9.4 * cm
        for line in title:
            c.drawString(2 * cm, y, line)
            y -= 1.5 * cm
        c.setFont("Sans", 17)
        c.setFillColor(colors.HexColor("#b8c7cf"))
        c.drawString(2 * cm, y - 0.2 * cm, subtitle)
        c.setFont("Sans", 13)
        yy = 3.2 * cm
        for line in lines:
            c.drawString(2 * cm, yy, line)
            yy -= 0.65 * cm
        c.showPage()

    def slide(self, title, kicker=None):
        self._bg()
        self.n += 1
        c = self.c
        c.setFillColor(ACCENT)
        c.rect(0, H - 0.35 * cm, W, 0.35 * cm, stroke=0, fill=1)
        if kicker:
            c.setFont("Sans-Bold", 11)
            c.setFillColor(ACCENT)
            c.drawString(1.6 * cm, H - 1.55 * cm, kicker.upper())
        c.setFont("Sans-Bold", 25)
        c.setFillColor(INK)
        c.drawString(1.6 * cm, H - 2.65 * cm, title)
        c.setFont("Sans", 9)
        c.setFillColor(MUTED)
        c.drawRightString(W - 1.2 * cm, 0.7 * cm, f"{self.n}")
        c.drawString(1.6 * cm, 0.7 * cm, f"Lab 02 · Nivel C · {EQUIPO}")

    def text(self, x, y, items, size=15, width=13 * cm, gap=0.45 * cm, bullet=True, color=INK):
        style = ParagraphStyle("s", fontName="Sans", fontSize=size, leading=size * 1.3, textColor=color)
        for it in items:
            p = Paragraph(("• " if bullet else "") + it, style)
            _, h = p.wrap(width, H)
            p.drawOn(self.c, x, y - h)
            y -= h + gap
        return y

    def image(self, name, x, y, w=None, h=None):
        img = ImageReader(str(FIG / name))
        iw, ih = img.getSize()
        if w is not None and h is not None:
            scale = min(w / iw, h / ih)
            y += h - ih * scale
            w, h = iw * scale, ih * scale
        elif w is not None:
            h = w * ih / iw
        else:
            w = h * iw / ih
        self.c.drawImage(img, x, y, w, h, preserveAspectRatio=True, mask="auto")
        return w, h

    def big_number(self, x, y, value, label, color=INK):
        c = self.c
        c.setFont("Sans-Bold", 34)
        c.setFillColor(color)
        c.drawString(x, y, value)
        c.setFont("Sans", 12)
        c.setFillColor(MUTED)
        c.drawString(x, y - 0.75 * cm, label)

    def table(self, rows, x, y, widths, size=11.5):
        data = [[Paragraph(str(v), ParagraphStyle("t", fontName="Sans-Bold" if i == 0 else "Sans", fontSize=size,
                                                   leading=size * 1.25, textColor=colors.white if i == 0 else INK))
                 for v in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3d9")),
                               ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        _, h = t.wrap(sum(widths), H)
        t.drawOn(self.c, x, y - h)
        return h

    def end(self):
        self.c.showPage()

    def save(self):
        self.c.save()


def build_deck():
    rp_tot, rp_tr, rp_te = (perf.loc[("Risk Parity", p)] for p in ("total", "train", "test"))
    bh_tot = perf.loc[("Buy & hold equiponderado", "total")]
    deg = meta["degradation"]["train"]
    wfe = deg["mean_oos_annual_return"] / deg["mean_is_annual_return"]
    be = meta["breakeven_commission"]
    vc = meta["vote_correlation"]
    br = pd.read_csv(RES / "metrics_by_regime_rp.csv", index_col=0)
    cf = pd.read_csv(RES / "confirmation_vs_single.csv").groupby(["rule", "period"])[["n_entries", "calmar"]].median()
    singles_tr = cf.xs("train", level="period").drop("3 de 6")
    n_confirm = cf.loc[("3 de 6", "train"), "n_entries"]
    busier = singles_tr[singles_tr["n_entries"] > n_confirm]
    sens, flips, sens_base = sensitivity_summary()
    d = Deck(DOCS / "presentacion.pdf")
    top = H - 3.6 * cm

    d.cover(["Laboratorio 02", "Estrategias de trading con análisis técnico"],
            f"Nivel C · 6 acciones · Risk Parity · {EQUIPO}",
            [n for n, _ in TEAM] + [f"Microestructuras y Sistemas de Trading · ITESO · {FECHA}"])

    # 1. Planteamiento
    d.slide("¿Hay una ventaja real o solo ajuste a la historia?", "Planteamiento")
    d.text(1.6 * cm, top, [
        "<b>Universo:</b> AAPL, MSFT, JPM, XOM, JNJ, WMT — diarios ajustados, 2018 → 2026-09",
        "<b>Capital:</b> $1,000,000 · comisión 0.125% por lado · sin apalancamiento · largos y cortos",
        f"<b>Walk-forward en train:</b> 6 meses → 1 mes, paso mensual ({meta['n_windows']} ventanas, 2019–2023)",
        "<b>Test 2024-01 → 2026-09:</b> θ promedio por activo, congelado"], width=17 * cm)
    d.big_number(21 * cm, top - 1.6 * cm, pct(rp_te["annual_return"]), "Risk Parity, retorno anual en test", ACCENT)
    d.big_number(21 * cm, top - 4.6 * cm, pct(bh_tot["annual_return"]), "Buy & hold equiponderado (todo el periodo)")
    d.big_number(21 * cm, top - 7.6 * cm, f"{meta['n_configs_evaluated']:,}", "configuraciones evaluadas")
    d.end()

    # 2. Estrategia
    vf = vc["vote_frequency"]
    d.slide("Seis indicadores, regla 3 de 6", "Estrategia por activo")
    d.table([["Familia", "Indicador", "Voto", "% días"],
             ["Tendencia", "ADX ± DI", "dirección si ADX > umbral", pct(vf["v_adx"], 0)],
             ["Momento", "RSI con tolerancia", "reversión tras k días en zona", pct(vf["v_rsi"], 0)],
             ["Volumen", "Desbalance OBV", "±1 fuera de ±umbral", pct(vf["v_obv"], 0)],
             ["Distribución", "Skew", "±1 fuera de ±0.5", pct(vf["v_skew"], 0)],
             ["Régimen", "Autocorrelación ρ1", "racha si ρ1 significativa", pct(vf["v_autocorr"], 0)],
             ["Vol-momento", "z de volumen", "sign(r) si z > 1", pct(vf["v_volz"], 0)]],
            1.6 * cm, top, [3.4 * cm, 4.2 * cm, 6.4 * cm, 2.2 * cm], size=12)
    d.c.setFillColor(LIGHT)
    d.c.roundRect(18.6 * cm, 9.3 * cm, 13.8 * cm, 3.4 * cm, 8, stroke=0, fill=1)
    d.text(19.1 * cm, 12.2 * cm, ["Largo (s<sub>t</sub> = +1): N<super>+</super> ≥ 3 y N<super>+</super> &gt; N<super>−</super>",
                                  "Corto (s<sub>t</sub> = −1): N<super>−</super> ≥ 3 y N<super>−</super> &gt; N<super>+</super>"],
           size=15, width=13 * cm, bullet=False, gap=0.25 * cm)
    d.text(18.6 * cm, 8.6 * cm, ["Empate 3–3: no abre posición",
                                 "Señal al cierre de <b>t</b>, ejecución al open de <b>t+1</b>",
                                 "Salida: SL/TP por ATR (gana el SL si ambos en la barra), señal opuesta u holding máximo"],
           size=13.5, width=13.6 * cm)
    d.end()

    # 3. RSI y correlación
    d.slide("RSI con zona de tolerancia y votos no redundantes", "Diseño de las señales")
    d.text(1.6 * cm, top, [
        "Entra a sobrecompra al pasar de <b>L</b>; sigue dentro mientras no baje de <b>L − b</b> (un 79 no lo saca de 80)",
        "Tras <b>k</b> días seguidos en la zona vota <b>−1</b> (reversión); simétrico en sobreventa (+1)",
        "<b>L, b y k</b> los elige Optuna",
        f"Restricción en la optimización: |ρ| entre votos &lt; {vc['threshold']}",
        f"Fuera de muestra: máx. |ρ| = {vc['max_abs_train']:.2f} (train) y {vc['max_abs_test']:.2f} (test)"],
        size=14, width=13.4 * cm)
    d.image("17_vote_correlation.png", 15.4 * cm, 1.4 * cm, w=17.4 * cm, h=14.4 * cm)
    d.end()

    # 4. Motor y protocolo
    d.slide("Motor event-driven y θ congelado para test", "Backtest y walk-forward")
    d.text(1.6 * cm, top, [
        "Estado explícito: efectivo, acciones por activo, valor = efectivo + posiciones",
        "Comisión en cada apertura, cierre y rebalanceo · nocional bruto ≤ capital",
        "Optuna TPE, 100 pruebas por estudio · θ global + θ por régimen × 3 clasificadores",
        "Restricciones: ≥ 4 operaciones por ventana y votos con |ρ| &lt; 0.5",
        "Test: se promedian los θ de las 60 ventanas de cada activo y se congelan",
        f"{N_TESTS} pruebas automáticas (causalidad, 3 de 6, RSI, contabilidad, régimen, Risk Parity)"], width=17 * cm)
    d.big_number(21 * cm, top - 1.6 * cm, f"≈{meta['wf_minutes']:.0f} min", "cómputo del walk-forward, 8 procesos")
    d.big_number(21 * cm, top - 4.6 * cm, f"{meta['n_windows']}", "ventanas mensuales en train")
    d.big_number(21 * cm, top - 7.6 * cm, pct(audit["share_of_all_studies_affected"]),
                 "estudios donde el Calmar premió más drawdown")
    d.end()

    # 5. Régimen
    d.slide("Régimen: el HMM se opera con la etiqueta filtrada", "Detección de régimen")
    rows = [["Clasificador", "Silhouette", "Días por régimen"]]
    for m, name in (("rules", "Reglas"), ("kmeans", "K-means"), ("hmm", "HMM filtrado")):
        v = val["methods"][m]
        rows.append([name, num(v["train_fit"]["silhouette"], 3), num(v["causal_operated"]["avg_duration_days"], 0)])
    d.table(rows, 1.6 * cm, top, [4.2 * cm, 3.2 * cm, 4.2 * cm], size=12.5)
    d.text(1.6 * cm, top - 4.6 * cm, [
        "Vol, eficiencia y autocorrelación en ventana de 3 meses",
        "Filtrada: P(S<sub>t</sub> | x<sub>1..t</sub>); Viterbi usa el futuro → solo comparación",
        "Ninguno llega a silhouette 0.4"], size=13.5, width=12 * cm)
    d.image("06c_hmm_filtered_vs_viterbi.png", 14 * cm, 1.4 * cm, w=18.8 * cm, h=14.4 * cm)
    d.end()

    # 6. Comparación de clasificadores
    d.slide("Con régimen se pierde menos; en test solo gana el HMM", "Comparación en train y test")
    d.image("16_regime_methods_train.png", 1.0 * cm, 8.6 * cm, w=20.5 * cm, h=7.3 * cm)
    d.image("16_regime_methods_test.png", 1.0 * cm, 1.2 * cm, w=20.5 * cm, h=7.3 * cm)
    rows = [["Valor final ($1M)", "Train", "Test"]]
    for n, lab in (("Régimen · Reglas", "Reglas"), ("Régimen · K-means", "K-means"),
                   ("Régimen · HMM (filtrada)", "HMM (elegido)"), ("Sin régimen", "Sin régimen"),
                   ("Buy & hold equiponderado", "Buy & hold")):
        rows.append([lab] + [f"${1e6 * (1 + comp.loc[(n, p), 'total_return']) / 1000:,.0f}k" for p in ("train", "test")])
    d.table(rows, 22.2 * cm, top, [4.4 * cm, 3 * cm, 3 * cm], size=12.5)
    d.text(22.2 * cm, top - 6.4 * cm, ["El método se eligió con <b>train</b> (mayor Calmar)",
                                       f"Kruskal-Wallis por régimen: p = {meta['kruskal_pvalue_by_regime']['rp']:.2f}"],
           size=12.5, width=10.4 * cm)
    d.end()

    # 7. Portafolio
    d.slide("Tres formas de asignar el riesgo", "Portafolio")
    rows = [["", "Calmar", "MDD", "Ret. anual"]]
    for n, lab in (("Risk Parity", "RP minimize"), ("Risk Parity naive (1/σ)", "Naive 1/σ"), ("Pesos iguales", "1/n")):
        x = perf.loc[(n, "total")]
        rows.append([lab, num(x["calmar"], 3), pct(x["max_drawdown"]), pct(x["annual_return"])])
    d.table(rows, 1.6 * cm, top, [4 * cm, 2.6 * cm, 2.6 * cm, 3 * cm], size=13)
    d.text(1.6 * cm, top - 5 * cm, ["Minimize: scipy.optimize.minimize sobre ½yᵀΣy − Σ b ln y",
                                    "Covarianza Ledoit-Wolf, 126 días",
                                    f"Exposición media {pct(meta['avg_gross_exposure']['rp'], 0)}: diferencias pequeñas"],
           size=13.5, width=12.8 * cm)
    d.image("09_risk_contributions.png", 15.2 * cm, 1.4 * cm, w=17.6 * cm, h=14.4 * cm)
    d.end()

    # 8. Resultados
    d.slide("Resultados: positivo en test, lejos del buy & hold", "Resultados")
    d.image("01_equity.png", 1.2 * cm, 1.4 * cm, w=21.5 * cm, h=14.2 * cm)
    d.big_number(23.6 * cm, top - 1.4 * cm, pct(rp_te["annual_return"]), "retorno anual RP en test", ACCENT)
    d.big_number(23.6 * cm, top - 4.2 * cm, num(rp_te["calmar"]), "Calmar en test")
    d.big_number(23.6 * cm, top - 7.0 * cm, pct(rp_tr["annual_return"]), "retorno anual RP en train (WF)", WARN)
    d.big_number(23.6 * cm, top - 9.8 * cm, f"${rp_tot['total_costs'] / 1000:,.0f}k", "en comisiones")
    d.end()

    # 9. Q1 y Q2
    d.slide("Menos operaciones, pero la ventaja in-sample no sobrevive", "Preguntas 1 y 2")
    d.image("14_confirmation_trades.png", 1.2 * cm, 1.4 * cm, w=16 * cm, h=13.5 * cm)
    d.text(18.2 * cm, top, [
        f"<b>3 de 6 vs. un indicador:</b> {num(cf.loc[('3 de 6', 'train'), 'n_entries'], 0)} operaciones vs. "
        f"{num(busier['n_entries'].min(), 0)}–{num(busier['n_entries'].max(), 0)} de {', '.join(busier.index)} (train)",
        "No mejora el Calmar por activo: baja la rotación y los costos",
        f"<b>Walk-forward:</b> retorno IS {pct(deg['mean_is_annual_return'], 0)} → OOS {pct(deg['mean_oos_annual_return'], 0)}",
        f"Sobrevive {wfe:.0%} de la ventaja (&lt; 50% → ruido ajustado)"], size=14.5, width=14 * cm)
    d.end()

    # 10. Q3
    d.slide("Sensibilidad ±20%: plana, pero alrededor de un valor pobre", "Pregunta 3")
    d.image("04_sensitivity.png", 1.2 * cm, 1.2 * cm, w=21 * cm, h=14.6 * cm)
    d.text(23 * cm, top, [f"Más sensibles: {', '.join(sens.head(3).index)}",
                          f"El Calmar cambia de signo en {flips:.0%} de los casos",
                          f"Calmar base negativo en {int((sens_base < 0).sum())} de 6 activos",
                          "Los parámetros del RSI casi no mueven el resultado"], size=13.5, width=9.6 * cm)
    d.end()

    # 11. Q4
    d.slide("Operar menos hizo viable la estrategia frente a costos", "Pregunta 4 · costo de equilibrio")
    d.image("05_cost_curve.png", 1.2 * cm, 1.4 * cm, w=18.5 * cm, h=14.2 * cm)
    d.big_number(21 * cm, top - 1.4 * cm, f"{be['rp'] * 100:.2f}%", "comisión de equilibrio (Risk Parity)", ACCENT)
    d.big_number(21 * cm, top - 4.2 * cm, "0.125%", "comisión del laboratorio")
    d.text(21 * cm, top - 6.6 * cm, [f"Margen de seguridad: {be['rp'] / 0.00125:.1f}×",
                                     "Con la regla anterior (2 de 3) el equilibrio era 0.008%",
                                     f"Rotación ≈ {rp_tot['turnover']:.0f}× el capital al año"],
           size=13.5, width=11 * cm)
    d.end()

    # 12. Limitaciones y conclusión
    d.slide("Preguntas 5–7: limitaciones y conclusión", "Cierre del análisis")
    d.text(1.6 * cm, top, [
        f"<b>Régimen:</b> Sharpe por régimen de {br['sharpe'].min():.2f} a {br['sharpe'].max():.2f}, no significativo; "
        "controla exposición y depende del clasificador",
        "<b>Risk Parity vs. 1/n:</b> reparte mejor el riesgo; Calmar ligeramente mayor",
        "<b>Data snooping:</b> la estrategia se rediseñó tras ver la prueba de la versión anterior",
        f"<b>Ejecución:</b> llenado completo al open; impacto estimado ≈{pct(meta['market_impact']['impact_cost_annual_pct_equity'], 2)} anual",
        "<b>Selección:</b> universo con sesgo de supervivencia, un solo periodo alcista"], size=14, width=19 * cm)
    d.c.setFillColor(INK)
    d.c.roundRect(21.6 * cm, 3 * cm, 11 * cm, 10.6 * cm, 10, stroke=0, fill=1)
    d.text(22.3 * cm, 12.8 * cm, ["<b>Conclusión</b>",
                                  "Operar menos volvió la estrategia viable frente a costos y positiva en test con el HMM.",
                                  "La ventaja no está demostrada: falta probarla congelada en un periodo nuevo."],
           size=15, width=9.6 * cm, bullet=False, color=colors.white)
    d.end()

    # Cierre
    d._bg(dark=True)
    c = d.c
    c.setFillColor(colors.white)
    c.setFont("Sans-Bold", 36)
    c.drawString(2 * cm, 10.5 * cm, "Gracias")
    c.setFont("Sans", 16)
    c.setFillColor(colors.HexColor("#b8c7cf"))
    c.drawString(2 * cm, 8.8 * cm, "Preguntas")
    c.setFont("Sans", 12)
    c.drawString(2 * cm, 3.6 * cm, " · ".join(n for n, _ in TEAM))
    c.drawString(2 * cm, 2.8 * cm, f"{REPO} · reproducir con: python main.py")
    c.showPage()
    d.save()
    return d.n


if __name__ == "__main__":
    build_report()
    n = build_deck()
    print(f"docs/reporte.pdf y docs/presentacion.pdf generados ({n} diapositivas de contenido)")
