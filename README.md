# Lab02_MyST_Equipo5 — Estrategias de trading con análisis técnico

**Integrantes y activos a cargo** (2 activos por integrante):

| Integrante | Activos |
|---|---|
| Adrián Marcelo Ballesteros Herrera | AAPL, MSFT |
| Erik del Castillo Román | JPM, XOM |
| Jesús Emmanuel Flores Cortés | JNJ, WMT |

**Nivel de alcance: C** (3 integrantes → 6 acciones diarias, régimen de mercado y portafolio Risk Parity).

## Descripción

Sistema de trading sistemático sobre seis acciones líquidas de EE. UU. (datos diarios ajustados,
2018-01-02 → 2026-09-30). Cada activo genera señales con seis indicadores de familias distintas
(ADX±DI, RSI con zona de tolerancia en modo reversión, desbalance de OBV, skew de retornos,
autocorrelación de rezago 1 y z-score de volumen) y abre posición cuando al menos 3 de 6 coinciden
en dirección; la correlación entre los votos se restringe a |ρ| < 0.5. Un motor event-driven con
comisión de 0.125% por ejecución, sin apalancamiento, con stop-loss y take-profit por ATR, simula la
estrategia. Los hiperparámetros se optimizan con Optuna (TPE, 100 pruebas por estudio) maximizando el
Calmar en un walk-forward de 6 meses de entrenamiento / 1 mes de prueba con paso mensual dentro del
periodo de entrenamiento (2019–2023), por activo y por régimen de mercado; para el periodo de prueba
(2024-01 → 2026-09) se promedian los θ de cada activo y se congelan. El régimen (tendencia, reversión
a la media, crisis) se detecta con variables en ventana móvil de 3 meses y tres clasificadores
causales: reglas, K-means y un HMM gaussiano operado con su etiqueta filtrada (algoritmo forward,
nunca Viterbi); el que se opera se elige solo con entrenamiento. Las señales se agregan en un
portafolio con tres formas de asignación: pesos iguales (1/n), Risk Parity naive (1/σ) y Risk Parity
por minimización (`scipy.optimize.minimize`, covarianza Ledoit-Wolf), con las mismas señales, costos
y rebalanceo, y se comparan contra buy & hold equiponderado.

## Instalación

Requiere Python 3.12. Se recomienda un entorno virtual (`.venv/` está en `.gitignore`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

## Reproducir todos los resultados

```bash
python main.py
```

El walk-forward tarda ≈ 55 minutos de cómputo con 8 núcleos (corre en paralelo con joblib y guarda
cada bloque de ventanas en `results/cache/`, así que una corrida interrumpida retoma donde se quedó).
Escribe tablas (`.csv`, `.json`) y figuras (`results/figures/*.png`) en `results/`. Los datos están
congelados en `data/`; solo se descargan de Yahoo Finance si faltan los CSV.

Pruebas:

```bash
python -m pytest -q tests
```

Análisis y figuras: `notebooks/analysis.ipynb` (lee lo que dejó `main.py` en `results/`).

Reporte y presentación: `python docs/build_docs.py` regenera `docs/reporte.pdf` y
`docs/presentacion.pdf` a partir de `results/` (usa las fuentes DejaVu que trae matplotlib, así que corre
en cualquier sistema operativo).

## Semilla

`SEED = 42` (en `main.py` para `random`/`numpy`, en `src/optimize.py` para Optuna y en
`src/regimes.py` para K-means y el HMM). Cada estudio de Optuna usa una semilla derivada de la
ventana, el activo, el clasificador y el régimen, de modo que los resultados no dependen del orden en
que los procesos paralelos terminan.

## Estructura

```
main.py              orquesta todo el proyecto
data/                precios diarios congelados (un CSV por activo)
src/data.py          descarga, carga, validación y auditoría de datos
src/signals.py       seis indicadores, votos y regla de confirmación 3 de 6
src/backtest.py      motor event-driven (efectivo, posiciones, equity) con costos y SL/TP
src/metrics.py       Sharpe, Sortino, Calmar, MDD, win rate, tablas de retornos, impacto estimado
src/optimize.py      espacio de parámetros, Optuna, walk-forward, θ promedio congelado, sensibilidad
src/regimes.py       variables de régimen; reglas, K-means y HMM filtrado causales; validación
src/portfolio.py     pesos iguales, Risk Parity naive y por minimización, agregación, rebalanceo
src/plots.py         figuras
tests/               pruebas con pytest
notebooks/           análisis y figuras (sin lógica)
docs/                reporte.pdf, presentacion.pdf, build_docs.py (los genera) y hallazgo_calmar.md
results/             tablas y figuras que generan main.py y leen el notebook y build_docs.py
```

## Supuestos principales

- Señal calculada al cierre de t, ejecución al open de t+1.
- Comisión de 0.125% sobre el nocional de cada ejecución (entrada, salida y ajustes de rebalanceo).
  No se modelan spread, préstamo de acciones en cortos ni impacto de mercado; el impacto se estima
  aparte (`metrics.estimate_market_impact`).
- Empate intrabar entre SL y TP: se ejecuta primero el stop-loss. Si el open abre más allá de un
  nivel, la salida es al open.
- Sin apalancamiento: el nocional bruto ejecutado nunca excede equity / (1 + comisión).
- Regla 3 de 6 en lugar de 2 de 3, autorizada por el profesor (mínimo 3 indicadores).
- Función objetivo: Calmar, como exige el laboratorio. Con retorno negativo el Calmar puede premiar
  drawdowns mayores; su efecto se midió en `docs/hallazgo_calmar.md` y `results/calmar_audit.csv`.
- La versión final de la estrategia se definió después de ver resultados de prueba de una versión
  anterior; el reporte lo declara como posible data snooping.

## Uso de asistencia de IA

Se usó Claude (Anthropic) como asistente de programación para: estructurar el repositorio,
escribir la primera versión de los módulos de `src/` y de las pruebas, generar las figuras y el
reporte. Las decisiones de diseño (indicadores, regla de confirmación, zona de tolerancia del RSI,
reglas por régimen, estimador de covarianza, protocolo de prueba) se discutieron y aprobaron por el
equipo.

Cada integrante revisó, corrió y validó el código de su parte, y es responsable de poder explicarlo:

- **Adrián Marcelo Ballesteros Herrera** — `src/data.py`, `src/backtest.py`, `src/metrics.py` y
  `tests/test_backtest.py`: descarga y auditoría de datos, contabilidad del motor (efectivo +
  posiciones), convención de stop-loss primero, regla de no apalancamiento y métricas de desempeño.
- **Erik del Castillo Román** — `src/signals.py`, `src/optimize.py`, `tests/test_signals.py` y
  `main.py`: indicadores, regla de confirmación 3 de 6, zona de tolerancia del RSI, espacio de
  parámetros, walk-forward con Optuna y θ promedio congelado.
- **Jesús Emmanuel Flores Cortés** — `src/regimes.py`, `src/portfolio.py`, `src/plots.py`,
  `tests/test_regimes.py`, `tests/test_portfolio.py`, `notebooks/analysis.ipynb` y `docs/`:
  clasificadores de régimen (reglas, K-means y HMM filtrado), Risk Parity, agregación de señales,
  figuras, reporte y presentación.
