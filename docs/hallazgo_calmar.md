# Hallazgo: el Calmar premia drawdowns mayores cuando todo pierde

## El defecto

La optimización maximiza

$$\text{Calmar} = \frac{R_{anual}}{\text{MDD}}$$

Si $R_{anual} > 0$, a igual retorno gana la configuración con menor drawdown, que es lo que se
busca. Si $R_{anual} < 0$, el cociente es negativo y **se acerca a 0 cuando el MDD crece**. Entre
dos configuraciones perdedoras, el optimizador puede preferir la que pierde más con un drawdown
mayor.

Ejemplo real de la corrida (MSFT, ventana 2, θ global):

| Configuración | $R_{anual}$ | MDD | Calmar |
|---|---|---|---|
| Elegida por el optimizador | −6.0% | 12.2% | **−0.50** |
| La que menos perdía | −4.5% | 5.9% | −0.76 |

La elegida pierde más y tiene el doble de drawdown, pero su Calmar es "mejor".

El defecto solo actúa cuando **todas** las configuraciones factibles de un estudio pierden. Si
alguna gana, su Calmar positivo le gana a cualquier negativo.

## Medición en este laboratorio

`optimize.audit_negative_calmar` (llamada desde `main.py`) repite, con su semilla original, cada
estudio de Optuna del walk-forward cuyo mejor Calmar fue negativo, y guarda retorno y MDD de cada
prueba. Resultados en `results/calmar_audit.csv` y en `calmar_negative_audit` de
`results/run_meta.json`.

| Medida | Valor |
|---|---|
| Estudios de Optuna factibles en el walk-forward | 2,176 |
| Estudios con mejor Calmar negativo (todo perdía) | 163 (7.5%) |
| Reproducen exactamente el valor original | 163 de 163 |
| Eligió la configuración que menos perdía | 99 de 163 |
| **Eligió una que perdía más** | **64 de 163 = 2.9% de todos los estudios** |
| Pérdida anual extra en esos 64 (mediana) | +1.4 puntos porcentuales |
| MDD extra en esos 64 (mediana) | +2.4 puntos porcentuales |
| Percentil medio del MDD del elegido entre las perdedoras | 29% |
| Correlación media Calmar–MDD entre perdedoras | −0.16 |

## Interpretación

- **El defecto existe y se observa** en 64 estudios: ahí el optimizador eligió perder más con más
  drawdown.
- **Su efecto es acotado.** En 92.5% de los estudios el mejor Calmar es positivo, y ahí el criterio
  favorece drawdowns chicos. Aun en los estudios perdedores, el elegido suele estar entre los de
  menor drawdown (percentil 29%).
- **Pesa más que en la versión anterior de la estrategia** (2 de 3, donde afectaba 0.6% de los
  estudios): con la regla 3 de 6 hay más ventanas donde ninguna configuración gana.
- **Decisión:** se conserva el Calmar como función objetivo porque el laboratorio lo exige, y se
  declara esta limitación. Una corrección posible sería usar el Calmar cuando $R \ge 0$ y
  $J = R - \text{MDD}$ cuando $R < 0$. No se aplicó para no cambiar el objetivo después de ver los
  resultados.
