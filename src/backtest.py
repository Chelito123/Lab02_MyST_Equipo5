"""Motor de backtest event-driven con costos, para uno o varios activos.

Estado explícito en cada barra: efectivo (`cash`), acciones por activo
(`shares`, negativas en cortos) y valor del portafolio
`equity = cash + Σ shares·precio`.

Entradas, decididas al cierre de la barra t y ejecutadas en t+1:
- `direction[t, j]` ∈ {-1, 0, +1}: lado que se quiere tener en el activo j
  (0 = sin posición).
- `weight[t, j]` ≥ 0: fracción del equity que debe ocupar esa posición.
- `rebalance[t]`: si es True, las posiciones que siguen abiertas se
  re-dimensionan a su peso objetivo.

Convenciones:
1. Ejecución al open de t+1. Primero se revisan los gaps: si el open ya cruzó
   el stop-loss o el take-profit, la posición sale al open.
2. Salidas por cambio de dirección (incluye dirección 0) al open de t+1.
3. Entradas nuevas al open de t+1. Stops: SL = P ∓ sl_mult·ATR_t y
   TP = P ± tp_mult·ATR_t, con el ATR de la barra de la señal.
4. Dentro de la barra: si el low o el high tocan SL y TP en la misma barra,
   se ejecuta primero el stop-loss (convención conservadora).
5. Después de un SL o TP no se vuelve a entrar del mismo lado hasta que la
   dirección deseada cambie.
6. Comisión `commission` sobre el nocional de cada ejecución (apertura,
   cierre y ajustes de rebalanceo). Sin spread, borrow ni impacto.
7. Sin apalancamiento: el nocional bruto Σ|shares·P| nunca excede
   equity / (1 + commission), así que la comisión de la entrada también cabe
   en el capital. Si los pesos piden más, se recortan los incrementos.
8. Los ajustes de rebalanceo menores a `min_rebalance_frac`·equity no se
   ejecutan (evita pagar comisión por diferencias mínimas).
"""

import numpy as np
import pandas as pd

COMMISSION = 0.00125
INITIAL_CASH = 1_000_000.0


def _as_matrix(x, index, columns) -> np.ndarray:
    """Convierte un escalar o un DataFrame a una matriz (barras x activos) alineada."""
    if isinstance(x, pd.DataFrame):
        return x.reindex(index=index, columns=columns).to_numpy(dtype=float)
    if isinstance(x, pd.Series):
        return x.reindex(index).to_numpy(dtype=float)[:, None].repeat(len(columns), axis=1)
    return np.full((len(index), len(columns)), float(x))


def run_backtest(prices: dict, direction: pd.DataFrame, weight, atr: pd.DataFrame,
                 sl_mult, tp_mult, rebalance=None, commission: float = COMMISSION,
                 initial_cash: float = INITIAL_CASH, min_rebalance_frac: float = 0.01) -> dict:
    """Simula la estrategia barra por barra.

    prices: {ticker: DataFrame con open/high/low/close}.
    direction, atr: DataFrames (fechas x tickers). weight, sl_mult, tp_mult:
    escalar, Series por fecha o DataFrame (fechas x tickers).
    rebalance: Series booleana por fecha (None = nunca).

    Regresa dict con equity (Series), cash, shares (Series final), positions
    (DataFrame de valor de mercado por activo al cierre), fills (DataFrame de
    ejecuciones) y trades (DataFrame de operaciones ida y vuelta cerradas).
    """
    index, tickers = direction.index, list(direction.columns)
    n, m = len(index), len(tickers)
    O = np.column_stack([prices[t]["open"].reindex(index).to_numpy(float) for t in tickers])
    H = np.column_stack([prices[t]["high"].reindex(index).to_numpy(float) for t in tickers])
    L = np.column_stack([prices[t]["low"].reindex(index).to_numpy(float) for t in tickers])
    C = np.column_stack([prices[t]["close"].reindex(index).to_numpy(float) for t in tickers])
    D = np.nan_to_num(direction.to_numpy(float))
    W = np.nan_to_num(_as_matrix(weight, index, tickers))
    A = _as_matrix(atr, index, tickers)
    SL = _as_matrix(sl_mult, index, tickers)
    TP = _as_matrix(tp_mult, index, tickers)
    R = np.zeros(n, bool) if rebalance is None else rebalance.reindex(index, fill_value=False).to_numpy(bool)

    cash = float(initial_cash)
    shares = np.zeros(m)
    side = np.zeros(m)              # lado de la posición abierta
    blocked = np.zeros(m)           # lado bloqueado tras SL/TP
    stop = np.full(m, np.nan)
    take = np.full(m, np.nan)
    entry_price = np.full(m, np.nan)
    entry_date = [None] * m
    pos_cashflow = np.zeros(m)      # flujo neto de la posición (incluye comisiones)
    equity = np.empty(n)
    positions = np.zeros((n, m))
    equity[0] = cash
    fills, trades = [], []

    def execute(i, j, qty, price, reason):
        nonlocal cash
        notional = abs(qty) * price
        fee = notional * commission
        cash -= qty * price + fee
        shares[j] += qty
        pos_cashflow[j] -= qty * price + fee
        fills.append((index[i], tickers[j], qty, price, notional, fee, reason))

    def close(i, j, price, reason):
        direction_closed = side[j]
        execute(i, j, -shares[j], price, reason)
        trades.append((tickers[j], "LONG" if direction_closed > 0 else "SHORT", entry_date[j], index[i],
                       entry_price[j], price, pos_cashflow[j], reason))
        shares[j] = 0.0
        side[j] = 0.0
        pos_cashflow[j] = 0.0
        if reason in ("stop_loss", "take_profit"):
            blocked[j] = direction_closed

    for i in range(1, n):
        o, h, lo, c = O[i], H[i], L[i], C[i]
        d = D[i - 1].copy()

        # 1. Gaps: el open ya cruzó el SL o el TP.
        for j in range(m):
            if side[j] > 0 and o[j] <= stop[j]:
                close(i, j, o[j], "stop_loss")
            elif side[j] > 0 and o[j] >= take[j]:
                close(i, j, o[j], "take_profit")
            elif side[j] < 0 and o[j] >= stop[j]:
                close(i, j, o[j], "stop_loss")
            elif side[j] < 0 and o[j] <= take[j]:
                close(i, j, o[j], "take_profit")

        # 2. Bloqueo tras SL/TP: se libera cuando la dirección deseada cambia.
        for j in range(m):
            if blocked[j] != 0:
                if d[j] != blocked[j]:
                    blocked[j] = 0.0
                else:
                    d[j] = 0.0

        # 3. Salidas por cambio de dirección.
        for j in range(m):
            if side[j] != 0 and d[j] != side[j]:
                close(i, j, o[j], "signal")

        # 4. Entradas y rebalanceo con el tope de no apalancamiento.
        equity_open = cash + shares @ o
        cap = equity_open / (1 + commission)
        current = np.abs(shares * o)
        target = current.copy()
        new = (d != 0) & (side == 0) & ~np.isnan(A[i - 1]) & (W[i - 1] > 0)
        target[new] = W[i - 1][new] * cap
        if R[i - 1]:
            keep = side != 0
            resize = keep & (np.abs(W[i - 1] * cap - current) >= min_rebalance_frac * equity_open)
            target[resize] = W[i - 1][resize] * cap
        decrease = np.maximum(current - target, 0.0)
        increase = np.maximum(target - current, 0.0)
        room = max(cap - np.minimum(current, target).sum(), 0.0)
        if increase.sum() > room:
            increase *= room / increase.sum()

        for j in np.flatnonzero(decrease > 0):
            if target[j] == 0:
                close(i, j, o[j], "rebalance")
            else:
                execute(i, j, -side[j] * decrease[j] / o[j], o[j], "rebalance")
        for j in np.flatnonzero(increase > 0):
            if side[j] == 0:
                side[j] = d[j]
                entry_price[j] = o[j]
                entry_date[j] = index[i]
                stop[j] = o[j] - side[j] * SL[i - 1, j] * A[i - 1, j]
                take[j] = o[j] + side[j] * TP[i - 1, j] * A[i - 1, j]
                execute(i, j, side[j] * increase[j] / o[j], o[j], "entry")
            else:
                execute(i, j, side[j] * increase[j] / o[j], o[j], "rebalance")

        # 5. Stops dentro de la barra (incluye posiciones abiertas hoy). SL primero.
        for j in range(m):
            if side[j] > 0:
                if lo[j] <= stop[j]:
                    close(i, j, stop[j], "stop_loss")
                elif h[j] >= take[j]:
                    close(i, j, take[j], "take_profit")
            elif side[j] < 0:
                if h[j] >= stop[j]:
                    close(i, j, stop[j], "stop_loss")
                elif lo[j] <= take[j]:
                    close(i, j, take[j], "take_profit")

        positions[i] = shares * c
        equity[i] = cash + positions[i].sum()

    fills_df = pd.DataFrame(fills, columns=["date", "ticker", "qty", "price", "notional", "commission", "reason"])
    trades_df = pd.DataFrame(trades, columns=["ticker", "side", "entry_date", "exit_date",
                                              "entry_price", "exit_price", "pnl", "exit_reason"])
    return {
        "equity": pd.Series(equity, index=index, name="equity"),
        "cash": cash,
        "shares": pd.Series(shares, index=tickers),
        "positions": pd.DataFrame(positions, index=index, columns=tickers),
        "fills": fills_df,
        "trades": trades_df,
    }


def single_asset_backtest(df: pd.DataFrame, sig: pd.DataFrame, params: dict, ticker: str = "asset",
                          commission: float = COMMISSION, initial_cash: float = INITIAL_CASH,
                          direction: pd.Series = None) -> dict:
    """Backtest de un activo: dirección = señal mantenida (`hold_direction`), peso = position_frac.

    sig: salida de `signals.generate_signals` (columnas signal y atr) alineada con df.
    direction: dirección ya calculada (opcional, p. ej. con señal enmascarada por régimen).
    """
    from .signals import hold_direction

    if direction is None:
        direction = pd.Series(hold_direction(sig["signal"], params["max_holding"]), index=sig.index)
    return run_backtest(
        {ticker: df}, direction.to_frame(ticker), params["position_frac"], sig[["atr"]].set_axis([ticker], axis=1),
        params["sl_mult"], params["tp_mult"], commission=commission, initial_cash=initial_cash,
    )
