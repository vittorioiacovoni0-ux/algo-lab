"""Motore di backtest su pesi di portafoglio, condiviso dalle Fasi 2 e 3.

Convenzioni:
- `targets`: pesi obiettivo (frazione del NAV, negativi = short) decisi a FINE periodo t, usando solo dati
  fino a t. Righe solo nelle date di decisione: nelle altre il portafoglio non tratta e i pesi DERIVANO con i
  prezzi (niente ribilanciamento implicito gratuito).
- `lag`: dopo quanti periodi dalla decisione si esegue. lag=1: esecuzione alla chiusura successiva alla
  decisione (prudente con dati giornalieri: non si può tradare esattamente al prezzo con cui si decide).
  lag=0: esecuzione allo stesso prezzo usato per decidere; usarlo solo se l'esecuzione è davvero posteriore
  all'informazione (es. segnale noto ore prima).
- Rendimenti in eccesso sul cash: r_p - rf = sum_i w_i (r_i - rf). Il cash (1 - sum w) e i proventi degli
  short rendono rf.
- Costi: cost_bps per unità di turnover all'esecuzione; prestito titoli annuo sulle posizioni short.
"""
import numpy as np
import pandas as pd


def run(targets, returns, rf=None, cost_bps=5.0, lag=1, borrow_bps=50.0, periods_per_year=252):
    """Ritorna un DataFrame per periodo: gross e net (rendimenti in eccesso), rf, borrow, turnover, trade."""
    returns = returns.fillna(0.0)
    cols, idx = returns.columns, returns.index
    rf = pd.Series(0.0, index=idx) if rf is None else rf.reindex(idx).fillna(0.0)
    targets = targets.reindex(columns=cols).fillna(0.0)
    # data di decisione -> posizione della barra di esecuzione
    exec_at = {}
    for d, row in targets.iterrows():
        pos = idx.searchsorted(d)
        if pos < len(idx) and idx[pos] == d and pos + lag < len(idx):
            exec_at[pos + lag] = row.values
    R, RF = returns.values, rf.values
    w = np.zeros(len(cols))
    gross, borrow, turnover, trades = (np.zeros(len(idx)) for _ in range(4))
    for t in range(len(idx)):
        excess = R[t] - RF[t]
        gross[t] = w @ excess                                   # pesi detenuti durante il periodo t
        borrow[t] = -w[w < 0].sum() * borrow_bps / 1e4 / periods_per_year
        w = w * (1 + R[t]) / (1 + RF[t] + gross[t])             # deriva fino a fine periodo
        if t in exec_at:
            delta = exec_at[t] - w
            turnover[t] = np.abs(delta).sum()
            trades[t] = (np.abs(delta) > 1e-4).sum()
            w = exec_at[t].copy()
    net = gross - turnover * cost_bps / 1e4 - borrow
    return pd.DataFrame({"gross": gross, "net": net, "rf": RF, "borrow": borrow, "turnover": turnover,
                         "trade": trades}, index=idx)


def metrics(res, periods_per_year=252):
    """Metriche annualizzate. Sharpe sui rendimenti in eccesso; CAGR e drawdown sul rendimento totale (rf + net)."""
    def ann(x):
        return x.mean() * periods_per_year

    def sharpe(x):
        return x.mean() / x.std() * np.sqrt(periods_per_year) if x.std() > 0 else 0.0

    total = (1 + res.rf + res.net).cumprod()
    years = len(res) / periods_per_year
    out = {
        "cagr_%": 100 * (total.iloc[-1] ** (1 / years) - 1),
        "excess_lordo_%": 100 * ann(res.gross),
        "excess_netto_%": 100 * ann(res.net),
        "vol_%": 100 * res.net.std() * np.sqrt(periods_per_year),
        "sharpe_lordo": sharpe(res.gross),
        "sharpe_netto": sharpe(res.net),
        "max_dd_%": 100 * (total / total.cummax() - 1).min(),
        "turnover_annuo": res.turnover.sum() / years,
        "trade": int(res.trade.sum()),
        # costo per unità di turnover che azzera il rendimento netto (al netto del prestito)
        "costo_pareggio_bps": 1e4 * (res.gross.sum() - res.borrow.sum()) / res.turnover.sum()
        if res.turnover.sum() > 0 else np.nan,
    }
    return out
