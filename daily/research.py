"""Fase 2 - Ricerca: strategie giornaliere su ETF, 2016-oggi, contro i benchmark.

Protocollo: parametri dalla letteratura fissati a priori; il campione è quasi tutto POST-pubblicazione dei paper
(TSMOM 2012, momentum settoriale 1999, vol targeting 2017), quindi è un test fuori campione delle anomalie.
Le varianti dei parametri sono analisi di robustezza, non selezione: se il risultato regge solo con una
combinazione specifica, non è un risultato.

Uso: python -m daily.research
"""
import numpy as np
import pandas as pd

import backtest as bt
from daily import strategies as st

MULTI_ASSET = {
    "azionario": ["SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ"],
    "obbligazionario": ["TLT", "IEF", "LQD", "HYG", "TIP", "EMB"],
    "materie prime": ["GLD", "SLV", "DBC", "USO"],
    "immobiliare": ["VNQ"],
    "valute": ["UUP", "FXE", "FXY"],
}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CASH = "BIL"
COST_BPS = 5.0          # per unità di turnover: half-spread 0.5-5 bps sugli ETF dell'universo + slippage, prudente
BORROW_BPS = 50.0       # prestito annuo sugli short (ETF liquidi: tipicamente 25-50 bps)
START = "2016-01-01"


def evaluate(targets, returns, rf, start):
    res = bt.run(targets, returns, rf=rf, cost_bps=COST_BPS, lag=1, borrow_bps=BORROW_BPS)
    return res.loc[start:]


def beta_corr(res, bench):
    b = bench.reindex(res.index)
    return np.cov(res.net, b)[0, 1] / b.var(), res.net.corr(b)


def strategies(prices, multi, cash):
    return {
        "TSMOM multi-asset": (st.tsmom(prices[multi], prices[cash]), multi),
        "XSMOM settori top3": (st.xsmom(prices[SECTORS]), SECTORS),
        "Vol target SPY": (st.vol_target(prices[["SPY"]]), ["SPY"]),
        "SPY buy & hold": (st.fixed_weights(prices.index, {"SPY": 1.0}), ["SPY"]),
        "60/40 SPY/IEF": (st.fixed_weights(prices.index, {"SPY": 0.6, "IEF": 0.4}), ["SPY", "IEF"]),
        "Equipesato multi-asset": (st.fixed_weights(prices.index, {s: 1 / len(multi) for s in multi}), multi),
        "Equipesato settori": (st.fixed_weights(prices.index, {s: 1 / len(SECTORS) for s in SECTORS}), SECTORS),
    }


def table(prices, returns, rf, specs, start):
    out, series = {}, {}
    for name, (targets, cols) in specs.items():
        res = evaluate(targets, returns[cols], rf, start)
        m = bt.metrics(res)
        m["beta_spy"], m["corr_spy"] = beta_corr(res, returns.SPY.sub(rf))
        out[name], series[name] = m, res
    return pd.DataFrame(out).T, series


def robustness(prices, returns, rf, multi, cash, start):
    rows = {}
    for lb in (1, 3, 6, 12):
        rows[f"TSMOM lookback {lb}m"] = (st.tsmom(prices[multi], prices[cash], lookback_months=lb), multi)
    for lb, top in ((6, 3), (12, 2), (12, 4)):
        rows[f"XSMOM {lb}-1m top{top}"] = (st.xsmom(prices[SECTORS], lookback_months=lb, top=top), SECTORS)
    rows["Vol target finestra 3m"] = (st.vol_target(prices[["SPY"]], window=63), ["SPY"])
    rows["Vol target leva max 1"] = (st.vol_target(prices[["SPY"]], max_lev=1.0), ["SPY"])
    t, _ = table(prices, returns, rf, rows, start)
    return t[["sharpe_netto", "t_stat", "cagr_%", "max_dd_%", "turnover_annuo"]]


if __name__ == "__main__":
    from data import get_bars, wide
    multi = sum(MULTI_ASSET.values(), [])
    prices = wide(get_bars(multi + SECTORS + [CASH], start=START))
    missing = prices.isna().sum()
    if missing.any():
        print("Giorni mancanti per simbolo:", missing[missing > 0].to_dict())
    returns = prices.pct_change(fill_method=None)
    rf = returns[CASH]
    start = prices.index[st.DAYS_PER_YEAR + 30]       # stesso periodo per tutti: dopo il riscaldamento di 12 mesi
    pd.set_option("display.width", 200)
    print(f"Periodo di valutazione: {start:%Y-%m-%d} - {prices.index[-1]:%Y-%m-%d}, costi {COST_BPS} bps per "
          f"unità di turnover, esecuzione alla chiusura successiva alla decisione.\n")

    specs = strategies(prices, multi, CASH)
    main, series = table(prices, returns, rf, specs, start)
    print(main.round(2).to_string(), "\n")

    print("Robustezza (non serve a scegliere: un risultato vero non dipende da una sola combinazione):")
    print(robustness(prices, returns, rf, multi, CASH, start).round(2).to_string(), "\n")

    print("Rendimento totale per anno solare, netto (%):")
    yearly = {n: (1 + r.net + r.rf).groupby(r.index.year).prod().sub(1).mul(100) for n, r in series.items()}
    print(pd.DataFrame(yearly).round(1).to_string(), "\n")

    mid = start + (prices.index[-1] - start) / 2
    print(f"Sottoperiodi (Sharpe netto), divisi a {mid:%Y-%m-%d}:")
    sub = {n: {"prima metà": bt.metrics(r.loc[:mid])["sharpe_netto"],
               "seconda metà": bt.metrics(r.loc[mid:])["sharpe_netto"]} for n, r in series.items()}
    print(pd.DataFrame(sub).T.round(2).to_string())
