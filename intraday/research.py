"""Fase 3 - Ricerca: intraday momentum su ETF azionari USA, 2016-oggi.

Il paper usa SPY 1993-2013: il nostro campione è interamente POST-campione e quasi tutto post-pubblicazione
(2018), quindi è un test fuori campione. Altri ETF come verifica di robustezza.

Uso: python -m intraday.research
"""
import pandas as pd

import backtest as bt
from intraday import momentum as im

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
COST_BPS = 1.0          # per lato: half-spread di SPY ~0.1-0.2 bps (1 cent su ~500$) + slippage, prudente
START = "2016-01-01"
RULES = {"r1": False, "r1_r12": True, "long": False}      # regola -> entrata ritardata alle 15:35


def evaluate(features, symbol):
    rows = {}
    for rule, late in RULES.items():
        res = im.pnl(features, im.signal(features, rule), COST_BPS, late_entry=late)
        rows[f"{symbol} {rule}"] = bt.metrics(res)
    return rows, im.pnl(features, im.signal(features, "r1"), COST_BPS)


if __name__ == "__main__":
    from data import get_bars, wide
    closes = wide(get_bars(SYMBOLS, minutes=im.BAR_MIN, start=START))
    pd.set_option("display.width", 200)
    regs, strat, series = {}, {}, {}
    for s in SYMBOLS:
        f = im.daily_features(closes[s].dropna())
        regs[f"{s} r13 ~ r1"] = im.predictive_regression(f, "r1")
        regs[f"{s} r13 ~ r12"] = im.predictive_regression(f, "r12")
        hv = im.high_vol_days(f)
        regs[f"{s} r13 ~ r1, alta vol"] = im.predictive_regression(f[hv == True], "r1")    # noqa: E712
        regs[f"{s} r13 ~ r1, bassa vol"] = im.predictive_regression(f[hv == False], "r1")  # noqa: E712
        rows, series[s] = evaluate(f, s)
        strat.update(rows)
    print(f"Dal {closes.index[0]:%Y-%m-%d} al {closes.index[-1]:%Y-%m-%d}, costi {COST_BPS} bps per lato.\n")
    print("Regressione predittiva (errori Newey-West; nel paper, su SPY 1993-2013: R2 ~1.6%, t ~ 4-5):")
    print(pd.DataFrame(regs).T.round(3).to_string(), "\n")

    cols = ["excess_lordo_%", "excess_netto_%", "vol_%", "sharpe_lordo", "sharpe_netto", "t_stat", "max_dd_%",
            "trade", "costo_pareggio_bps"]
    print("Strategie (r1: segno della prima mezz'ora; r1_r12: solo se concordi, entrata 15:35; long: sempre long):")
    print(pd.DataFrame(strat).T[cols].round(2).to_string(), "\n")

    print("Strategia r1, rendimento netto per anno (%):")
    print(pd.DataFrame({s: r.net.groupby(r.index.year).sum() * 100 for s, r in series.items()}).round(1).to_string())
