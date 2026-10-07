"""Fase 1 - Ricerca: cerca coppie cointegrate sul train, le valuta OUT-OF-SAMPLE con costi."""
import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from strategy import leg_weights, positions, zscore

UNIVERSE = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "C", "WFC",
            "V", "MA", "HD", "LOW", "MCD", "YUM", "UPS", "FDX"]
MINUTES = 15
DAYS = 180
TRAIN_FRAC = 0.6
WINDOW, Z_IN, Z_OUT, Z_STOP = 100, 2.0, 0.5, 4.0
COST_BPS = 3.0                              # costo per unità di turnover (spread + slippage), sii pessimista
BARS_PER_YEAR = (390 // MINUTES) * 252


def hedge_ratio(log_a, log_b):
    return sm.OLS(log_a, sm.add_constant(log_b)).fit().params.iloc[1]


def backtest(pa, pb, beta, test_start):
    spread = np.log(pa) - beta * np.log(pb)
    z = zscore(spread, WINDOW)              # rolling solo sul passato: usabile anche a cavallo del train
    z = z[z.index >= test_start]
    pos = positions(z, Z_IN, Z_OUT, Z_STOP)
    wa, wb = leg_weights(beta)
    ret = wa * pa.pct_change() + wb * pb.pct_change()
    ret = ret.loc[pos.index]
    gross = pos.shift(1).fillna(0) * ret    # shift: decido a fine barra t, guadagno da t+1
    cost = pos.diff().abs().fillna(0) * COST_BPS / 1e4
    net = gross - cost
    equity = net.cumsum()
    return {
        "sharpe": net.mean() / net.std() * np.sqrt(BARS_PER_YEAR) if net.std() > 0 else 0.0,
        "ret_%": 100 * net.sum(),
        "ret_lordo_%": 100 * gross.sum(),
        "max_dd_%": 100 * (equity - equity.cummax()).min(),
        "trade": int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum()),
        "in_mercato_%": 100 * (pos != 0).mean(),
    }


def run(closes):
    split = closes.index[int(len(closes) * TRAIN_FRAC)]
    train = np.log(closes[closes.index < split])
    rows = []
    for a, b in itertools.combinations(closes.columns, 2):
        pval = coint(train[a], train[b])[1]
        if pval > 0.05:
            continue
        beta = hedge_ratio(train[a], train[b])
        rows.append({"a": a, "b": b, "beta": round(beta, 4), "pvalue": round(pval, 4),
                     **backtest(closes[a], closes[b], beta, split)})
    n_pairs = len(list(itertools.combinations(closes.columns, 2)))
    print(f"Train fino a {split:%Y-%m-%d}. Coppie testate: {n_pairs}, "
          f"cointegrate al 5%: {len(rows)} (per puro caso ne attendi ~{0.05 * n_pairs:.0f})\n")
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False) if rows else pd.DataFrame()


if __name__ == "__main__":
    from data import get_closes
    closes = get_closes(UNIVERSE, MINUTES, DAYS)
    res = run(closes)
    if res.empty:
        print("Nessuna coppia cointegrata: allarga l'universo o il periodo.")
    else:
        pd.set_option("display.width", 160)
        print(res.round(2).to_string(index=False))
        best = res.iloc[0]
        cfg = {"a": best.a, "b": best.b, "beta": float(best.beta), "minutes": MINUTES,
               "window": WINDOW, "z_in": Z_IN, "z_out": Z_OUT, "z_stop": Z_STOP}
        json.dump(cfg, open("pair_config.json", "w"), indent=2)
        print(f"\nSalvata in pair_config.json: {best.a}/{best.b}. "
              "Controlla che lo Sharpe out-of-sample abbia senso PRIMA di andare in paper.")
