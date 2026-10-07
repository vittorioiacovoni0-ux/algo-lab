"""Fase 1 - Ricerca: cerca coppie cointegrate sul train, sceglie sulla validation, misura sul test."""
import itertools
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from strategy import leg_weights, must_be_flat, positions, zscore

UNIVERSE = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "C", "WFC",
            "V", "MA", "HD", "LOW", "MCD", "YUM", "UPS", "FDX"]
MINUTES = 15
DAYS = 180
TRAIN_FRAC, VALID_FRAC = 0.5, 0.25          # il resto (25%) è il test: lo guardo solo per la coppia scelta
WINDOW, Z_IN, Z_OUT, Z_STOP = 100, 2.0, 0.5, 4.0
COST_BPS = 3.0                              # costo per unità di turnover (spread + slippage), sii pessimista
BARS_PER_YEAR = (390 // MINUTES) * 252


def hedge_ratio(log_a, log_b):
    return sm.OLS(log_a, sm.add_constant(log_b)).fit().params.iloc[1]


def coint_pvalue(log_a, log_b):
    """Engle-Granger dipende da quale serie fa da dipendente: tengo il p-value peggiore delle due direzioni."""
    return max(coint(log_a, log_b)[1], coint(log_b, log_a)[1])


def session_flat(index, minutes):
    """Barre in cui la strategia deve essere flat. La chiusura di sessione è la fine dell'ultima barra del
    giorno: è nota in anticipo dal calendario di borsa (anche nelle mezze giornate), quindi non è lookahead."""
    bar_close = pd.Series(index + pd.Timedelta(minutes=minutes), index=index)
    session_close = bar_close.groupby(index.date).transform("max")
    return must_be_flat(bar_close, session_close, minutes).values


def pnl(pa, pb, beta, start, end):
    """Serie per barra in [start, end): rendimento lordo e netto (su esposizione lorda 1) e stato."""
    spread = np.log(pa) - beta * np.log(pb)
    z = zscore(spread, WINDOW)              # rolling solo sul passato: usabile anche a cavallo degli split
    keep = (z.index >= start) & (z.index < end)
    z, pa, pb = z[keep], pa[keep], pb[keep]
    pos = positions(z, Z_IN, Z_OUT, Z_STOP, flat=session_flat(z.index, MINUTES))
    # come nel live: le quantità di azioni si fissano al cambio di stato e restano costanti fino al successivo
    # (esposizione lorda = 1 all'ingresso, poi i pesi in dollari derivano con i prezzi)
    wa, wb = leg_weights(beta)
    change = pos.diff().fillna(pos) != 0
    qa = (pos * wa / pa.where(change)).ffill().fillna(0)
    qb = (pos * wb / pb.where(change)).ffill().fillna(0)
    gross = (qa.shift(1) * pa.diff() + qb.shift(1) * pb.diff()).fillna(0)   # decido a fine barra t, PnL da t+1
    turnover = qa.diff().fillna(qa).abs() * pa + qb.diff().fillna(qb).abs() * pb
    net = gross - turnover * COST_BPS / 1e4
    return pd.DataFrame({"gross": gross, "net": net, "pos": pos})


def metrics(gross, net, pos):
    equity = net.cumsum()
    return {
        "sharpe": net.mean() / net.std() * np.sqrt(BARS_PER_YEAR) if net.std() > 0 else 0.0,
        "ret_%": 100 * net.sum(),
        "ret_lordo_%": 100 * gross.sum(),
        "max_dd_%": 100 * (equity - equity.cummax()).min(),
        "trade": int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum()),
        "in_mercato_%": 100 * (pos != 0).mean(),
    }


def backtest(pa, pb, beta, start, end):
    r = pnl(pa, pb, beta, start, end)
    return metrics(r.gross, r.net, r.pos)


def run(closes):
    """Ritorna (tabella delle coppie cointegrate con metriche di validation, coppia scelta, metriche di test)."""
    n = len(closes)
    valid_start = closes.index[int(n * TRAIN_FRAC)]
    test_start = closes.index[int(n * (TRAIN_FRAC + VALID_FRAC))]
    end = closes.index[-1] + pd.Timedelta(minutes=MINUTES)
    train = np.log(closes[closes.index < valid_start])
    pairs = list(itertools.combinations(closes.columns, 2))
    rows = []
    for a, b in pairs:
        pval = coint_pvalue(train[a], train[b])
        if pval > 0.05:
            continue
        beta = hedge_ratio(train[a], train[b])
        rows.append({"a": a, "b": b, "beta": round(beta, 4), "pvalue": round(pval, 4),
                     **backtest(closes[a], closes[b], beta, valid_start, test_start)})
    print(f"Train fino a {valid_start:%Y-%m-%d}, validation fino a {test_start:%Y-%m-%d}, test dopo. "
          f"Coppie testate: {len(pairs)}, cointegrate al 5%: {len(rows)} "
          f"(per puro caso ne attendi ~{0.05 * len(pairs):.0f})\n")
    if not rows:
        return pd.DataFrame(), None, None
    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    best = res.iloc[0]
    test = backtest(closes[best.a], closes[best.b], best.beta, test_start, end)
    return res, best, test


if __name__ == "__main__":
    from data import get_closes
    raw = get_closes(UNIVERSE, MINUTES, DAYS, fill=False)
    stale = 100 * raw.isna().mean()
    print("Barre senza scambi su IEX (prezzo riempito con l'ultimo noto), %:")
    print(stale.round(1).sort_values(ascending=False).to_string(), "\n")
    closes = raw.ffill().dropna()
    res, best, test = run(closes)
    if res.empty:
        print("Nessuna coppia cointegrata: allarga l'universo o il periodo.")
    else:
        pd.set_option("display.width", 160)
        print("VALIDATION (serve solo a scegliere la coppia, lo Sharpe del primo è ottimistico):")
        print(res.round(2).to_string(index=False))
        print(f"\nTEST out-of-sample di {best.a}/{best.b}, mai usato per scegliere:")
        print(pd.Series(test).round(2).to_string())
        cfg = {"a": best.a, "b": best.b, "beta": float(best.beta), "minutes": MINUTES,
               "window": WINDOW, "z_in": Z_IN, "z_out": Z_OUT, "z_stop": Z_STOP}
        # il bot live legge pair_config.json: cambiarlo con posizioni aperte lascerebbe orfane le gambe vecchie
        if "--save" in sys.argv:
            with open("pair_config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"\nSalvata in pair_config.json: {best.a}/{best.b}. "
                  "Vai in paper solo se il risultato sul TEST ha senso.")
        else:
            print(f"\nConfigurazione proposta (NON salvata): {json.dumps(cfg)}\n"
                  "Per sovrascrivere pair_config.json: python research_pairs.py --save "
                  "(solo con il bot fermo e senza posizioni aperte).")
