"""Day trading - ricerca: strategie di breakout intraday su SPY e QQQ, barre da 1 minuto, 2016-oggi.

Protocollo:
- Parametri dei paper, non ottimizzati. Le varianti sono robustezza, non selezione.
- I paper sono recenti e i loro dati si sovrappongono ai nostri: separo il periodo PRIMA della pubblicazione
  (dove gli autori potrebbero aver scelto le regole guardando i dati) da quello DOPO, l'unico fuori campione vero.
- Costi: 0.5 bps per esecuzione + commissioni regolamentari. Half-spread NBBO misurato (spreads.py, 90 istanti,
  fine settembre - inizio ottobre 2026): SPY 0.13 bps, QQQ 0.17 bps; 0.5 bps ~ 3 volte lo spread, per coprire
  impatto e ritardo. Il risultato dipende molto da questa ipotesi: per questo stampo la sensibilità ai costi.

Uso: python -m daytrade.research
"""
import numpy as np
import pandas as pd

import backtest as bt
from daytrade import engine as en
from daytrade.strategies import NoiseBoundary, OpeningRangeBreakout

START = "2016-01-01"
PUBLISHED = {"noise": "2024-06-01", "orb": "2023-06-01"}     # date approssimative di uscita dei paper (SSRN)
COSTS = en.Costs(slip_bps=0.5)
SLIPS = (0.0, 0.25, 0.5, 1.0)


def summary(daily, trades=None):
    m = bt.metrics(daily)
    out = {k: m[k] for k in ("cagr_%", "excess_netto_%", "vol_%", "sharpe_lordo", "sharpe_netto", "t_stat",
                             "max_dd_%", "costo_pareggio_bps")}
    if trades is not None:
        st = en.trade_stats(trades, daily)
        out.update({k: st.get(k, np.nan) for k in ("trade_al_giorno", "vincenti_%", "profit_factor")})
    return out


def by_period(res, published):
    d, t = res.daily, res.trades
    cut = pd.Timestamp(published)
    rows = {"tutto": summary(d, t),
            "prima della pubblicazione": summary(d[d.index < cut], t[t.date < cut] if len(t) else t),
            "dopo la pubblicazione": summary(d[d.index >= cut], t[t.date >= cut] if len(t) else t)}
    return pd.DataFrame(rows).T


def cost_sensitivity(df, cls, published):
    rows = {}
    for slip in SLIPS:
        d = en.run(df, cls(), costs=en.Costs(slip_bps=slip)).daily
        post = d[d.index >= pd.Timestamp(published)]
        rows[f"{slip} bps"] = {"sharpe_tutto": bt.metrics(d)["sharpe_netto"], "t_tutto": bt.metrics(d)["t_stat"],
                               "sharpe_dopo_pub": bt.metrics(post)["sharpe_netto"],
                               "t_dopo_pub": bt.metrics(post)["t_stat"]}
    return pd.DataFrame(rows).T


def diagnostics(res, df):
    """Controlli da risk manager: dipendenza dal mercato, dai giorni estremi, stabilità nel tempo, lati."""
    d, t = res.daily, res.trades
    g = df.groupby(df.index.normalize().tz_localize(None))
    bh = g.close.last().pct_change().reindex(d.index)
    logr = np.log1p(d.net)
    top = logr.nlargest(10).sum()
    yearly_sharpe = d.net.groupby(d.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
    return {
        "correlazione con buy & hold": round(d.net.corr(bh), 2),
        "bps medi nei giorni di mercato su / giù": (round(1e4 * d.net[bh > 0].mean(), 1),
                                                    round(1e4 * d.net[bh < 0].mean(), 1)),
        "% del log-rendimento dai 10 giorni migliori": round(100 * top / logr.sum()),
        "CAGR % senza i 10 giorni migliori": round(100 * (np.exp((logr.sum() - top) / (len(d) / 252)) - 1), 1),
        "anni con Sharpe > 0": f"{(yearly_sharpe > 0).sum()} su {len(yearly_sharpe)}",
        "PnL lordo long / short ($)": tuple(t.groupby("side").gross_pnl.sum().reindex([1, -1]).round(0)),
    }


def buy_and_hold(df):
    """Benchmark: buy & hold (chiusura-chiusura, overnight compreso) e solo-intraday (apertura-chiusura)."""
    g = df.groupby(df.index.normalize().tz_localize(None))
    close, open_ = g.close.last(), g.open.first()
    out = {}
    for name, r in (("buy & hold", close.pct_change().dropna()), ("long apertura-chiusura", close / open_ - 1)):
        daily = pd.DataFrame({"gross": r, "net": r, "rf": 0.0, "borrow": 0.0, "turnover": 0.0, "trade": 0})
        out[name] = summary(daily)
    return pd.DataFrame(out).T


def robustness(df, make, grid, published):
    rows = {}
    for name, kw in grid.items():
        res = en.run(df, make(**kw), costs=COSTS)
        d = res.daily[res.daily.index >= pd.Timestamp(published)]
        rows[name] = {"sharpe_netto_tutto": bt.metrics(res.daily)["sharpe_netto"],
                      "sharpe_netto_dopo_pub": bt.metrics(d)["sharpe_netto"],
                      "cagr_%_tutto": bt.metrics(res.daily)["cagr_%"]}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from data import get_bars
    bars = get_bars(["SPY", "QQQ"], minutes=1, start=START)
    data = {s: bars.loc[s] for s in ("SPY", "QQQ")}
    pd.set_option("display.width", 220)
    print(f"Barre da 1 minuto dal {bars.index.get_level_values(1).min():%Y-%m-%d} al "
          f"{bars.index.get_level_values(1).max():%Y-%m-%d}. Costi: {COSTS}\n")

    for sym in ("SPY", "QQQ"):
        print(f"=== Benchmark {sym} ===")
        print(buy_and_hold(data[sym]).round(2).to_string(), "\n")

    specs = [("Noise boundary", "noise", NoiseBoundary, "SPY"), ("Noise boundary", "noise", NoiseBoundary, "QQQ"),
             ("Opening range breakout 5m", "orb", OpeningRangeBreakout, "QQQ"),
             ("Opening range breakout 5m", "orb", OpeningRangeBreakout, "SPY")]
    for name, key, cls, sym in specs:
        res = en.run(data[sym], cls(), costs=COSTS)
        print(f"=== {name} su {sym} (pubblicazione ~{PUBLISHED[key]}) ===")
        print(by_period(res, PUBLISHED[key]).round(2).to_string())
        if len(res.trades):
            print("Uscite:", res.trades.reason.value_counts().to_dict())
        yearly = res.daily.net.groupby(res.daily.index.year).apply(lambda r: 100 * ((1 + r).prod() - 1))
        print("Rendimento netto per anno (%):", yearly.round(1).to_dict(), "\n")

    for name, cls, sym, key in [("Noise boundary", NoiseBoundary, "SPY", "noise"),
                                ("Noise boundary", NoiseBoundary, "QQQ", "noise"),
                                ("Opening range breakout 5m", OpeningRangeBreakout, "QQQ", "orb")]:
        print(f"=== Sensibilità ai costi per esecuzione: {name} su {sym} ===")
        print(cost_sensitivity(data[sym], cls, PUBLISHED[key]).round(2).to_string(), "\n")

    print("=== Diagnostica di rischio: Noise boundary su QQQ ===")
    for k, v in diagnostics(en.run(data["QQQ"], NoiseBoundary(), costs=COSTS), data["QQQ"]).items():
        print(f"{k}: {v}")
    print()

    print("=== Robustezza: Noise boundary su SPY ===")
    grid = {"base (14g, banda 1.0, ogni 30m)": {}, "banda 0.8": {"band": 0.8}, "banda 1.2": {"band": 1.2},
            "decisioni ogni 15m": {"every": 15}, "decisioni ogni 60m": {"every": 60},
            "lookback 10g": {"lookback": 10}, "lookback 20g": {"lookback": 20}}
    print(robustness(data["SPY"], NoiseBoundary, grid, PUBLISHED["noise"]).round(2).to_string(), "\n")

    print("=== Robustezza: Opening range breakout su QQQ ===")
    grid = {"base (5m, target 10R)": {}, "range 15m": {"minutes": 15}, "range 30m": {"minutes": 30},
            "target 3R": {"target_r": 3.0}, "senza target": {"target_r": 1e9}}
    print(robustness(data["QQQ"], OpeningRangeBreakout, grid, PUBLISHED["orb"]).round(2).to_string())
