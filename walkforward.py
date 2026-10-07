"""Fase 1 - Walk-forward: la strategia di pairs ricalibrata a finestre mobili, valutata tutta out-of-sample.

Ogni TRADING_DAYS giorni di borsa: cointegrazione e hedge ratio sui FORMATION_DAYS giorni precedenti, poi si
tratta per i TRADING_DAYS successivi un portafoglio equipesato di TUTTE le coppie che passano il filtro (nessuna
scelta per Sharpe = nessuna distorsione da selezione). Le finestre di trading concatenate formano un'unica serie
out-of-sample, con molti più trade di un singolo split.

Ipotesi nulla: stesse regole su tutte le coppie, senza filtro di cointegrazione. Se le coppie cointegrate non
battono coppie qualsiasi, il filtro non aggiunge informazione.
"""
import itertools

import numpy as np
import pandas as pd

import research_pairs as rp

DAYS = 365
FORMATION_DAYS, TRADING_DAYS = 60, 10
P_MAX = 0.05


def windows(index, formation_days=FORMATION_DAYS, trading_days=TRADING_DAYS):
    """Lista di (inizio formazione, inizio trading, fine trading), allineate ai giorni di borsa nei dati."""
    days = index.normalize().unique()
    end = index[-1] + pd.Timedelta(minutes=rp.MINUTES)
    out = []
    for i in range(formation_days, len(days), trading_days):
        stop = days[i + trading_days] if i + trading_days < len(days) else end
        out.append((days[i - formation_days], days[i], stop))
    return out


def portfolio(closes, chosen, start, end):
    """Rendimento per barra di un portafoglio equipesato di coppie (capitale diviso in parti uguali)."""
    idx = closes.index[(closes.index >= start) & (closes.index < end)]
    if not chosen:
        return pd.DataFrame({"gross": 0.0, "net": 0.0}, index=idx), 0
    runs = [rp.pnl(closes[a], closes[b], beta, start, end) for a, b, beta in chosen]
    trades = sum(int(((r.pos != 0) & (r.pos.shift(1).fillna(0) == 0)).sum()) for r in runs)
    gross = pd.concat([r.gross for r in runs], axis=1).mean(axis=1)
    net = pd.concat([r.net for r in runs], axis=1).mean(axis=1)
    return pd.DataFrame({"gross": gross, "net": net}), trades


def walk_forward(closes, p_max=P_MAX, **win):
    """Ritorna {"coint": ..., "tutte": ...}, ciascuno con la serie out-of-sample e il dettaglio per finestra."""
    pairs = list(itertools.combinations(closes.columns, 2))
    out = {k: {"series": [], "rows": []} for k in ("coint", "tutte")}
    for f0, t0, t1 in windows(closes.index, **win):
        form = np.log(closes[(closes.index >= f0) & (closes.index < t0)])
        candidates = []
        for a, b in pairs:
            beta = rp.hedge_ratio(form[a], form[b])
            if beta <= 0:                   # un pairs trade compra un titolo e vende l'altro
                continue
            candidates.append((a, b, beta, rp.coint_pvalue(form[a], form[b])))
        selections = {"coint": [c[:3] for c in candidates if c[3] <= p_max],
                      "tutte": [c[:3] for c in candidates]}
        for k, chosen in selections.items():
            r, trades = portfolio(closes, chosen, t0, t1)
            out[k]["series"].append(r)
            out[k]["rows"].append({"trading_da": t0.date(), "coppie": len(chosen), "trade": trades,
                                   "ret_lordo_%": 100 * r.gross.sum(), "ret_%": 100 * r.net.sum()})
    for v in out.values():
        v["series"] = pd.concat(v["series"])
        v["rows"] = pd.DataFrame(v["rows"])
    return out


def summary(series, rows):
    net, gross = series.net, series.gross
    daily = net.groupby(net.index.date).sum()
    equity = net.cumsum()
    return {
        "sharpe": net.mean() / net.std() * np.sqrt(rp.BARS_PER_YEAR) if net.std() > 0 else 0.0,
        "t_stat_giornaliero": daily.mean() / daily.std() * np.sqrt(len(daily)) if daily.std() > 0 else 0.0,
        "ret_%": 100 * net.sum(),
        "ret_lordo_%": 100 * gross.sum(),
        "max_dd_%": 100 * (equity - equity.cummax()).min(),
        # net = gross - turnover * costo  =>  costo per lato che azzera il rendimento netto
        "costo_pareggio_bps": rp.COST_BPS * gross.sum() / (gross.sum() - net.sum()) if gross.sum() > net.sum() else np.nan,
        "trade": int(rows.trade.sum()),
        "giorni_oos": len(daily),
    }


if __name__ == "__main__":
    from data import get_closes
    closes = get_closes(rp.UNIVERSE, rp.MINUTES, DAYS)
    n_pairs = len(rp.UNIVERSE) * (len(rp.UNIVERSE) - 1) // 2
    res = walk_forward(closes)
    pd.set_option("display.width", 160)
    print(f"Dati dal {closes.index[0]:%Y-%m-%d} al {closes.index[-1]:%Y-%m-%d}. Formazione {FORMATION_DAYS} giorni, "
          f"trading {TRADING_DAYS}. Coppie cointegrate attese per caso per finestra: ~{P_MAX * n_pairs:.0f}\n")
    print("Per finestra (solo coppie cointegrate):")
    print(res["coint"]["rows"].round(2).to_string(index=False))
    print("\nOUT-OF-SAMPLE aggregato:")
    print(pd.DataFrame({k: summary(v["series"], v["rows"]) for k, v in res.items()}).round(2).to_string())
