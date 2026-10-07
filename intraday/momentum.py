"""Fase 3 - Intraday momentum (Gao, Han, Li, Zhou 2018, "Market intraday momentum", JFE).

Il rendimento della prima mezz'ora (dalla chiusura di ieri alle 10:00, overnight compreso) predice il
rendimento dell'ultima mezz'ora (15:30-16:00). Spiegazione proposta: chi deve ribilanciare (hedging di gamma
dei dealer, ETF a leva, fondi) e i trader informati che arrivano tardi tradano verso la chiusura nella stessa
direzione delle notizie della mattina.

Tutti i prezzi sono chiusure di barre da 5 minuti (indice = inizio barra, ora di New York): il prezzo "alle
10:00" è la chiusura della barra che inizia alle 09:55.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

BAR_MIN = 5


def price_at(closes, hhmm):
    """Prezzo all'orario hhmm di ogni giorno = chiusura della barra che finisce a quell'ora."""
    start = (pd.Timestamp(hhmm) - pd.Timedelta(minutes=BAR_MIN)).time()
    s = closes[closes.index.time == start]
    return pd.Series(s.values, index=s.index.normalize().tz_localize(None))


def daily_features(closes):
    """Una riga per giorno: r1 (chiusura di ieri -> 10:00), r12 (15:00 -> 15:30), r13 (15:30 -> 16:00) e
    r13_late (15:35 -> 16:00, entrata ritardata per i segnali che usano r12). Giorni con barre mancanti o
    chiusura anticipata (es. 13:00) restano NaN e vengono scartati."""
    close_16 = price_at(closes, "16:00")
    # chiusura precedente = ultima barra del giorno di borsa precedente, qualunque ora sia (le mezze giornate
    # chiudono alle 13:00: con la barra delle 16:00 salterei un giorno e r1 includerebbe due sedute)
    last = closes.groupby(closes.index.normalize().tz_localize(None)).last()
    f = pd.DataFrame({
        "prev_close": last.shift(1),
        "p1000": price_at(closes, "10:00"),
        "p1500": price_at(closes, "15:00"),
        "p1530": price_at(closes, "15:30"),
        "p1535": price_at(closes, "15:35"),
        "p1600": close_16,
    })
    f["r1"] = f.p1000 / f.prev_close - 1
    f["r12"] = f.p1530 / f.p1500 - 1
    f["r13"] = f.p1600 / f.p1530 - 1
    f["r13_late"] = f.p1600 / f.p1535 - 1
    return f.dropna()


def signal(f, rule="r1"):
    """Posizione per l'ultima mezz'ora: +1 long, -1 short, 0 flat.
    - "r1": segno del rendimento della prima mezz'ora (noto dalle 10:00).
    - "r1_r12": solo se r1 e r12 hanno lo stesso segno (r12 è noto alle 15:30: si entra alle 15:35).
    - "long": sempre long (benchmark: separa il segnale dalla deriva media dell'ultima mezz'ora)."""
    if rule == "r1":
        return np.sign(f.r1)
    if rule == "r1_r12":
        return np.sign(f.r1).where(np.sign(f.r1) == np.sign(f.r12), 0.0)
    if rule == "long":
        return pd.Series(1.0, index=f.index)
    raise ValueError(rule)


def pnl(f, pos, cost_bps, late_entry=False):
    """Risultato giornaliero in formato backtest.metrics. Round trip ogni giorno con posizione: entrata e uscita
    (ordine market-on-close) pagano cost_bps ciascuna. Nessun prestito: lo short è chiuso in giornata."""
    r = f.r13_late if late_entry else f.r13
    gross = pos * r
    turnover = 2 * pos.abs()
    return pd.DataFrame({"gross": gross, "net": gross - turnover * cost_bps / 1e4, "rf": 0.0, "borrow": 0.0,
                         "turnover": turnover, "trade": (pos != 0).astype(int)}, index=f.index)


def predictive_regression(f, x="r1", y="r13", lags=5):
    """r13 = a + b * x + e, errori standard Newey-West (robusti ad autocorrelazione ed eteroschedasticità)."""
    fit = sm.OLS(f[y], sm.add_constant(f[x])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"beta": fit.params[x], "t_nw": fit.tvalues[x], "r2_%": 100 * fit.rsquared, "giorni": int(fit.nobs)}


def high_vol_days(f, window=20, q=2 / 3, min_periods=250):
    """Giorni ad alta volatilità, dove il paper trova l'effetto più forte. Solo dati passati: volatilità = |r1|
    medio dei `window` giorni PRECEDENTI, soglia = quantile `q` della sua storia fino al giorno prima."""
    vol = f.r1.abs().rolling(window).mean().shift(1)
    thr = vol.expanding(min_periods).quantile(q).shift(1)
    return (vol > thr).where(thr.notna())
