"""Fase 2 - Strategie giornaliere: funzioni pure prezzi -> pesi obiettivo nelle date di ribilanciamento.

Ogni funzione riceve prezzi total return (chiusure aggiustate, colonne = ETF) e ritorna un DataFrame di pesi
con righe SOLO nelle date di decisione, calcolati con dati fino a quella data inclusa. L'esecuzione (ritardo,
costi, deriva dei pesi) è compito di backtest.run.

Parametri presi dalla letteratura e fissati a priori, non ottimizzati sui nostri dati.
"""
import numpy as np
import pandas as pd

DAYS_PER_MONTH = 21
DAYS_PER_YEAR = 252


def month_ends(index):
    """Ultimo giorno di borsa di ogni mese presente nell'indice."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def ewma_vol(returns, com=60):
    """Volatilità annualizzata ex-ante, media esponenziale dei quadrati (Moskowitz-Ooi-Pedersen: com=60 giorni)."""
    return np.sqrt(returns.pow(2).ewm(com=com, min_periods=com).mean() * DAYS_PER_YEAR)


def tsmom(prices, rf_prices, lookback_months=12, asset_vol=0.40, max_gross=2.0, com=60):
    """Time-series momentum (Moskowitz, Ooi, Pedersen 2012). Per ogni asset: segno del rendimento in eccesso sul
    cash degli ultimi `lookback_months`, posizione dimensionata per una volatilità ex-ante di `asset_vol`,
    divisa per il numero di asset. Con 40% per asset e asset poco correlati il portafoglio sta intorno al 10-15%.
    Leva lorda limitata a `max_gross` (2x = margine Reg T di un conto retail)."""
    lb = lookback_months * DAYS_PER_MONTH
    excess = prices.pct_change(lb, fill_method=None).sub(rf_prices.pct_change(lb, fill_method=None), axis=0)
    vol = ewma_vol(prices.pct_change(fill_method=None), com)
    dates = month_ends(prices.index)
    sig = np.sign(excess.loc[dates])
    w = sig * (asset_vol / vol.loc[dates]) / prices.shape[1]
    w = w.dropna(how="all").fillna(0.0)
    gross = w.abs().sum(axis=1)
    return w.mul(np.minimum(1.0, max_gross / gross.replace(0, np.nan)).fillna(1.0), axis=0)


def xsmom(prices, lookback_months=12, skip_months=1, top=3):
    """Momentum cross-sectional (Jegadeesh-Titman; sui settori Moskowitz-Grinblatt 1999): long equipesato sui
    `top` asset con il rendimento più alto da t-12 a t-1 mesi. Si salta l'ultimo mese perché a un mese i
    rendimenti tendono a invertirsi (short-term reversal). Long-only: niente short su ETF settoriali."""
    past = prices.shift(skip_months * DAYS_PER_MONTH)
    mom = past / prices.shift(lookback_months * DAYS_PER_MONTH) - 1
    dates = month_ends(prices.index)
    m = mom.loc[dates].dropna(how="any")
    rank = m.rank(axis=1, ascending=False, method="first")
    return (rank <= top).astype(float) / top


def vol_target(prices, target_vol=0.15, window=DAYS_PER_MONTH, max_lev=1.5):
    """Volatility targeting (Moreira-Muir 2017): esposizione = target / volatilità realizzata dell'ultimo mese,
    ribilanciata mensilmente, leva massima `max_lev`. Lo Sharpe non dipende dal livello del target (che scala
    solo l'esposizione); conta che la volatilità sia prevedibile e il rendimento atteso molto meno."""
    r = prices.pct_change(fill_method=None)
    vol = r.rolling(window).std() * np.sqrt(DAYS_PER_YEAR)
    dates = month_ends(prices.index)
    return (target_vol / vol.loc[dates]).clip(upper=max_lev).dropna()


def fixed_weights(index, weights):
    """Benchmark a pesi fissi ribilanciati a fine mese (es. 60/40, equipesato)."""
    dates = month_ends(index)
    return pd.DataFrame([weights] * len(dates), index=dates)
