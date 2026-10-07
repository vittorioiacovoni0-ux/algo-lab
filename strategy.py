"""Logica della strategia pairs trading, condivisa tra ricerca e live (stesso codice = niente discrepanze)."""
import numpy as np
import pandas as pd


def leg_weights(beta):
    """Pesi in dollari delle due gambe, esposizione lorda = 1."""
    gross = 1 + abs(beta)
    return 1 / gross, -beta / gross


def zscore(spread, window):
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def next_state(state, z, z_in, z_out, z_stop):
    """Macchina a stati: 0 flat, +1 long spread, -1 short spread. Non inverte mai in un solo passo."""
    if np.isnan(z):
        return state
    if state == 0:
        if z_in < z < z_stop:
            return -1
        if -z_stop < z < -z_in:
            return 1
        return 0
    if abs(z) >= z_stop:                       # stop: lo spread si sta allargando, la relazione forse è rotta
        return 0
    if state == 1 and z > -z_out:
        return 0
    if state == -1 and z < z_out:
        return 0
    return state


def must_be_flat(bar_close, session_close, minutes):
    """Strategia intraday: flat dalla barra che chiude a meno di una barra dalla fine della sessione.
    Così il live chiude a mercato ancora aperto e nessuna posizione resta esposta al gap overnight."""
    return bar_close >= session_close - pd.Timedelta(minutes=minutes)


def positions(z, z_in, z_out, z_stop, flat=None):
    """Stato a fine di ogni barra. `flat` (opzionale, booleana): barre in cui bisogna essere flat."""
    flat = np.zeros(len(z), dtype=bool) if flat is None else np.asarray(flat, dtype=bool)
    out, s = [], 0
    for v, f in zip(z.values, flat):
        s = 0 if f else next_state(s, v, z_in, z_out, z_stop)
        out.append(s)
    return pd.Series(out, index=z.index, dtype=float)
