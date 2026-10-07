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


def positions(z, z_in, z_out, z_stop):
    out, s = [], 0
    for v in z.values:
        s = next_state(s, v, z_in, z_out, z_stop)
        out.append(s)
    return pd.Series(out, index=z.index, dtype=float)
