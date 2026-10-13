"""Strategie di day trading. Le stesse classi girano nel backtest (daytrade.engine) e, poi, nel bot live.

Regole riprese dai paper come le ricordo: prima di trarre conclusioni vanno confrontate con il testo originale,
e ogni scostamento va dichiarato.
"""
import warnings
from collections import deque

import numpy as np

from daytrade.engine import Intent, Strategy

OPEN_MIN = 9 * 60 + 30          # 570: inizio della barra delle 9:30


class NoiseBoundary(Strategy):
    """Intraday momentum con "area di rumore" (Zarattini, Aziz, Barbon 2024, "Beat the Market", su SPY).

    - Rumore per ora del giorno: sigma(t) = media degli ultimi `lookback` giorni di |prezzo alle t / apertura - 1|.
    - Confini: sopra = max(apertura, chiusura di ieri) * (1 + band * sigma(t)), sotto = min(...) * (1 - band * sigma(t)).
      Usare anche la chiusura di ieri evita di scambiare il gap di apertura per un breakout.
    - Decisioni solo ogni `every` minuti (10:00, 10:30, ... 15:30), per non tradare il rumore di ogni minuto.
      Flat: long sopra il confine superiore, short sotto quello inferiore.
      Long: esce se il prezzo torna sotto max(confine superiore, VWAP) (stop mobile), si inverte sotto il confine
      inferiore. Short simmetrico.
    - Esposizione = target di volatilità giornaliera / volatilità realizzata degli ultimi `lookback` giorni,
      massimo `max_lev`: si rischia lo stesso ogni giorno, a prescindere dal regime.
    """

    def __init__(self, lookback=14, band=1.0, every=30, first="10:00", last="15:30", target_vol=0.02,
                 max_lev=4.0):
        self.lookback, self.band, self.every = lookback, band, every
        self.first = int(first[:2]) * 60 + int(first[3:])
        self.last = int(last[:2]) * 60 + int(last[3:])
        self.target_vol, self.max_lev = target_vol, max_lev
        self.moves = deque(maxlen=lookback)         # per giorno: |close/open - 1| per minuto di fine barra
        self.closes = deque(maxlen=lookback + 1)    # chiusure giornaliere, per la volatilità
        self.seen = 0

    def start_day(self, bars, history):
        # aggiorna le statistiche con i giorni non ancora visti (solo giorni PRECEDENTI a oggi)
        for d in history[self.seen:]:
            mv = np.full(24 * 60, np.nan)
            mv[d.minute + 1] = np.abs(d.close / d.open[0] - 1)          # indice = minuto di FINE barra
            self.moves.append(mv)
            self.closes.append(d.close[-1])
        self.seen = len(history)
        ready = len(self.moves) == self.lookback and len(self.closes) >= 2
        with warnings.catch_warnings():             # minuti fuori sessione: tutti NaN, media NaN voluta
            warnings.simplefilter("ignore", RuntimeWarning)
            self.sigma = np.nanmean(np.vstack(self.moves), axis=0) if ready else None
        if ready:
            rets = np.diff(np.log(np.asarray(self.closes)))
            vol = rets.std(ddof=1) if len(rets) > 1 else np.nan
            self.lev = min(self.max_lev, self.target_vol / vol) if vol and vol > 0 else 0.0
            self.prev_close = self.closes[-1]

    def on_bar(self, bars, side):
        end = bars.minute[-1] + 1
        if self.sigma is None or end < self.first or end > self.last or (end - OPEN_MIN) % self.every:
            return None
        if bars.minute[0] != OPEN_MIN:              # manca la barra di apertura: giorno anomalo, non opero
            return None
        s = self.sigma[end]
        if np.isnan(s):
            return None
        px, day_open = bars.close[-1], bars.open[0]
        upper = max(day_open, self.prev_close) * (1 + self.band * s)
        lower = min(day_open, self.prev_close) * (1 - self.band * s)
        typical = (bars.high + bars.low + bars.close) / 3
        vwap = (typical * bars.volume).sum() / bars.volume.sum() if bars.volume.sum() > 0 else px
        if side == 1:                               # stop mobile del long: max(confine superiore, VWAP)
            want = -1 if px < lower else (0 if px < max(upper, vwap) else 1)
        elif side == -1:                            # stop mobile dello short: min(confine inferiore, VWAP)
            want = 1 if px > upper else (0 if px > min(lower, vwap) else -1)
        else:
            want = 1 if px > upper else (-1 if px < lower else 0)
        return Intent(want, self.lev) if want != side else None


class OpeningRangeBreakout(Strategy):
    """Opening range breakout a 5 minuti (Zarattini, Aziz 2023, "Can Day Trading Really Be Profitable?", su QQQ).

    - La prima candela da `minutes` minuti decide la direzione: rialzista -> long, ribassista -> short, nulla -> niente.
    - Ingresso all'apertura della barra successiva. Stop all'estremo opposto della candela (minimo per un long).
    - Target a `target_r` volte il rischio (R = distanza ingresso-stop); altrimenti uscita a fine giornata.
    - Dimensione: si rischia `risk` dell'equity se scatta lo stop, con leva massima `max_lev`.
    Un solo trade al giorno.
    """

    def __init__(self, minutes=5, target_r=10.0, risk=0.01, max_lev=4.0):
        self.minutes, self.target_r, self.risk, self.max_lev = minutes, target_r, risk, max_lev

    def on_bar(self, bars, side):
        if len(bars.close) != self.minutes or bars.minute[0] != OPEN_MIN \
                or bars.minute[-1] != OPEN_MIN + self.minutes - 1:
            return None
        o, c = bars.open[0], bars.close[-1]
        hi, lo = bars.high.max(), bars.low.min()
        if c > o:
            d, stop = 1, lo
        elif c < o:
            d, stop = -1, hi
        else:
            return None
        r = abs(c - stop)                           # rischio per azione, stimato sull'ultima chiusura
        if r <= 0:
            return None
        lev = min(self.max_lev, self.risk / (r / c))
        return Intent(d, lev, stop=stop, take=c + d * self.target_r * r)
