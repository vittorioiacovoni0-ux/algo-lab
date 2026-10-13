"""Motore di simulazione intraday a barre da 1 minuto, un solo strumento, sempre flat a fine giornata.

Convenzioni (prudenti di proposito):
- La strategia decide a FINE barra i e vede solo le barre 0..i del giorno (riceve viste tagliate degli array:
  il futuro non è raggiungibile nemmeno per errore) più i giorni precedenti.
- Un cambio di posizione viene eseguito all'APERTURA della barra i+1.
- Stop e target sono ordini fermi presso il broker, controllati sui massimi e minimi di ogni barra dopo l'ingresso.
  Se l'apertura salta oltre lo stop, l'eseguito è all'apertura (gap), non allo stop. Se nella stessa barra sono
  toccati sia stop sia target, si assume prima lo stop (non sappiamo l'ordine dentro la barra: caso peggiore).
- A fine giornata chiusura alla chiusura dell'ultima barra (ordine market-on-close). Nelle mezze giornate
  l'ultima barra è quella delle 12:59.
- Dimensionamento sull'equity di inizio giornata; leva lorda massima `max_lev` (4x = potere d'acquisto intraday
  di un conto margin USA "pattern day trader").
- Limite di perdita giornaliero: se la perdita del giorno (anche non realizzata, a fine barra) supera
  `max_daily_loss`, si chiude alla barra successiva e non si opera più fino a domani.
- Costi per esecuzione: slippage in bps sul nozionale (half-spread + impatto), commissione per azione,
  commissioni regolamentari sulle vendite (SEC fee in bps del venduto, FINRA TAF per azione).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Intent:
    """Posizione desiderata. side: +1 long, -1 short, 0 flat. leverage: esposizione / equity di inizio giornata.
    stop/take: prezzi degli ordini fermi da piazzare dopo l'ingresso (None = nessuno)."""
    side: int
    leverage: float = 1.0
    stop: float | None = None
    take: float | None = None


@dataclass
class Bars:
    """Barre di un giorno fino alla barra corrente inclusa. minute: minuto del giorno dell'INIZIO barra (570 = 9:30)."""
    date: pd.Timestamp
    minute: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def upto(self, i):
        return Bars(self.date, self.minute[:i + 1], self.open[:i + 1], self.high[:i + 1], self.low[:i + 1],
                    self.close[:i + 1], self.volume[:i + 1])


@dataclass
class Costs:
    slip_bps: float = 1.0           # per esecuzione, sul nozionale
    per_share: float = 0.0          # commissione (Alpaca: 0)
    sec_bps: float = 0.3            # SEC fee sulle vendite (tariffa variabile: verificare quella in vigore)
    taf_per_share: float = 0.000166  # FINRA TAF sulle vendite

    def of(self, shares, price):
        notional = abs(shares) * price
        c = notional * self.slip_bps / 1e4 + abs(shares) * self.per_share
        if shares < 0:              # vendita
            c += notional * self.sec_bps / 1e4 + abs(shares) * self.taf_per_share
        return c


class Strategy:
    """Interfaccia. start_day riceve i giorni precedenti (lista di Bars completi); on_bar la vista fino ad ora e
    la posizione corrente (-1/0/+1). Ritorna un Intent per cambiare qualcosa, None per non fare nulla."""

    def start_day(self, bars, history):
        pass

    def on_bar(self, bars, side):
        return None


def split_days(df):
    """DataFrame di barre da 1 minuto (indice = inizio barra, ora di New York) -> lista di Bars, uno per giorno."""
    idx = df.index
    minute = (idx.hour * 60 + idx.minute).to_numpy()
    dates = idx.normalize().tz_localize(None)
    cols = {c: df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close", "volume")}
    days = []
    bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        days.append(Bars(dates[a], minute[a:b], *(cols[c][a:b] for c in ("open", "high", "low", "close", "volume"))))
    return days


@dataclass
class Result:
    daily: pd.DataFrame
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(df, strategy, equity=100_000.0, max_lev=4.0, max_daily_loss=0.03, costs=None, compound=True):
    costs = costs or Costs()
    days = split_days(df)
    daily, trades = [], []
    for k, day in enumerate(days):
        strategy.start_day(day, days[:k])
        eq0 = equity
        shares, entry_px, entry_min, stop, take = 0, 0.0, 0, None, None
        pending, halted = None, False
        pnl, gross, cost, traded, n_trades = 0.0, 0.0, 0.0, 0.0, 0
        n = len(day.close)

        def fill(new_shares, px, minute, reason):
            nonlocal shares, entry_px, entry_min, gross, cost, traded, n_trades
            delta = new_shares - shares
            if delta == 0:
                return
            if shares != 0 and np.sign(new_shares) != np.sign(shares):      # chiusura (o inversione)
                g = shares * (px - entry_px)
                trades.append({"date": day.date, "side": int(np.sign(shares)), "shares": abs(shares),
                               "entry_min": entry_min, "entry": entry_px, "exit_min": minute, "exit": px,
                               "gross_pnl": g, "reason": reason})
                gross += g
            c = costs.of(delta, px)
            cost += c
            traded += abs(delta) * px
            if new_shares != 0 and np.sign(new_shares) != np.sign(shares):
                entry_px, entry_min = px, minute
                n_trades += 1
            shares = new_shares

        for i in range(n):
            o, h, lo, c, m = day.open[i], day.high[i], day.low[i], day.close[i], day.minute[i]
            # 1) esecuzione all'apertura della decisione presa a fine barra precedente
            if pending is not None:
                side, lev = pending.side, min(pending.leverage, max_lev)
                target = int(round(side * lev * eq0 / o)) if side else 0
                if np.sign(target) == np.sign(shares) and target != 0:
                    target = shares                                          # stessa direzione: niente resize
                fill(target, o, m, "segnale" if not halted else "limite perdita")
                stop, take = (pending.stop, pending.take) if shares else (None, None)
                pending = None
            # 2) ordini fermi durante la barra
            if shares:
                long = shares > 0
                hit_stop = stop is not None and (lo <= stop if long else h >= stop)
                hit_take = take is not None and (h >= take if long else lo <= take)
                if hit_stop:
                    px = min(stop, o) if long else max(stop, o)
                    fill(0, px, m, "stop")
                elif hit_take:
                    px = max(take, o) if long else min(take, o)
                    fill(0, px, m, "target")
                if not shares:
                    stop = take = None
            # 3) limite di perdita giornaliero sul mark-to-market a fine barra
            unreal = shares * (c - entry_px) if shares else 0.0
            if not halted and (gross + unreal - cost) / eq0 <= -max_daily_loss:
                halted = True
                pending = Intent(0) if shares else None
                continue
            # 4) decisione della strategia a fine barra (l'ultima barra non ha una barra successiva)
            if not halted and i < n - 1:
                intent = strategy.on_bar(day.upto(i), int(np.sign(shares)))
                if intent is not None:
                    if intent.side == np.sign(shares) and intent.side != 0:
                        stop, take = intent.stop, intent.take               # aggiorna solo gli ordini fermi
                    else:
                        pending = intent
        # 5) fine giornata: market-on-close
        if shares:
            fill(0, day.close[-1], day.minute[-1], "chiusura")
        pnl = gross - cost
        daily.append({"date": day.date, "gross": gross / eq0, "net": pnl / eq0, "rf": 0.0, "borrow": 0.0,
                      "turnover": traded / eq0, "trade": n_trades, "equity": eq0 + pnl})
        if compound:
            equity = eq0 + pnl
    return Result(pd.DataFrame(daily).set_index("date"), pd.DataFrame(trades))


def trade_stats(trades, daily):
    """Statistiche per trade: numero, % vincenti, guadagno medio / perdita media, profit factor (lordi)."""
    if trades.empty:
        return {"trade": 0}
    g = trades.gross_pnl
    wins, losses = g[g > 0], g[g <= 0]
    return {
        "trade": len(trades),
        "trade_al_giorno": len(trades) / len(daily),
        "vincenti_%": 100 * len(wins) / len(g),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "media_vincita/perdita": wins.mean() / -losses.mean() if len(losses) and losses.mean() < 0 else np.inf,
        "uscite": trades.reason.value_counts().to_dict(),
    }
