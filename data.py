"""Download di barre da Alpaca.
- get_closes: barre intraday recenti per la Fase 1 (feed IEX, quello che vede il bot live).
- get_bars: storico per la ricerca (feed SIP consolidato, aggiustato per split e dividendi), con cache Parquet.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import API_KEY, API_SECRET

_client = StockHistoricalDataClient(API_KEY, API_SECRET)
TZ = "America/New_York"
CACHE = Path("data/cache")


def get_closes(symbols, minutes=15, days=120, fill=True):
    """Prezzi di chiusura (colonne = simboli), solo orario regolare, solo barre COMPLETE.
    fill=False lascia NaN dove IEX non ha scambi in quella barra (prezzo che sarebbe stantio)."""
    end = datetime.now(timezone.utc)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=end - timedelta(days=days),
        end=end,
        feed=DataFeed.IEX,
    )
    df = _client.get_stock_bars(req).df            # MultiIndex (symbol, timestamp)
    closes = df["close"].unstack(level=0)
    closes.index = closes.index.tz_convert(TZ)
    last_start = (pd.Timestamp("16:00") - pd.Timedelta(minutes=minutes)).strftime("%H:%M")
    closes = closes.between_time("09:30", last_start)
    # scarta la barra in corso: usare una barra incompleta = lookahead nel live
    now = pd.Timestamp.now(tz=TZ)
    closes = closes[closes.index + pd.Timedelta(minutes=minutes) <= now]
    return closes.ffill().dropna() if fill else closes


def get_bars(symbols, minutes=None, start="2016-01-01", end=None, adjustment=Adjustment.ALL):
    """Barre storiche SIP. minutes=None: giornaliere (indice = data); altrimenti barre intraday di sola sessione
    regolare (indice = inizio barra, ora di New York). DataFrame con MultiIndex (symbol, timestamp).

    adjustment=ALL: prezzi aggiustati all'indietro per split e dividendi, quindi i rendimenti sono total return
    (per un ETF obbligazionario la cedola è gran parte del rendimento). Il fattore usa dividendi futuri, ma è
    costante tra due stacchi: i RAPPORTI di prezzo sono corretti, i LIVELLI no. Mai usare livelli di prezzo
    aggiustati come segnale (es. "prezzo > 100"): sarebbe lookahead.

    Il piano gratuito dà SIP solo oltre gli ultimi 15 minuti. Barre incomplete scartate. La cache è per giorno di
    download: con adjustment=ALL ogni nuovo dividendo cambia tutto lo storico, quindi niente append incrementali.
    """
    symbols = [symbols] if isinstance(symbols, str) else sorted(symbols)
    now = pd.Timestamp.now(tz=TZ)
    end = pd.Timestamp(end, tz=TZ) if end else now - pd.Timedelta(minutes=16)
    key = f"{symbols}|{minutes}|{start}|{end:%Y-%m-%d}|{adjustment}"
    path = CACHE / f"bars-{hashlib.sha1(key.encode()).hexdigest()[:16]}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    tf = TimeFrame.Day if minutes is None else TimeFrame(minutes, TimeFrameUnit.Minute)
    # a blocchi (simbolo x anno per l'intraday): alpaca-py crea un oggetto Python per barra, e 10 anni di barre
    # da 5 minuti in una sola richiesta saturano la memoria
    edges = [pd.Timestamp(start, tz=TZ)]
    if minutes is not None:
        edges += list(pd.date_range(edges[0], end, freq="YS", tz=TZ)[1:])
    edges.append(end)
    groups = [symbols] if minutes is None else [[s] for s in symbols]
    chunks = []
    for group in groups:
        for a, b in zip(edges[:-1], edges[1:]):
            req = StockBarsRequest(symbol_or_symbols=group, timeframe=tf, start=a.to_pydatetime(),
                                   end=b.to_pydatetime(), adjustment=adjustment, feed=DataFeed.SIP)
            bars = _client.get_stock_bars(req)
            if bars.data:
                chunks.append(bars.df.reset_index())
    df = pd.concat(chunks, ignore_index=True).drop_duplicates(["symbol", "timestamp"])
    ts = df.timestamp.dt.tz_convert(TZ)
    if minutes is None:
        df["timestamp"] = ts.dt.tz_localize(None).dt.normalize()
        # la barra di oggi è completa solo dopo la chiusura
        df = df[(df.timestamp < now.tz_localize(None).normalize()) | (now.time() >= pd.Timestamp("16:20").time())]
    else:
        df["timestamp"] = ts
        last_start = (pd.Timestamp("16:00") - pd.Timedelta(minutes=minutes)).time()
        t = ts.dt.time
        df = df[(t >= pd.Timestamp("09:30").time()) & (t <= last_start)
                & (ts + pd.Timedelta(minutes=minutes) <= end)]
    df = df.set_index(["symbol", "timestamp"]).sort_index()[["open", "high", "low", "close", "volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def wide(bars, field="close"):
    """Da formato lungo (symbol, timestamp) a tabella timestamp x simboli."""
    return bars[field].unstack(level=0)
