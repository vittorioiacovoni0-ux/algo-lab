"""Download di barre intraday da Alpaca (feed IEX, gratuito)."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from config import API_KEY, API_SECRET

_client = StockHistoricalDataClient(API_KEY, API_SECRET)
TZ = "America/New_York"


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
