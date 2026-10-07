"""Fase 1 - Costi: half-spread quotato (NBBO, feed SIP) nel momento in cui il bot live manda gli ordini.

Un ordine a mercato paga almeno metà dello spread rispetto al mid: è il pavimento di COST_BPS. Il resto
(impatto, movimento del prezzo tra chiusura barra e invio dell'ordine) lo misura il bot live nei log dei fill.
Il piano gratuito di Alpaca dà i dati SIP storici, esclusi gli ultimi 15 minuti.
"""
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

from config import API_KEY, API_SECRET
from data import get_closes
from research_pairs import MINUTES, UNIVERSE

BAR_DELAY_S = 45                      # come in live_pairs.py (non lo importo: all'import apre config e log)

N_DAYS = 5
LOOKBACK_S = 10                       # cerco l'ultima quota nei secondi prima dell'istante di invio


def quoted_half_spreads(client, symbols, send_times):
    rows = []
    for t in send_times:
        req = StockQuotesRequest(symbol_or_symbols=symbols, start=t - pd.Timedelta(seconds=LOOKBACK_S),
                                 end=t, feed=DataFeed.SIP)
        q = client.get_stock_quotes(req).df
        if q.empty:
            continue
        last = q.groupby(level=0).tail(1).droplevel(1)
        last = last[(last.bid_price > 0) & (last.ask_price > last.bid_price)]
        mid = (last.bid_price + last.ask_price) / 2
        for sym, hs in ((last.ask_price - last.bid_price) / 2 / mid * 1e4).items():
            rows.append({"time": t, "symbol": sym, "half_spread_bps": hs})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    bars = get_closes(UNIVERSE[:1], MINUTES, days=N_DAYS * 2)
    days = bars.index.normalize().unique()[-N_DAYS:]
    bar_starts = bars.index[bars.index.normalize().isin(days)]
    closes_at = bar_starts + pd.Timedelta(minutes=MINUTES)
    # il live non opera sulla barra delle 16:00 (mercato già chiuso all'invio)
    send_times = [t + pd.Timedelta(seconds=BAR_DELAY_S) for t in closes_at if t.time() < pd.Timestamp("16:00").time()]
    client = StockHistoricalDataClient(API_KEY, API_SECRET)
    hs = quoted_half_spreads(client, UNIVERSE, send_times)
    print(f"Half-spread quotato (bps) a chiusura barra + {BAR_DELAY_S}s, ultimi {N_DAYS} giorni, "
          f"{len(send_times)} istanti:\n")
    stats = hs.groupby("symbol").half_spread_bps.describe(percentiles=[0.5, 0.9])[["mean", "50%", "90%"]]
    print(stats.sort_values("mean", ascending=False).round(2).to_string())
    print(f"\nMedia sull'universo: {hs.half_spread_bps.mean():.2f} bps per lato (COST_BPS attuale: 3)")
