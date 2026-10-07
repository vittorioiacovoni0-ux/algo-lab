"""Fase 2 - Order book L2 ricostruito dagli eventi registrati (snapshot + aggiornamenti per livello di prezzo)."""
import numpy as np
import pandas as pd
from sortedcontainers import SortedDict


class OrderBook:
    def __init__(self):
        self.bids = SortedDict()        # prezzo -> quantità; il best bid è l'ultimo
        self.asks = SortedDict()        # il best ask è il primo

    def clear(self):
        self.bids.clear()
        self.asks.clear()

    def apply(self, side, price, qty):
        """Aggiornamento L2: la quantità è il NUOVO totale al livello (0 = livello rimosso), non un delta."""
        book = self.bids if side == "bid" else self.asks
        if qty == 0:
            book.pop(price, None)
        else:
            book[price] = qty

    def top(self):
        """(best bid, quantità, best ask, quantità); NaN se un lato è vuoto."""
        bid, bq = self.bids.peekitem(-1) if self.bids else (np.nan, np.nan)
        ask, aq = self.asks.peekitem(0) if self.asks else (np.nan, np.nan)
        return bid, bq, ask, aq

    def depth(self, levels):
        """Quantità totale sui primi `levels` livelli di ciascun lato."""
        bq = sum(self.bids.peekitem(-1 - i)[1] for i in range(min(levels, len(self.bids))))
        aq = sum(self.asks.peekitem(i)[1] for i in range(min(levels, len(self.asks))))
        return bq, aq


def replay(l2, levels=5):
    """Top of book dopo ogni messaggio (stesso seq) di un prodotto. `l2`: righe del recorder ordinate per seq.
    Uno snapshot azzera il book: dopo un buco di sequenza il recorder si riconnette e ne riceve uno nuovo."""
    book, out = OrderBook(), []
    for seq, msg in l2.groupby("seq", sort=True):
        if (msg.type == "snapshot").any():
            book.clear()
        for side, price, qty in zip(msg.side.values, msg.price.values, msg.qty.values):
            book.apply(side, price, qty)
        bid, bq, ask, aq = book.top()
        dbq, daq = book.depth(levels)
        out.append((seq, msg.exch_ts.iloc[-1], msg.recv_ns.iloc[-1], bid, bq, ask, aq, dbq, daq))
    return pd.DataFrame(out, columns=["seq", "exch_ts", "recv_ns", "bid", "bid_qty", "ask", "ask_qty",
                                      f"bid_depth{levels}", f"ask_depth{levels}"])
