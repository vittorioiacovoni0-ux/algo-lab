"""Fase 1 - Live in PAPER: stesso codice della ricerca, eseguito a fine di ogni barra."""
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import API_KEY, API_SECRET
from data import get_closes
from strategy import leg_weights, next_state, zscore

cfg = json.load(open("pair_config.json"))
A, B, BETA, MIN = cfg["a"], cfg["b"], cfg["beta"], cfg["minutes"]
GROSS_NOTIONAL = 10_000          # dollari di esposizione lorda totale sulle due gambe
BAR_DELAY_S = 45                 # attesa dopo la chiusura della barra, per dare tempo ad Alpaca di pubblicarla

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("live_pairs.log"), logging.StreamHandler()])
log = logging.getLogger()
trading = TradingClient(API_KEY, API_SECRET, paper=True)   # paper=True: non toccare


def current_qty():
    pos = {p.symbol: float(p.qty) for p in trading.get_all_positions()}
    return pos.get(A, 0.0), pos.get(B, 0.0)


def state_from_qty(qa):
    # assume che queste posizioni le apra solo questo bot
    return int(np.sign(qa))


def send(symbol, delta):
    if delta == 0:
        return
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    order = trading.submit_order(MarketOrderRequest(symbol=symbol, qty=abs(delta), side=side,
                                                    time_in_force=TimeInForce.DAY))
    log.info(f"ORDINE {side.value} {abs(delta)} {symbol} id={order.id}")


def step():
    closes = get_closes([A, B], MIN, days=10)
    spread = np.log(closes[A]) - BETA * np.log(closes[B])
    z = zscore(spread, cfg["window"]).iloc[-1]
    qa, qb = current_qty()
    state = state_from_qty(qa)
    new = next_state(state, z, cfg["z_in"], cfg["z_out"], cfg["z_stop"])
    pa, pb = closes[A].iloc[-1], closes[B].iloc[-1]
    wa, wb = leg_weights(BETA)
    ta, tb = round(new * wa * GROSS_NOTIONAL / pa), round(new * wb * GROSS_NOTIONAL / pb)
    log.info(f"barra {closes.index[-1]:%H:%M} z={z:+.2f} stato {state}->{new} "
             f"{A}: {qa:.0f}->{ta}  {B}: {qb:.0f}->{tb}")
    if new != state:                      # ribilancia solo ai cambi di stato, non a ogni barra
        send(A, ta - int(qa))
        send(B, tb - int(qb))


def sleep_to_next_bar():
    now = datetime.now(timezone.utc)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=MIN - now.minute % MIN)
    time.sleep(max(1, (nxt - now).total_seconds() + BAR_DELAY_S))


if __name__ == "__main__":
    log.info(f"Avvio pairs {A}/{B} beta={BETA}")
    while True:
        try:
            clock = trading.get_clock()
            if not clock.is_open:
                wait = (clock.next_open - clock.timestamp).total_seconds()
                log.info(f"Mercato chiuso, riapre {clock.next_open}. Attendo.")
                time.sleep(min(wait + BAR_DELAY_S, 3600))
                continue
            step()
        except Exception:
            log.exception("Errore nel ciclo, riprovo alla prossima barra")
        sleep_to_next_bar()
