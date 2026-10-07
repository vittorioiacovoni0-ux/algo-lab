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
from strategy import leg_weights, must_be_flat, next_state, zscore

cfg = json.load(open("pair_config.json"))
A, B, BETA, MIN = cfg["a"], cfg["b"], cfg["beta"], cfg["minutes"]
GROSS_NOTIONAL = 10_000          # dollari di esposizione lorda totale sulle due gambe
BAR_DELAY_S = 45                 # attesa dopo la chiusura della barra, per dare tempo ad Alpaca di pubblicarla
FILL_TIMEOUT_S = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("live_pairs.log"), logging.StreamHandler()])
log = logging.getLogger()
trading = TradingClient(API_KEY, API_SECRET, paper=True)   # paper=True: non toccare


def current_qty():
    pos = {p.symbol: float(p.qty) for p in trading.get_all_positions()}
    return pos.get(A, 0.0), pos.get(B, 0.0)


def state_from_qty(qa, qb):
    """Stato dalle posizioni reali (assume che su A e B operi solo questo bot). None se le gambe non sono
    coerenti con un pairs trade (es. un solo ordine eseguito): in quel caso si chiude tutto."""
    sa, sb = int(np.sign(qa)), int(np.sign(qb))
    expected_sb = sa * int(np.sign(leg_weights(BETA)[1]))
    if sa == 0 and sb == 0:
        return 0
    if sa != 0 and sb == expected_sb:
        return sa
    return None


def send(symbol, delta):
    if delta == 0:
        return None
    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
    order = trading.submit_order(MarketOrderRequest(symbol=symbol, qty=abs(delta), side=side,
                                                    time_in_force=TimeInForce.DAY))
    log.info(f"ORDINE {side.value} {abs(delta)} {symbol} id={order.id}")
    return order


def log_fill(order, ref_price):
    """Aspetta il fill e logga lo slippage rispetto alla chiusura della barra (bps, positivo = costo).
    Serve a ricalibrare COST_BPS della ricerca."""
    if order is None:
        return
    for _ in range(FILL_TIMEOUT_S):
        order = trading.get_order_by_id(order.id)
        if order.filled_avg_price is not None and float(order.filled_qty) == float(order.qty):
            fill = float(order.filled_avg_price)
            sign = 1 if order.side == OrderSide.BUY else -1
            slip = sign * (fill - ref_price) / ref_price * 1e4
            log.info(f"FILL {order.symbol} {order.filled_qty} @ {fill:.4f} chiusura barra {ref_price:.4f} "
                     f"slippage {slip:+.1f} bps")
            return
        time.sleep(1)
    log.warning(f"{order.symbol}: ordine {order.id} non eseguito del tutto entro {FILL_TIMEOUT_S}s "
                f"(stato {order.status}, eseguiti {order.filled_qty})")


def last_bar_close(now):
    """Chiusura dell'ultima barra che dovrebbe essere completa adesso (allineata alla griglia di MIN minuti)."""
    return now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % MIN)


def step(clock):
    expected_close = last_bar_close(datetime.now(timezone.utc))
    # chiusura di fine giornata decisa sull'orario: ha la precedenza sui controlli dei dati (per azzerare
    # le posizioni non serve un prezzo aggiornato, e saltarla vorrebbe dire restare esposti overnight)
    eod = must_be_flat(expected_close, clock.next_close, MIN)
    raw = get_closes([A, B], MIN, days=10, fill=False)
    bar_close = raw.index[-1] + timedelta(minutes=MIN)
    if not eod and bar_close < expected_close:
        log.warning(f"Ultima barra disponibile chiusa alle {bar_close:%H:%M}: dati in ritardo, salto la barra")
        return
    if not eod and raw.iloc[-1][[A, B]].isna().any():
        log.warning(f"Barra {raw.index[-1]:%H:%M} senza scambi IEX su una gamba: prezzo stantio, salto la barra")
        return
    closes = raw.ffill().dropna()
    spread = np.log(closes[A]) - BETA * np.log(closes[B])
    z = zscore(spread, cfg["window"]).iloc[-1]
    qa, qb = current_qty()
    state = state_from_qty(qa, qb)
    if state is None:
        log.error(f"Gambe incoerenti ({A}: {qa:.0f}, {B}: {qb:.0f}): chiudo tutto")
        new = 0
    elif eod:
        new = 0
    else:
        new = next_state(state, z, cfg["z_in"], cfg["z_out"], cfg["z_stop"])
    pa, pb = closes[A].iloc[-1], closes[B].iloc[-1]
    wa, wb = leg_weights(BETA)
    ta, tb = round(new * wa * GROSS_NOTIONAL / pa), round(new * wb * GROSS_NOTIONAL / pb)
    log.info(f"barra {closes.index[-1]:%H:%M} z={z:+.2f} stato {state}->{new} "
             f"{A}: {qa:.0f}->{ta}  {B}: {qb:.0f}->{tb}")
    if new != state:                      # ribilancia solo ai cambi di stato, non a ogni barra
        oa, ob = send(A, ta - int(qa)), send(B, tb - int(qb))   # prima invio entrambe le gambe, poi controllo
        log_fill(oa, pa)
        log_fill(ob, pb)


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
            step(clock)
        except Exception:
            log.exception("Errore nel ciclo, riprovo alla prossima barra")
        sleep_to_next_bar()
