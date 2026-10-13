"""Day trading - bot live in PAPER: NoiseBoundary su QQQ, stessa classe del backtest.

Flusso di una giornata:
- Prima della prima decisione: storia dei giorni precedenti dal feed SIP (come nel backtest) -> rumore per ora
  del giorno, volatilità, chiusura di ieri, leva del giorno. Equity di riferimento = equity del conto, max CAPITAL.
- A ogni orario di decisione (10:00, 10:30, ... 15:30) + DECISION_DELAY_S: barre di oggi dal feed IEX (l'unico
  gratuito in tempo reale), controllo che la barra appena chiusa ci sia, decisione con strategy.on_bar, ordini.
- Inversioni in due ordini (chiudo, poi apro). Prima di ogni ordine cancello lo stop di emergenza: finché è
  aperto "impegna" le azioni e il broker rifiuterebbe la vendita.
- Stop di emergenza presso il broker dopo ogni ingresso, alla distanza che fa perdere MAX_DAILY_LOSS dell'equity:
  protegge anche se il PC si spegne. Il backtest controlla il limite a fine barra; qui è una rete di sicurezza.
- FLATTEN_BEFORE_CLOSE minuti prima della chiusura: ordine market-on-close (scadenza Nasdaq 15:50; nelle mezze
  giornate l'orario si sposta con clock.next_close).
- Kill switch: crea un file KILL nella cartella -> cancella gli ordini, chiude la posizione, esce.

Uso:
  python -m daytrade.live --check      # stato e cosa farebbe adesso, nessun ordine
  python -m daytrade.live --dry-run    # gira tutta la giornata ma logga gli ordini invece di inviarli
  python -m daytrade.live              # paper trading
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from daytrade.engine import split_days, target_shares
from daytrade.strategies import OPEN_MIN, NoiseBoundary

SYMBOL = "QQQ"
CAPITAL = 100_000.0          # equity massima usata per il dimensionamento
MAX_LEV = 4.0
MAX_DAILY_LOSS = 0.03
DECISION_DELAY_S = 15        # attesa dopo l'orario di decisione: la barra da 1 minuto deve essere pubblicata
DATA_RETRIES = 3
FLATTEN_BEFORE_CLOSE = pd.Timedelta(minutes=14)
HISTORY_CALENDAR_DAYS = 45   # bastano per 14+ giorni di borsa anche con festività
KILL_FILE = Path("KILL")
TZ = "America/New_York"

log = logging.getLogger("daytrade")


# ---------------------------------------------------------------- dati

def history_days(today):
    """Giorni COMPLETI precedenti a oggi, barre da 1 minuto SIP aggiustate: la stessa fonte del backtest."""
    import data
    start = (today - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    df = data.get_bars(SYMBOL, minutes=1, start=start).loc[SYMBOL]
    return split_days(df[df.index.normalize().tz_localize(None) < today])


def today_bars(now):
    """Barre da 1 minuto di oggi dal feed IEX, solo quelle già chiuse, sessione regolare."""
    import data
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    start = now.normalize() + pd.Timedelta(minutes=OPEN_MIN)
    req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Minute, start=start.to_pydatetime(),
                           end=now.to_pydatetime(), feed=DataFeed.IEX)
    df = data._client.get_stock_bars(req).df
    if df.empty:
        return None
    df = df.loc[SYMBOL]
    df.index = df.index.tz_convert(TZ)
    df = df[(df.index >= start) & (df.index + pd.Timedelta(minutes=1) <= now)]
    return split_days(df)[0] if len(df) else None


def decision_times(strategy, day, close):
    """Orari di decisione di oggi (fine barra), prima dell'uscita di fine giornata."""
    out = []
    for end in range(strategy.first, strategy.last + 1):
        t = day + pd.Timedelta(minutes=end)
        if strategy.is_decision_time(end) and t < close - FLATTEN_BEFORE_CLOSE:
            out.append(t)
    return out


# ---------------------------------------------------------------- broker

class Broker:
    """Accesso al conto paper. Isolato in una classe per poterlo sostituire con un finto nei test."""

    def __init__(self, trading):
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopOrderRequest
        self.t = trading
        self.OrderSide, self.TIF, self.Status = OrderSide, TimeInForce, QueryOrderStatus
        self.Market, self.Stop, self.Orders = MarketOrderRequest, StopOrderRequest, GetOrdersRequest

    def position(self):
        return int(sum(float(p.qty) for p in self.t.get_all_positions() if p.symbol == SYMBOL))

    def equity(self):
        return float(self.t.get_account().equity)

    def cancel_open_orders(self):
        for o in self.t.get_orders(self.Orders(status=self.Status.OPEN, symbols=[SYMBOL])):
            self.t.cancel_order_by_id(o.id)
        for _ in range(20):                         # aspetto che le cancellazioni siano effettive
            if not self.t.get_orders(self.Orders(status=self.Status.OPEN, symbols=[SYMBOL])):
                return
            time.sleep(0.5)
        raise RuntimeError("ordini ancora aperti dopo la cancellazione")

    def _side(self, qty):
        return self.OrderSide.BUY if qty > 0 else self.OrderSide.SELL

    def market(self, qty):
        """Ordine a mercato; ritorna il prezzo medio eseguito (None se non eseguito entro 30 s)."""
        o = self.t.submit_order(self.Market(symbol=SYMBOL, qty=abs(qty), side=self._side(qty),
                                            time_in_force=self.TIF.DAY))
        for _ in range(60):
            o = self.t.get_order_by_id(o.id)
            if o.filled_avg_price is not None and float(o.filled_qty) == abs(qty):
                return float(o.filled_avg_price)
            time.sleep(0.5)
        log.error(f"ordine {o.id} non eseguito del tutto: stato {o.status}, eseguiti {o.filled_qty}")
        return None

    def stop(self, qty, price):
        self.t.submit_order(self.Stop(symbol=SYMBOL, qty=abs(qty), side=self._side(qty), stop_price=round(price, 2),
                                      time_in_force=self.TIF.DAY))

    def market_on_close(self, qty):
        self.t.submit_order(self.Market(symbol=SYMBOL, qty=abs(qty), side=self._side(qty),
                                        time_in_force=self.TIF.CLS))


# ---------------------------------------------------------------- logica di trading

class Trader:
    def __init__(self, broker, dry_run=False):
        self.broker, self.dry_run = broker, dry_run
        self.strategy, self.halted = None, False

    def start_day(self, history):
        self.strategy = NoiseBoundary(max_lev=MAX_LEV)
        self.strategy.start_day(None, history)
        self.acct0 = self.broker.equity()
        self.eq0 = min(self.acct0, CAPITAL)
        self.halted = False
        if self.strategy.sigma is None:
            log.warning(f"storia insufficiente ({len(history)} giorni): oggi non si opera")
            self.halted = True
        else:
            log.info(f"inizio giornata: equity {self.acct0:,.0f}$ (uso {self.eq0:,.0f}$), leva del giorno "
                     f"{self.strategy.lev:.2f}, chiusura di ieri {self.strategy.prev_close:.2f}")

    def on_decision(self, bars, end):
        """Decisione all'orario `end` (minuto di fine barra). bars: barre di oggi già chiuse."""
        if self.halted:
            return
        qty = self.broker.position()
        if (self.broker.equity() - self.acct0) / self.eq0 <= -MAX_DAILY_LOSS:
            log.error("limite di perdita giornaliero raggiunto: chiudo e non opero più oggi")
            self.halted = True
            self.rebalance(qty, 0, None)
            return
        if bars is None or bars.minute[-1] + 1 != end:
            log.warning(f"{end // 60:02d}:{end % 60:02d} barra mancante o in ritardo: salto la decisione")
            return
        lv = self.strategy.levels(bars)
        intent = self.strategy.on_bar(bars, int(np.sign(qty)))
        if lv:
            log.info(f"{end // 60:02d}:{end % 60:02d} prezzo {lv['price']:.2f} sopra {lv['upper']:.2f} "
                     f"sotto {lv['lower']:.2f} vwap {lv['vwap']:.2f} posizione {qty}"
                     + (f" -> lato {intent.side}" if intent else " -> nessun cambio"))
        if intent is not None:
            self.rebalance(qty, target_shares(intent, qty, self.eq0, bars.close[-1], MAX_LEV), bars.close[-1])

    def rebalance(self, qty, target, ref_price):
        if target == qty:
            return
        if self.dry_run:
            log.info(f"[DRY RUN] da {qty} a {target} azioni {SYMBOL}")
            return
        self.broker.cancel_open_orders()            # lo stop di emergenza impegna le azioni
        if qty and (target == 0 or np.sign(target) != np.sign(qty)):
            px = self.broker.market(-qty)
            self._fill_log(px, -qty, ref_price)
            if px is None:                          # chiusura non riuscita: non apro il lato opposto
                log.error("chiusura non eseguita: non apro nuove posizioni, ricontrollo al prossimo orario")
                return
            qty = 0
        if target:
            px = self.broker.market(target)
            self._fill_log(px, target, ref_price)
            if px is not None:
                stop_px = px - np.sign(target) * MAX_DAILY_LOSS * self.eq0 / abs(target)
                self.broker.stop(-target, stop_px)
                log.info(f"stop di emergenza a {stop_px:.2f}")

    def flatten_eod(self):
        qty = self.broker.position()
        if not qty:
            return
        if self.dry_run:
            log.info(f"[DRY RUN] market-on-close per chiudere {qty} azioni")
            return
        self.broker.cancel_open_orders()
        self.broker.market_on_close(-qty)
        log.info(f"market-on-close inviato: chiudo {qty} azioni")

    def kill(self):
        log.error("KILL: cancello gli ordini e chiudo la posizione")
        self.broker.cancel_open_orders()
        qty = self.broker.position()
        if qty:
            self.broker.market(-qty)

    @staticmethod
    def _fill_log(px, qty, ref):
        if px is None or ref is None:
            return
        slip = np.sign(qty) * (px - ref) / ref * 1e4
        log.info(f"FILL {qty:+d} {SYMBOL} @ {px:.2f}, prezzo di decisione {ref:.2f}, slippage {slip:+.2f} bps")


# ---------------------------------------------------------------- ciclo principale

def sleep_until(t, now_fn):
    """Dorme fino a t a passi brevi, per accorgersi del file KILL. Ritorna False se è comparso KILL."""
    while True:
        if KILL_FILE.exists():
            return False
        left = (t - now_fn()).total_seconds()
        if left <= 0:
            return True
        time.sleep(min(left, 5))


def run(dry_run=False):
    from alpaca.trading.client import TradingClient
    from config import API_KEY, API_SECRET
    trading = TradingClient(API_KEY, API_SECRET, paper=True)       # paper=True: non toccare
    trader = Trader(Broker(trading), dry_run)

    def now():
        return pd.Timestamp.now(tz=TZ)

    if KILL_FILE.exists():
        log.error(f"esiste il file {KILL_FILE}: rimuovilo per avviare il bot (serve solo a fermarlo)")
        return
    log.info(f"avvio day trading {SYMBOL}{' (DRY RUN)' if dry_run else ''}")
    while True:
        if KILL_FILE.exists():
            trader.kill()
            return
        clock = trading.get_clock()
        if not clock.is_open:
            nxt = pd.Timestamp(clock.next_open).tz_convert(TZ)
            log.info(f"mercato chiuso, riapre {nxt:%Y-%m-%d %H:%M}")
            sleep_until(min(nxt, now() + pd.Timedelta(hours=1)), now)
            continue
        today = now().normalize()
        close = pd.Timestamp(clock.next_close).tz_convert(TZ)
        trader.start_day(history_days(today.tz_localize(None)))
        for t in decision_times(trader.strategy, today, close):
            if t + pd.Timedelta(seconds=DECISION_DELAY_S) < now():
                continue                                            # già passato (avvio a giornata in corso)
            if not sleep_until(t + pd.Timedelta(seconds=DECISION_DELAY_S), now):
                break
            end = t.hour * 60 + t.minute
            bars = None
            for attempt in range(DATA_RETRIES + 1):
                bars = today_bars(now())
                if bars is not None and bars.minute[-1] + 1 == end:
                    break
                time.sleep(10)
            try:
                trader.on_decision(bars, end)
            except Exception:
                log.exception("errore nella decisione: riprovo al prossimo orario")
        if KILL_FILE.exists():
            continue
        sleep_until(close - FLATTEN_BEFORE_CLOSE, now)
        trader.flatten_eod()
        sleep_until(close + pd.Timedelta(minutes=2), now)


def check():
    """Stato del conto e cosa farebbe il bot adesso. Nessun ordine."""
    from alpaca.trading.client import TradingClient
    from config import API_KEY, API_SECRET
    trading = TradingClient(API_KEY, API_SECRET, paper=True)
    broker = Broker(trading)
    acct = trading.get_account()
    clock = trading.get_clock()
    now = pd.Timestamp.now(tz=TZ)
    print(f"conto {acct.account_number} ({'PAPER' if str(acct.account_number).startswith('PA') else '??'}), "
          f"equity {float(acct.equity):,.0f}$, buying power {float(acct.buying_power):,.0f}$, shorting "
          f"{'abilitato' if acct.shorting_enabled else 'NON abilitato'}")
    print(f"mercato {'aperto' if clock.is_open else 'chiuso'}; prossima chiusura {clock.next_close}")
    print(f"posizione {SYMBOL}: {broker.position()} azioni; altre posizioni: "
          f"{[p.symbol for p in trading.get_all_positions() if p.symbol != SYMBOL] or 'nessuna'}")
    # a mercato chiuso mi riferisco alla PROSSIMA sessione (la sua storia include la seduta appena conclusa)
    session = now.normalize() if clock.is_open else pd.Timestamp(clock.next_open).tz_convert(TZ).normalize()
    hist = history_days(session.tz_localize(None))
    s = NoiseBoundary(max_lev=MAX_LEV)
    s.start_day(None, hist)
    if s.sigma is None:
        print(f"storia insufficiente: {len(hist)} giorni")
        return
    print(f"sessione {session:%Y-%m-%d}: storia di {len(hist)} giorni fino al {hist[-1].date:%Y-%m-%d}; "
          f"chiusura precedente {s.prev_close:.2f}; "
          f"leva del giorno {s.lev:.2f} -> {s.lev * min(float(acct.equity), CAPITAL) / s.prev_close:.0f} azioni circa")
    bars = today_bars(now) if clock.is_open else None
    if bars is not None:
        lv = s.levels(bars)
        if lv:
            print(f"ultima barra {bars.minute[-1] // 60:02d}:{bars.minute[-1] % 60:02d}: prezzo {lv['price']:.2f}, "
                  f"confini {lv['lower']:.2f} / {lv['upper']:.2f}, vwap {lv['vwap']:.2f}")
    close = pd.Timestamp(clock.next_close).tz_convert(TZ)
    times = decision_times(s, session, close)
    print(f"orari di decisione: {', '.join(f'{t:%H:%M}' for t in times)}; uscita market-on-close alle "
          f"{close - FLATTEN_BEFORE_CLOSE:%H:%M}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler("daytrade_live.log"), logging.StreamHandler()])
    if args.check:
        check()
    else:
        run(dry_run=args.dry_run)
