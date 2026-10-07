"""Controlli prima di avviare live_pairs.py. NON invia ordini.

Verifica: chiavi e conto paper, orari di mercato, posizioni e ordini aperti sulle due gambe, dati IEX
aggiornati, e mostra cosa farebbe il bot adesso (stato, z-score, quantità obiettivo).

Uso: python preflight.py
"""
import json

import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from config import API_KEY, API_SECRET
from data import get_closes
from strategy import leg_weights, next_state, zscore

GROSS_NOTIONAL = 10_000          # come in live_pairs.py


def check(ok, msg):
    print(f"[{'OK' if ok else '!!'}] {msg}")
    return ok


def main():
    cfg = json.load(open("pair_config.json"))
    a, b, beta = cfg["a"], cfg["b"], cfg["beta"]
    trading = TradingClient(API_KEY, API_SECRET, paper=True)
    acct = trading.get_account()
    ok = check(str(acct.account_number).startswith("PA"), f"conto paper ({acct.account_number}), stato {acct.status}")
    ok &= check(not acct.trading_blocked, "trading non bloccato")
    ok &= check(float(acct.buying_power) > 2 * GROSS_NOTIONAL,
                f"buying power {float(acct.buying_power):,.0f}$ (servono ~{GROSS_NOTIONAL:,}$ lordi)")
    clock = trading.get_clock()
    check(True, f"mercato {'APERTO' if clock.is_open else 'chiuso'}; prossima apertura {clock.next_open}, "
                f"prossima chiusura {clock.next_close}")

    pos = {p.symbol: float(p.qty) for p in trading.get_all_positions()}
    others = {s: q for s, q in pos.items() if s not in (a, b)}
    ok &= check(True, f"posizioni su {a}/{b}: {pos.get(a, 0):.0f} / {pos.get(b, 0):.0f}")
    check(not others, f"altre posizioni nel conto: {others or 'nessuna'} (il bot le ignora)")
    open_orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[a, b]))
    ok &= check(not open_orders, f"ordini aperti su {a}/{b}: {len(open_orders)}")

    closes = get_closes([a, b], cfg["minutes"], days=10)
    n = len(closes)
    ok &= check(n >= cfg["window"], f"{n} barre IEX da {cfg['minutes']} min (servono almeno {cfg['window']})")
    spread = np.log(closes[a]) - beta * np.log(closes[b])
    z = zscore(spread, cfg["window"]).iloc[-1]
    state = int(np.sign(pos.get(a, 0)))
    new = next_state(state, z, cfg["z_in"], cfg["z_out"], cfg["z_stop"])
    wa, wb = leg_weights(beta)
    pa, pb = closes[a].iloc[-1], closes[b].iloc[-1]
    ta, tb = round(new * wa * GROSS_NOTIONAL / pa), round(new * wb * GROSS_NOTIONAL / pb)
    print(f"\nUltima barra {closes.index[-1]:%Y-%m-%d %H:%M}: {a} {pa:.2f}, {b} {pb:.2f}, z = {z:+.2f}")
    print(f"Il bot ora: stato {state} -> {new}, obiettivo {a} {ta}, {b} {tb} "
          f"(entra se |z| > {cfg['z_in']}, esce se |z| < {cfg['z_out']}, stop a |z| >= {cfg['z_stop']})")
    print("\nTutto pronto: python live_pairs.py" if ok else "\nSistema i punti [!!] prima di avviare il bot.")


if __name__ == "__main__":
    main()
