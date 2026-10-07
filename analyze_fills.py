"""Fase 1 - Analisi dei fill del paper trading: slippage reale contro chiusura barra, da live_pairs.log.

Lo slippage loggato è l'implementation shortfall rispetto alla chiusura della barra su cui il backtest assume
di eseguire: half-spread + movimento del prezzo nei ~45 s prima dell'invio + impatto. Confrontalo con
l'half-spread quotato di spreads.py e con COST_BPS della ricerca.
"""
import re
import sys

import pandas as pd

FILL = re.compile(r"^(\S+ \S+) .*FILL (\S+) (\S+) @ (\S+) chiusura barra (\S+) slippage (\S+) bps")


def parse(lines):
    rows = []
    for line in lines:
        m = FILL.search(line)
        if m:
            ts, sym, qty, fill, ref, slip = m.groups()
            rows.append({"time": pd.Timestamp(ts.replace(",", ".")), "symbol": sym, "qty": float(qty),
                         "fill": float(fill), "ref": float(ref), "slippage_bps": float(slip)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "live_pairs.log"
    with open(path) as f:
        fills = parse(f)
    if fills.empty:
        print("Nessun fill nel log.")
    else:
        print(fills.groupby("symbol").slippage_bps.describe()[["count", "mean", "50%", "std"]].round(2).to_string())
        se = fills.slippage_bps.std() / len(fills) ** 0.5
        print(f"\nMedia su {len(fills)} fill: {fills.slippage_bps.mean():+.2f} bps per lato "
              f"(errore standard {se:.2f}). COST_BPS della ricerca: 3")
