# Archivio HFT (non in uso)

Codice di microstruttura sviluppato e poi messo da parte: il progetto principale resta su strategie
intraday e giornaliere. Da riprendere in un progetto separato.

- `l2_recorder.py`: recorder di order book L2 + trade dal websocket pubblico Coinbase in Parquet, con
  timestamp dell'exchange e di ricezione, controllo dei numeri di sequenza e riconnessione.
- `orderbook.py`: ricostruzione del book dagli eventi registrati (`replay`).
- `l2-recorder.service`: unit systemd (path da adattare).

Appunti dai primi dati (ottobre 2026):
- In `market_trades` il campo `side` è il lato del MAKER: i trade "SELL" avvengono sull'ask.
- Spread BTC-USD tipicamente 1 tick ($0.01): nel market making conta la posizione in coda, non lo spread.
- ~10 MB/ora per BTC+ETH in fase calma.

Uso, dalla cartella `archive/hft`:
```bash
pip install -r requirements.txt
pytest tests
python l2_recorder.py BTC-USD ETH-USD
```
