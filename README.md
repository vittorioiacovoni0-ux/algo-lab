# algo-lab

Laboratorio personale di trading algoritmico: dalla ricerca al paper trading, fino alla microstruttura e all'HFT.

## Fase 1 — Pairs trading intraday (Alpaca paper)

Strategia di mean reversion su coppie di azioni USA cointegrate, su barre da 15 minuti.

- `strategy.py`: logica condivisa tra backtest e live (z-score dello spread, macchina a stati con ingresso, uscita e stop)
- `research_pairs.py`: test di cointegrazione Engle-Granger sul train, valutazione out-of-sample con costi di transazione e controllo dei test multipli
- `live_pairs.py`: esecuzione event-driven su conto paper Alpaca, con logging e gestione degli orari di mercato

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # inserisci le chiavi paper di Alpaca
python research_pairs.py
python live_pairs.py
```

## Roadmap
- [x] Fase 1: pairs trading, ricerca out-of-sample + paper trading
- [ ] Fase 2: registrazione order book crypto (L2) e analisi di microstruttura
- [ ] Fase 3: market making Avellaneda-Stoikov con `hftbacktest` (coda e latenza)
- [ ] Fase 4: pipeline completa su `nautilus_trader` + testnet
