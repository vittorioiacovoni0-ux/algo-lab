# algo-lab

Laboratorio personale di trading algoritmico intraday e giornaliero: dalla ricerca out-of-sample al paper trading.

## Fase 1 — Pairs trading intraday (Alpaca paper)

Strategia di mean reversion su coppie di azioni USA cointegrate, su barre da 15 minuti.

- `strategy.py`: logica condivisa tra backtest e live (z-score dello spread, macchina a stati con ingresso, uscita e stop)
- `research_pairs.py`: test di cointegrazione Engle-Granger sul train, valutazione out-of-sample con costi di transazione e controllo dei test multipli
- `live_pairs.py`: esecuzione event-driven su conto paper Alpaca, con logging e gestione degli orari di mercato
- `walkforward.py`: ricalibrazione a finestre mobili (60 giorni di formazione, 10 di trading), portafoglio equipesato delle coppie cointegrate contro l'ipotesi nulla "tutte le coppie"
- `spreads.py`: half-spread NBBO quotato nell'istante in cui il bot invia gli ordini

### Risultati (dati ott 2025 – ott 2026, 16 large cap USA, barre 15 min)

| Out-of-sample walk-forward | Coppie cointegrate | Tutte le coppie (nulla) |
|---|---|---|
| Rendimento lordo | +2.10% | +3.57% |
| Rendimento netto (3 bps/lato) | −2.45% | −1.59% |
| Sharpe netto | −0.79 | −0.85 |
| t-stat rendimenti giornalieri | −0.88 | −0.76 |
| Trade | 224 | 6300 |
| Costo di pareggio per lato | ~1.4 bps | ~2.1 bps |

Half-spread quotato medio sull'universo: **2.1 bps per lato** (da 0.6 di KO a 5.4 di YUM).

Conclusioni:
1. Esiste una piccola mean reversion intraday lorda negli spread, ma è dell'ordine del solo half-spread: con ordini a mercato non resta nulla.
2. Il filtro di cointegrazione non aggiunge valore: le coppie selezionate non battono coppie qualsiasi, e per finestra ne passano meno di quante ne attenderebbe il caso (~6 su 120).
3. Un singolo split train/validation/test aveva dato Sharpe +2.96 in validation e −2.25 in test sulla coppia scelta (C/WFC): distorsione da selezione su 11–17 trade, statisticamente indistinguibile da zero.

### Setup
Le chiavi paper di Alpaca vanno impostate come Codespaces secrets (`ALPACA_API_KEY`, `ALPACA_API_SECRET`); in locale si può usare un file `.env` (escluso da git).

```bash
pip install -r requirements.txt
pytest                 # test della logica della strategia
python research_pairs.py
python live_pairs.py
```

## Deploy su VM (Ubuntu)

```bash
curl -fsSL https://raw.githubusercontent.com/vittorioiacovoni0-ux/algo-lab/main/deploy/setup.sh | bash
```
Installa l'ambiente, esegue i test e registra il bot come servizio systemd (`algo-pairs`), che si avvia a mano dopo aver creato `~/algo-lab/.env` con le chiavi paper. Log: `journalctl -u algo-pairs -f`.

## Roadmap
- [ ] Fase 1 (in chiusura): pairs trading intraday. Ricerca conclusa (nessun edge netto); resta il paper trading come test di esecuzione e misura dello slippage
- [ ] Fase 2: strategie giornaliere su ETF multi-asset (trend following / time-series momentum, momentum cross-sectional, volatility targeting), walk-forward su ~10 anni
- [ ] Fase 3: intraday con effetti documentati in letteratura (es. intraday momentum su SPY: la prima mezz'ora predice l'ultima), valutati con i costi misurati in Fase 1
- [ ] Fase 4: portafoglio di strategie con risk management e deploy in paper su VM con monitoraggio

Il codice di microstruttura/HFT sviluppato e poi accantonato è in `archive/hft/` (non in uso).
