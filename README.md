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
python preflight.py    # controlli sul conto paper, nessun ordine
python live_pairs.py
python analyze_fills.py   # dopo qualche giorno: slippage reale dei fill
```

## Infrastruttura comune

- `data.py`: `get_bars` scarica storico SIP aggiustato per split e dividendi (total return) con cache Parquet
- `backtest.py`: motore su pesi di portafoglio, con deriva dei pesi tra i ribilanciamenti, ritardo di esecuzione, costi sul turnover, prestito sugli short, rendimenti in eccesso sul cash (BIL), t-stat e costo di pareggio

## Fase 2 — Strategie giornaliere su ETF (`daily/`)

Parametri dalla letteratura fissati a priori; campione 2017–2026 quasi tutto successivo alla pubblicazione dei paper, quindi un test fuori campione delle anomalie. Esecuzione alla chiusura successiva alla decisione, 5 bps per unità di turnover.

| 2017-02 – 2026-10 | CAGR | Sharpe netto | t-stat | Max DD | Corr. SPY |
|---|---|---|---|---|---|
| Trend following (TSMOM, 20 ETF multi-asset) | 3.6% | 0.17 | 0.52 | −23% | 0.08 |
| Momentum settoriale top 3 | 14.8% | 0.71 | 2.20 | −30% | 0.89 |
| Vol targeting su SPY | 14.6% | 0.76 | 2.35 | −26% | 0.91 |
| SPY buy & hold | 15.1% | 0.74 | 2.30 | −34% | 1.00 |
| 60/40 SPY/IEF | 9.5% | 0.68 | 2.12 | −21% | 0.97 |

Conclusioni:
1. Nessuna strategia batte SPY in Sharpe in modo distinguibile dal rumore. In un decennio dominato dal mercato azionario USA, il momentum settoriale è essenzialmente SPY con un altro nome (beta 0.92).
2. Il trend following ha rendimento basso ma correlazione quasi nulla con SPY e paga quando serve (2022: +16.7% contro −18.2%): il suo valore è da diversificatore, da valutare nel portafoglio (Fase 4).
3. Il vol targeting non aumenta lo Sharpe ma riduce il drawdown (−26% contro −34%); costo tipico: il rimbalzo a V perso nel 2020.
4. Anche lo Sharpe di SPY ha t-stat ~2.3 su quasi 10 anni: con questa lunghezza di storia, differenze di Sharpe di 0.1–0.2 non sono misurabili.

## Fase 3 — Intraday momentum (`intraday/`)

Gao, Han, Li, Zhou (2018): il rendimento dalla chiusura di ieri alle 10:00 predice l'ultima mezz'ora. Barre a 5 minuti, SPY/QQQ/IWM/DIA, 2016–2026 (fuori dal campione del paper, 1993–2013).

- Regressione predittiva su SPY: beta 0.016, t Newey-West 0.58, R² 0.15% (paper: R² ~1.6%). Su nessun ETF t > 1.5.
- Nei giorni ad alta volatilità (soglia calcolata solo sul passato) il beta è positivo come nel paper, nei giorni calmi nullo o negativo, ma senza significatività.
- La strategia (segno di r1, posizione 15:30–16:00) ha rendimento lordo intorno a zero e netto −5% l'anno: un round trip al giorno a 1 bp per lato costa ~5% annuo. Costo di pareggio < 0.5 bps.

Conclusione: l'effetto non sopravvive fuori campione sugli ETF; se esiste, è troppo piccolo per pagare anche i costi minimi.

## Deploy su VM (Ubuntu)

```bash
curl -fsSL https://raw.githubusercontent.com/vittorioiacovoni0-ux/algo-lab/main/deploy/setup.sh | bash
```
Installa l'ambiente, esegue i test e registra il bot come servizio systemd (`algo-pairs`), che si avvia a mano dopo aver creato `~/algo-lab/.env` con le chiavi paper. Log: `journalctl -u algo-pairs -f`.

## Roadmap
- [ ] Fase 1 (in chiusura): pairs trading intraday. Ricerca conclusa (nessun edge netto); resta il paper trading come test di esecuzione e misura dello slippage
- [x] Fase 2: strategie giornaliere su ETF multi-asset (trend following, momentum settoriale, vol targeting) contro i benchmark
- [x] Fase 3: intraday momentum su ETF azionari, regressioni predittive e strategie con costi
- [ ] Fase 4: portafoglio di strategie con risk management e deploy in paper su VM con monitoraggio

Il codice di microstruttura/HFT sviluppato e poi accantonato è in `archive/hft/` (non in uso).
