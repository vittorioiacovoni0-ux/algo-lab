# algo-lab — contesto per Claude Code

## Chi sono e obiettivo
Sono Vittorio, studente del master in Financial Technology & Analytics a Stevens (NJ), con background in asset management e derivati. Questo repo è un laboratorio personale per imparare il trading algoritmico intraday e giornaliero in modo rigoroso, dalla ricerca al paper trading. È anche un portfolio per candidature quant. Rispondimi in italiano; codice, nomi di variabili e commit in inglese o italiano va bene, ma coerenti con i file esistenti.

## Stato attuale — Fase 1: pairs trading intraday (Alpaca paper)
- `strategy.py`: logica condivisa tra backtest e live (z-score dello spread su log-prezzi, macchina a stati 0/+1/-1 con z_in=2, z_out=0.5, z_stop=4, mai inversione diretta long→short). REGOLA: backtest e live devono usare lo stesso codice.
- `research_pairs.py`: universo di 16 titoli USA, barre 15 min, 180 giorni. Split 50% train / 25% validation / 25% test: cointegrazione Engle-Granger (p-value peggiore delle due direzioni) e hedge ratio OLS sul train, scelta della coppia per Sharpe sulla validation, test guardato solo per la coppia scelta. Costi COST_BPS=3 per unità di turnover, PnL a quantità di azioni fisse come nel live. Posizione decisa a fine barra t, PnL da t+1. Flat a fine sessione (niente rischio overnight). Stampa il numero di coppie attese per caso (test multipli). Salva la migliore in `pair_config.json` solo con `--save` (il bot live lo legge: mai cambiarlo con posizioni aperte).
- `live_pairs.py`: loop event-driven a fine barra (+45 s di buffer), legge le posizioni reali dal conto, ribilancia solo ai cambi di stato, chiude a fine sessione, salta barre stantie, chiude tutto se le gambe sono incoerenti, logga lo slippage dei fill contro la chiusura barra in `live_pairs.log`.
- `data.py`: barre Alpaca, feed IEX gratuito, solo orario regolare, scarta la barra in corso.
- `config.py` legge le chiavi dalle variabili d'ambiente (Codespaces secrets; `.env` opzionale in locale, escluso via `.gitignore`).
- `walkforward.py`: walk-forward 60/10 giorni, portafoglio di coppie cointegrate vs nulla "tutte le coppie". `spreads.py`: half-spread NBBO (feed SIP storico, gratuito oltre i 15 min) all'istante di invio ordini.
- Test pytest in `tests/` (strategia, ricerca, walk-forward su dati sintetici, live con client finto).
- Risultato su dati reali (ott 2025 – ott 2026, vedi README): nessun edge netto. Lordo ≈ +2–3.5%, netto negativo, costo di pareggio 1.4–2.1 bps/lato contro half-spread medio 2.1 bps. Il filtro di cointegrazione non batte la nulla.

## Fasi 2 e 3 (ricerca fatta 2026-10-07, risultati nel README)
- `data.get_bars`: SIP storico dal 2016, aggiustato (total return), cache in `data/cache/` per giorno di download, scaricato a blocchi simbolo x anno (in un'unica richiesta 10 anni di barre a 5 min saturano la memoria).
- `backtest.py`: motore su pesi (deriva, lag, costi, borrow, excess su BIL, t-stat, costo di pareggio). Usato da `daily/` e, per le metriche, da `intraday/`.
- `daily/` (`python -m daily.research`): TSMOM multi-asset, momentum settoriale, vol targeting su SPY vs SPY, 60/40, equipesati. Nessuna batte SPY in Sharpe; TSMOM è un diversificatore (corr 0.08, +16.7% nel 2022).
- `intraday/` (`python -m intraday.research`): intraday momentum GHLZ 2018 su SPY/QQQ/IWM/DIA. Effetto non significativo fuori campione, netto negativo.

## Archivio
- `archive/hft/`: recorder L2 Coinbase + ricostruzione del book, accantonati (decisione 2026-10-07: il progetto resta su intraday e giornaliero, l'HFT eventualmente in progetti separati). Non usarlo né estenderlo salvo richiesta esplicita. Ha test e requirements propri, fuori dalla suite principale.
- `deploy/`: systemd unit e `setup.sh` per far girare il bot paper su una VM Ubuntu (utente ubuntu).

## Vincoli non negoziabili
- `TradingClient(..., paper=True)` sempre. Mai codice che punti a un conto live.
- Mai leggere, stampare o committare `.env` o chiavi API.
- Niente lookahead: ogni segnale usa solo dati disponibili al momento della decisione. Segnalami qualsiasi rischio di leakage.
- Ogni backtest riporta rendimento lordo e netto, Sharpe, max drawdown, numero di trade, ed è valutato out-of-sample.
- Sono negli USA: binance.com e Bybit bloccano gli IP statunitensi. Per i dati crypto usare Coinbase o Kraken (websocket pubblici) oppure i campioni di Tardis.dev.

## Roadmap
1. [in chiusura] Fase 1: ricerca su dati reali e walk-forward fatti (nessun edge con ordini a mercato). Resta: far girare il bot in paper (PC o VM) alcune settimane come test di esecuzione e misurare l'implementation shortfall reale con `analyze_fills.py`, da confrontare con l'half-spread quotato (~2 bps).
2. [fatta] Fase 2: strategie giornaliere su ETF (`daily/`).
3. [fatta] Fase 3: intraday momentum (`intraday/`).
4. Fase 4: portafoglio di strategie, risk management (sizing, vol targeting, limiti), deploy paper su VM con monitoraggio.

## Come voglio lavorare
- Prima di modifiche non banali spiegami il piano in breve, poi procedi.
- Scrivi test per la logica della strategia (pytest, dati sintetici con relazioni note).
- Spiegami il perché delle scelte quantitative: sto imparando, non voglio solo codice funzionante.
- Commit piccoli e descrittivi, uno per modifica logica.