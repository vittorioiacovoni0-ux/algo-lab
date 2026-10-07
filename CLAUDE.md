# algo-lab — contesto per Claude Code

## Chi sono e obiettivo
Sono Vittorio, studente del master in Financial Technology & Analytics a Stevens (NJ), con background in asset management e derivati. Questo repo è un laboratorio personale per imparare il trading algoritmico e l'HFT in modo rigoroso, dalla ricerca al paper trading. È anche un portfolio per candidature quant. Rispondimi in italiano; codice, nomi di variabili e commit in inglese o italiano va bene, ma coerenti con i file esistenti.

## Stato attuale — Fase 1: pairs trading intraday (Alpaca paper)
- `strategy.py`: logica condivisa tra backtest e live (z-score dello spread su log-prezzi, macchina a stati 0/+1/-1 con z_in=2, z_out=0.5, z_stop=4, mai inversione diretta long→short). REGOLA: backtest e live devono usare lo stesso codice.
- `research_pairs.py`: universo di 16 titoli USA, barre 15 min, 180 giorni. Test di cointegrazione Engle-Granger sul 60% train, hedge ratio OLS, valutazione solo out-of-sample sul 40% con costi (COST_BPS=3 per unità di turnover). Posizione decisa a fine barra t, PnL da t+1. Stampa il numero di coppie attese per caso (test multipli). Salva la migliore in `pair_config.json`.
- `live_pairs.py`: loop event-driven a fine barra (+45 s di buffer), legge le posizioni reali dal conto, ribilancia solo ai cambi di stato, logga in `live_pairs.log`, gestisce gli orari di mercato via clock API.
- `data.py`: barre Alpaca, feed IEX gratuito, solo orario regolare, scarta la barra in corso.
- `config.py` legge le chiavi dalle variabili d'ambiente (Codespaces secrets; `.env` opzionale in locale, escluso via `.gitignore`).
- Testato solo su dati sintetici; prima esecuzione su dati reali ancora da fare.

## Vincoli non negoziabili
- `TradingClient(..., paper=True)` sempre. Mai codice che punti a un conto live.
- Mai leggere, stampare o committare `.env` o chiavi API.
- Niente lookahead: ogni segnale usa solo dati disponibili al momento della decisione. Segnalami qualsiasi rischio di leakage.
- Ogni backtest riporta rendimento lordo e netto, Sharpe, max drawdown, numero di trade, ed è valutato out-of-sample.
- Sono negli USA: binance.com e Bybit bloccano gli IP statunitensi. Per i dati crypto usare Coinbase o Kraken (websocket pubblici) oppure i campioni di Tardis.dev.

## Roadmap
1. [in corso] Fase 1: eseguire la ricerca su dati reali, analizzare i risultati, far girare il bot in paper alcuni giorni, misurare lo slippage reale (fill contro chiusura barra) e ricalibrare COST_BPS. Possibili miglioramenti: hedge ratio rolling, cooldown dopo lo stop, walk-forward invece di un singolo split.
2. Fase 2: recorder di order book L2 + trade crypto (Coinbase/Kraken) in Parquet; analisi di microstruttura (spread, order book imbalance, impatto dei trade).
3. Fase 3: market making Avellaneda-Stoikov / GLFT con `hftbacktest`, confrontando i risultati con e senza modello di coda e latenza.
4. Fase 4: pipeline completa su `nautilus_trader` collegata a un testnet.

## Come voglio lavorare
- Prima di modifiche non banali spiegami il piano in breve, poi procedi.
- Scrivi test per la logica della strategia (pytest, dati sintetici con relazioni note).
- Spiegami il perché delle scelte quantitative: sto imparando, non voglio solo codice funzionante.
- Commit piccoli e descrittivi, uno per modifica logica.