"""Fase 2 - Recorder di order book L2 e trade da Coinbase (websocket pubblico Advanced Trade) in Parquet.

Salva gli eventi grezzi normalizzati, non il book ricostruito: così si può rigiocare tutto (orderbook.replay)
e misurare qualsiasi cosa, latenza compresa. Ogni riga ha due orologi:
- exch_ts: timestamp del messaggio lato Coinbase
- recv_ns: orologio locale quando il messaggio viene letto dal socket (ns da epoch UTC)
recv_ns - exch_ts = latenza di rete + skew tra gli orologi (sincronizza il clock con NTP).

ATTENZIONE: in market_trades `side` è il lato del MAKER (verificato contro il book ricostruito: i trade "SELL"
avvengono sull'ask). L'aggressore è il lato opposto.

sequence_num è unico per connessione su tutti i canali: un salto = messaggi persi = book non più affidabile.
In quel caso registro l'evento e mi riconnetto, ricevendo uno snapshot nuovo.

Layout: data/l2/{tabella}/date=YYYY-MM-DD/part-{recv_ns}.parquet, un file ogni FLUSH_S secondi
(se il processo muore si perde al massimo l'ultimo intervallo). Si legge come un unico dataset con pyarrow.

Uso: python l2_recorder.py [BTC-USD ETH-USD ...]
"""
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

URL = "wss://advanced-trade-ws.coinbase.com"
PRODUCTS = ["BTC-USD", "ETH-USD"]
CHANNELS = ["level2", "market_trades", "heartbeats"]     # heartbeats: tiene viva la connessione nei momenti calmi
ROOT = Path("data/l2")
FLUSH_S = 60

log = logging.getLogger("l2_recorder")

SCHEMAS = {
    "book": pa.schema([("recv_ns", pa.int64()), ("exch_ts", pa.timestamp("ns", tz="UTC")),
                       ("event_time", pa.timestamp("ns", tz="UTC")), ("seq", pa.int64()),
                       ("product", pa.string()), ("type", pa.string()), ("side", pa.string()),
                       ("price", pa.float64()), ("qty", pa.float64())]),
    "trades": pa.schema([("recv_ns", pa.int64()), ("exch_ts", pa.timestamp("ns", tz="UTC")),
                         ("trade_time", pa.timestamp("ns", tz="UTC")), ("seq", pa.int64()),
                         ("product", pa.string()), ("type", pa.string()), ("trade_id", pa.int64()),
                         ("price", pa.float64()), ("size", pa.float64()), ("side", pa.string())]),
    "events": pa.schema([("recv_ns", pa.int64()), ("kind", pa.string()), ("detail", pa.string())]),
}


def ts(s):
    return pd.Timestamp(s).tz_convert("UTC") if s else pd.NaT


def normalize(msg, recv_ns):
    """Messaggio websocket -> {tabella: [righe]}. Funzione pura, testabile senza rete."""
    ch, seq, exch_ts = msg.get("channel"), msg.get("sequence_num"), ts(msg.get("timestamp"))
    rows = {"book": [], "trades": []}
    if ch == "l2_data":
        for ev in msg["events"]:
            for u in ev["updates"]:
                rows["book"].append({"recv_ns": recv_ns, "exch_ts": exch_ts, "event_time": ts(u.get("event_time")),
                                     "seq": seq, "product": ev["product_id"], "type": ev["type"],
                                     "side": u["side"], "price": float(u["price_level"]),
                                     "qty": float(u["new_quantity"])})
    elif ch == "market_trades":
        for ev in msg["events"]:
            for t in ev["trades"]:
                rows["trades"].append({"recv_ns": recv_ns, "exch_ts": exch_ts, "trade_time": ts(t.get("time")),
                                       "seq": seq, "product": t["product_id"], "type": ev["type"],
                                       "trade_id": int(t["trade_id"]), "price": float(t["price"]),
                                       "size": float(t["size"]), "side": t["side"]})
    return rows


class SequenceGap(Exception):
    pass


class SequenceChecker:
    def __init__(self):
        self.expected = None

    def check(self, seq):
        if self.expected is not None and seq != self.expected:
            exp, self.expected = self.expected, None
            raise SequenceGap(f"atteso {exp}, ricevuto {seq}")
        self.expected = seq + 1


class ParquetSink:
    """Bufferizza le righe e le scrive in un file per tabella ogni FLUSH_S secondi, in un thread."""

    def __init__(self, root=ROOT, flush_s=FLUSH_S):
        self.root, self.flush_s = Path(root), flush_s
        self.buf = {k: [] for k in SCHEMAS}
        self.last_flush = time.monotonic()

    def add(self, rows):
        for k, v in rows.items():
            self.buf[k].extend(v)

    def event(self, kind, detail=""):
        log.info(f"{kind} {detail}")
        self.buf["events"].append({"recv_ns": time.time_ns(), "kind": kind, "detail": detail})

    def take(self):
        """Svuota il buffer e ritorna i dati da scrivere (lo swap è istantaneo: il loop non si blocca)."""
        out, self.buf = self.buf, {k: [] for k in SCHEMAS}
        self.last_flush = time.monotonic()
        return out

    def due(self):
        return time.monotonic() - self.last_flush >= self.flush_s

    def write(self, data):
        for name, rows in data.items():
            if not rows:
                continue
            first = rows[0]["recv_ns"]
            day = datetime.fromtimestamp(first / 1e9, timezone.utc).strftime("%Y-%m-%d")
            path = self.root / name / f"date={day}" / f"part-{first}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), tmp, compression="zstd")
            tmp.rename(path)                # un file è visibile solo quando è completo


async def record(products, sink):
    loop = asyncio.get_running_loop()
    backoff = 1
    while True:
        seqs = SequenceChecker()
        try:
            async with websockets.connect(URL, max_size=None, ping_interval=20) as ws:
                for ch in CHANNELS:
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": products, "channel": ch}))
                sink.event("connect", ",".join(products))
                async for raw in ws:
                    recv_ns = time.time_ns()
                    msg = json.loads(raw)
                    if msg.get("type") == "error":
                        raise RuntimeError(msg)
                    seqs.check(msg["sequence_num"])
                    sink.add(normalize(msg, recv_ns))
                    backoff = 1
                    if sink.due():
                        await loop.run_in_executor(None, sink.write, sink.take())
        except asyncio.CancelledError:
            raise
        except SequenceGap as e:
            sink.event("gap", str(e))
        except Exception as e:
            sink.event("disconnect", repr(e)[:500])
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def main(products):
    sink = ParquetSink()
    # systemd ferma il servizio con SIGTERM: lo trasformo in cancellazione, così il buffer viene scritto
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        await record(products, sink)
    finally:
        sink.event("stop")
        sink.write(sink.take())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler("l2_recorder.log"), logging.StreamHandler()])
    try:
        asyncio.run(main(sys.argv[1:] or PRODUCTS))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
