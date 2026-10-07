import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest

from l2_recorder import ParquetSink, SequenceChecker, SequenceGap, normalize
from orderbook import OrderBook, replay

T = "2026-10-07T21:48:18.002696588Z"


def l2_msg(seq, typ, updates, product="BTC-USD"):
    return {"channel": "l2_data", "timestamp": T, "sequence_num": seq,
            "events": [{"type": typ, "product_id": product,
                        "updates": [{"side": s, "event_time": "2026-10-07T21:48:17.682037Z",
                                     "price_level": str(p), "new_quantity": str(q)} for s, p, q in updates]}]}


TRADE_MSG = {"channel": "market_trades", "timestamp": T, "sequence_num": 16,
             "events": [{"type": "update", "trades": [
                 {"product_id": "BTC-USD", "trade_id": "1103548465", "price": "83153.66",
                  "size": "0.00014823", "time": "2026-10-07T21:48:18.425694Z", "side": "SELL"}]}]}


def test_normalize_l2():
    rows = normalize(l2_msg(5, "update", [("bid", 100.0, 1.5), ("offer", 101.0, 0)]), recv_ns=123)
    assert rows["trades"] == []
    b = rows["book"]
    assert [(r["side"], r["price"], r["qty"]) for r in b] == [("bid", 100.0, 1.5), ("offer", 101.0, 0.0)]
    assert all(r["seq"] == 5 and r["recv_ns"] == 123 and r["type"] == "update" for r in b)
    assert b[0]["exch_ts"] == pd.Timestamp(T)               # precisione al nanosecondo conservata


def test_normalize_trades_and_ignores_other_channels():
    rows = normalize(TRADE_MSG, recv_ns=1)
    (t,) = rows["trades"]
    assert (t["trade_id"], t["price"], t["size"], t["side"]) == (1103548465, 83153.66, 0.00014823, "SELL")
    hb = {"channel": "heartbeats", "timestamp": T, "sequence_num": 3, "events": [{"heartbeat_counter": 1}]}
    assert normalize(hb, 1) == {"book": [], "trades": []}


def test_sequence_checker_detects_gap_and_resets():
    c = SequenceChecker()
    for s in (0, 1, 2):
        c.check(s)
    with pytest.raises(SequenceGap):
        c.check(4)
    c.check(0)                                               # nuova connessione: si riparte da capo


def test_orderbook_apply_and_top():
    b = OrderBook()
    for side, p, q in [("bid", 99, 1), ("bid", 100, 2), ("offer", 101, 3), ("offer", 102, 4)]:
        b.apply(side, p, q)
    assert b.top() == (100, 2, 101, 3)
    b.apply("bid", 100, 0)                                   # quantità 0 = livello rimosso
    b.apply("offer", 101, 7)                                 # quantità = nuovo totale, non delta
    assert b.top() == (99, 1, 101, 7)
    assert b.depth(2) == (1, 11)
    b.clear()
    assert all(np.isnan(x) for x in b.top())


def test_replay_resets_on_snapshot():
    msgs = [l2_msg(0, "snapshot", [("bid", 100, 1), ("offer", 101, 1)]),
            l2_msg(1, "update", [("bid", 100.5, 2)]),
            l2_msg(7, "snapshot", [("bid", 90, 1), ("offer", 91, 1)])]    # dopo un buco: nuovo snapshot
    l2 = pd.DataFrame([r for i, m in enumerate(msgs) for r in normalize(m, i)["book"]])
    top = replay(l2)
    assert top[["bid", "ask"]].values.tolist() == [[100, 101], [100.5, 101], [90, 91]]


def test_sink_roundtrip(tmp_path):
    sink = ParquetSink(root=tmp_path, flush_s=0)
    sink.add(normalize(l2_msg(0, "snapshot", [("bid", 100, 1), ("offer", 101, 1)]), 1_791_000_000_000_000_000))
    sink.add(normalize(TRADE_MSG, 1_791_000_000_000_000_001))
    sink.event("connect", "BTC-USD")
    sink.write(sink.take())
    book = ds.dataset(tmp_path / "book", partitioning="hive").to_table().to_pandas()
    trades = ds.dataset(tmp_path / "trades", partitioning="hive").to_table().to_pandas()
    assert len(book) == 2 and len(trades) == 1
    assert book.exch_ts.iloc[0] == pd.Timestamp(T)
    assert not list(tmp_path.rglob("*.tmp"))
    assert sink.take() == {"book": [], "trades": [], "events": []}
