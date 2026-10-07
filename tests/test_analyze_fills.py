import pytest

from analyze_fills import parse

LOG = """\
2026-10-08 10:15:46,120 INFO barra 10:00 z=+2.31 stato 0->-1 C: 0->-54  WFC: 0->60
2026-10-08 10:15:46,301 INFO ORDINE sell 54 C id=abc
2026-10-08 10:15:47,512 INFO FILL C 54 @ 91.9800 chiusura barra 92.0000 slippage +2.2 bps
2026-10-08 10:15:47,933 INFO FILL WFC 60 @ 83.1200 chiusura barra 83.1000 slippage +2.4 bps
2026-10-08 10:30:46,002 WARNING Barra 10:15 senza scambi IEX su una gamba: prezzo stantio, salto la barra
"""


def test_parse_extracts_only_fills():
    df = parse(LOG.splitlines())
    assert df.symbol.tolist() == ["C", "WFC"]
    assert df.slippage_bps.tolist() == [2.2, 2.4]
    assert df.qty.tolist() == [54, 60]
    assert df.time.iloc[0].microsecond == 512000


def test_logged_slippage_matches_definition():
    """Lo slippage loggato da live_pairs ha segno: positivo = costo (venduto sotto / comprato sopra la chiusura)."""
    df = parse(LOG.splitlines())
    sell = df.iloc[0]
    assert -(sell.fill - sell.ref) / sell.ref * 1e4 == pytest.approx(sell.slippage_bps, abs=0.05)
    buy = df.iloc[1]
    assert (buy.fill - buy.ref) / buy.ref * 1e4 == pytest.approx(buy.slippage_bps, abs=0.05)
