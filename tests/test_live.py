"""Test del bot live senza rete: client Alpaca, dati e orologio sono sostituiti da fake."""
import importlib
import json
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from alpaca.trading.enums import OrderSide

TZ = "America/New_York"
BETA = 1.2


class FakeTrading:
    def __init__(self, positions):
        self.positions = positions
        self.orders = []

    def get_all_positions(self):
        return [SimpleNamespace(symbol=s, qty=str(q)) for s, q in self.positions.items()]

    def submit_order(self, req):
        o = SimpleNamespace(id=len(self.orders), symbol=req.symbol, qty=req.qty, side=req.side)
        self.orders.append(o)
        return o

    def get_order_by_id(self, oid):
        o = self.orders[oid]
        return SimpleNamespace(**vars(o), filled_qty=o.qty, filled_avg_price=100.0, status="filled")

    def signed(self):
        return {o.symbol: (o.qty if o.side == OrderSide.BUY else -o.qty) for o in self.orders}


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_API_SECRET", "test")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pair_config.json").write_text(json.dumps(
        {"a": "AAA", "b": "BBB", "beta": BETA, "minutes": 15, "window": 100,
         "z_in": 2.0, "z_out": 0.5, "z_stop": 4.0}))
    sys.modules.pop("live_pairs", None)
    mod = importlib.import_module("live_pairs")
    yield mod
    sys.modules.pop("live_pairs", None)


def closes_until(last_close, last_z_shock=0.0, nan_last_b=False):
    """Barre da 15 min che finiscono con la barra chiusa a `last_close`; spread stazionario attorno a 0."""
    starts = pd.date_range(end=last_close - pd.Timedelta(minutes=15), periods=150, freq="15min")
    rng = np.random.default_rng(0)
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.001, len(starts)))
    noise = rng.normal(0, 0.001, len(starts))
    noise[-1] = last_z_shock * 0.001
    df = pd.DataFrame({"AAA": np.exp(0.3 + BETA * log_b + noise), "BBB": np.exp(log_b)}, index=starts)
    if nan_last_b:
        df.iloc[-1, 1] = np.nan
    return df


def run_step(live, monkeypatch, now, positions, **closes_kw):
    now = pd.Timestamp(now, tz=TZ).to_pydatetime()
    fake = FakeTrading(positions)
    monkeypatch.setattr(live, "trading", fake)
    last_close = live.last_bar_close(now)
    monkeypatch.setattr(live, "get_closes", lambda *a, **k: closes_until(pd.Timestamp(last_close), **closes_kw))
    monkeypatch.setattr(live, "datetime", type("FakeDT", (datetime,), {"now": classmethod(lambda cls, tz=None: now)}))
    live.step(SimpleNamespace(next_close=pd.Timestamp(now.date().isoformat() + " 16:00", tz=TZ)))
    return fake.signed()


@pytest.mark.parametrize("qa,qb,expected", [
    (0, 0, 0), (10, -12, 1), (-10, 12, -1),
    (10, 0, None), (0, -12, None), (10, 12, None),
])
def test_state_from_qty(live, qa, qb, expected):
    assert live.state_from_qty(qa, qb) == expected


def test_entry_short_spread_on_high_z(live, monkeypatch):
    orders = run_step(live, monkeypatch, "2026-10-07 14:00:45", {}, last_z_shock=3.5)
    assert orders["AAA"] < 0 and orders["BBB"] > 0


def test_flat_market_stays_flat(live, monkeypatch):
    assert run_step(live, monkeypatch, "2026-10-07 14:00:45", {}) == {}


def test_inconsistent_legs_get_closed(live, monkeypatch):
    orders = run_step(live, monkeypatch, "2026-10-07 14:00:45", {"AAA": 10})
    assert orders == {"AAA": -10}


def test_stale_bar_skipped_intraday(live, monkeypatch):
    orders = run_step(live, monkeypatch, "2026-10-07 14:00:45", {}, last_z_shock=3.5, nan_last_b=True)
    assert orders == {}


def test_end_of_day_flattens_even_with_stale_bar(live, monkeypatch):
    orders = run_step(live, monkeypatch, "2026-10-07 15:45:45", {"AAA": -10, "BBB": 12}, nan_last_b=True)
    assert orders == {"AAA": 10, "BBB": -12}


def test_no_entry_in_last_bar(live, monkeypatch):
    assert run_step(live, monkeypatch, "2026-10-07 15:45:45", {}, last_z_shock=3.5) == {}
