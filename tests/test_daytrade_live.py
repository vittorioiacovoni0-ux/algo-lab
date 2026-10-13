"""Bot live di day trading senza rete: broker finto, barre sintetiche."""
import numpy as np
import pandas as pd
import pytest

from daytrade import engine as en
from daytrade import live
from daytrade.strategies import NoiseBoundary
from test_daytrade import trending_days


class Recorder(NoiseBoundary):
    """NoiseBoundary che registra le decisioni prese dentro il backtest."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def on_bar(self, bars, side):
        intent = super().on_bar(bars, side)
        if self.is_decision_time(bars.minute[-1] + 1):
            self.calls.append((bars.date, bars.minute[-1] + 1, side, intent))
        return intent


@pytest.fixture(scope="module")
def days():
    rng = np.random.default_rng(3)
    frames = [trending_days(1, drift_bp=float(rng.normal(0, 3)), seed=i, start=d)
              for i, d in enumerate(pd.bdate_range("2024-01-02", periods=40).strftime("%Y-%m-%d"))]
    px = 100.0
    for f in frames:                                   # giorni contigui: ogni giorno apre dove chiude il precedente
        f[["open", "high", "low", "close"]] *= px / f.open.iloc[0]
        px = f.close.iloc[-1]
    return pd.concat(frames)


def test_live_decisions_match_backtest(days):
    """Stessa storia, stesse barre, stessa posizione: il percorso live decide esattamente come il backtest."""
    rec = Recorder()
    en.run(days, rec, costs=en.Costs(slip_bps=0))
    split = en.split_days(days)
    assert len(rec.calls) > 50 and any(c[3] is not None for c in rec.calls)
    for date, end, side, intent in rec.calls:
        k = next(i for i, d in enumerate(split) if d.date == date)
        s = NoiseBoundary()
        s.start_day(None, split[:k])                   # il live ricostruisce la strategia ogni mattina
        day = split[k]
        i = int(np.flatnonzero(day.minute + 1 == end)[0])
        assert s.on_bar(day.upto(i), side) == intent


def test_decision_times_regular_and_half_day():
    s = NoiseBoundary()
    s.sigma = np.zeros(1)
    day = pd.Timestamp("2026-11-27", tz=live.TZ)
    full = live.decision_times(s, day, day + pd.Timedelta(hours=16))
    assert [f"{t:%H:%M}" for t in full][:2] == ["10:00", "10:30"] and f"{full[-1]:%H:%M}" == "15:30"
    assert len(full) == 12
    half = live.decision_times(s, day, day + pd.Timedelta(hours=13))
    assert f"{half[-1]:%H:%M}" == "12:30"


class FakeBroker:
    def __init__(self, qty=0, equity=100_000.0, fill=lambda q: 100.0):
        self.qty, self.eq, self.fill = qty, equity, fill
        self.log = []

    def position(self):
        return self.qty

    def equity(self):
        return self.eq

    def cancel_open_orders(self):
        self.log.append(("cancel",))

    def market(self, qty):
        px = self.fill(qty)
        self.log.append(("market", qty))
        if px is not None:
            self.qty += qty
        return px

    def stop(self, qty, price):
        self.log.append(("stop", qty, round(price, 2)))

    def market_on_close(self, qty):
        self.log.append(("moc", qty))


def trader_with(broker, lev=2.0):
    t = live.Trader(broker)
    t.acct0, t.eq0, t.halted = broker.equity(), min(broker.equity(), live.CAPITAL), False
    t.strategy = NoiseBoundary()
    t.strategy.lev = lev
    return t


def test_entry_places_emergency_stop():
    b = FakeBroker()
    t = trader_with(b)
    t.rebalance(0, 2000, 100.0)
    # stop alla distanza che fa perdere il 3% dell'equity: 3000$ / 2000 azioni = 1.50$
    assert b.log == [("cancel",), ("market", 2000), ("stop", -2000, 98.5)]


def test_flip_closes_then_opens_after_cancelling_stop():
    b = FakeBroker(qty=2000)
    t = trader_with(b)
    t.rebalance(2000, -2000, 100.0)
    assert b.log == [("cancel",), ("market", -2000), ("market", -2000), ("stop", 2000, 101.5)]


def test_failed_close_does_not_open_opposite_side():
    b = FakeBroker(qty=2000, fill=lambda q: None)
    t = trader_with(b)
    t.rebalance(2000, -2000, 100.0)
    assert b.log == [("cancel",), ("market", -2000)]


def test_dry_run_sends_nothing():
    b = FakeBroker()
    t = trader_with(b)
    t.dry_run = True
    t.rebalance(0, 2000, 100.0)
    t.flatten_eod()
    assert b.log == []


def test_daily_loss_limit_closes_and_halts():
    b = FakeBroker(qty=2000)
    t = trader_with(b)
    b.eq = 96_000.0                                    # -4% sull'equity di inizio giornata
    t.on_decision(None, 600)
    assert t.halted and ("market", -2000) in b.log
    b.log.clear()
    t.on_decision(None, 630)
    assert b.log == []


def test_missing_bar_skips_decision(days):
    b = FakeBroker()
    t = trader_with(b)
    day = en.split_days(days)[20]
    t.on_decision(day.upto(10), 600)                   # barra delle 9:40, non delle 10:00
    assert b.log == []


def test_flatten_eod_uses_market_on_close():
    b = FakeBroker(qty=-1500)
    t = trader_with(b)
    t.flatten_eod()
    assert b.log == [("cancel",), ("moc", 1500)]
