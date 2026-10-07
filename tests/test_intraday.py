import numpy as np
import pandas as pd
import pytest

from intraday import momentum as im

TZ = "America/New_York"


def session(day, close_time="16:00"):
    return pd.date_range(f"{day} 09:30", f"{day} {close_time}", freq="5min", inclusive="left", tz=TZ)


def synthetic(days=300, k=0.5, seed=0):
    """Barre da 5 min in cui il rendimento dell'ultima mezz'ora è k * r1 + rumore: relazione nota."""
    rng = np.random.default_rng(seed)
    idx, px, prices, p = [], [], [], 100.0
    for d in pd.bdate_range("2024-01-02", periods=days):
        bars = session(d.date())
        r1 = rng.normal(0, 0.005)
        path = []
        for t in bars:
            if t.time() == pd.Timestamp("09:55").time():
                p *= 1 + r1                                   # tutta la prima mezz'ora in una barra
            elif t.time() >= pd.Timestamp("15:30").time():
                p *= 1 + (k * r1 + rng.normal(0, 0.0005)) / 6     # ultima mezz'ora: 6 barre
            else:
                p *= 1 + rng.normal(0, 0.0003)
            path.append(p)
        idx.extend(bars)
        prices.extend(path)
    return pd.Series(prices, index=pd.DatetimeIndex(idx))


@pytest.fixture(scope="module")
def closes():
    return synthetic()


def test_price_at_uses_bar_ending_at_time(closes):
    p = im.price_at(closes, "10:00")
    day = p.index[0]
    bar = closes[(closes.index.normalize().tz_localize(None) == day) & (closes.index.time == pd.Timestamp("09:55").time())]
    assert p.iloc[0] == bar.iloc[0]


def test_features_recover_known_relation(closes):
    f = im.daily_features(closes)
    reg = im.predictive_regression(f)
    assert reg["beta"] == pytest.approx(0.5, abs=0.1)
    assert reg["t_nw"] > 5


def test_strategy_profits_from_known_relation(closes):
    f = im.daily_features(closes)
    res = im.pnl(f, im.signal(f, "r1"), cost_bps=1)
    assert res.gross.sum() > 0 and res.net.sum() < res.gross.sum()
    assert res.trade.sum() == len(f)


def test_signal_does_not_use_last_half_hour(closes):
    """Cambiare i prezzi dopo le 15:30 di un giorno cambia il suo r13 ma non il suo segnale. (Il giorno dopo può
    cambiare: la chiusura di ieri entra, legittimamente, nel suo r1.)"""
    day = pd.Timestamp("2024-03-14")
    moved = closes.copy()
    late = (moved.index.normalize().tz_localize(None) == day) & (moved.index.time >= pd.Timestamp("15:30").time())
    moved[late] *= 1.01
    f1, f2 = im.daily_features(closes), im.daily_features(moved)
    assert f1.loc[day, "r13"] != f2.loc[day, "r13"]
    for rule in ("r1", "r1_r12"):
        s1, s2 = im.signal(f1, rule), im.signal(f2, rule)
        pd.testing.assert_series_equal(s1.loc[:day], s2.loc[:day])


def test_r1_r12_needs_agreement():
    f = pd.DataFrame({"r1": [0.01, 0.01, -0.01], "r12": [0.002, -0.002, -0.001]})
    assert im.signal(f, "r1_r12").tolist() == [1, 0, -1]


def test_half_day_previous_close():
    """Dopo una mezza giornata la chiusura precedente è quella delle 13:00, non quella di due giorni prima."""
    idx = session("2024-11-27").append(session("2024-11-29", "13:00")).append(session("2024-12-02"))
    closes = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
    f = im.daily_features(closes)
    half_day_last = closes[closes.index.normalize().tz_localize(None) == pd.Timestamp("2024-11-29")].iloc[-1]
    assert pd.Timestamp("2024-11-29") not in f.index                   # niente barra delle 15:30: scartato
    assert f.loc[pd.Timestamp("2024-12-02"), "prev_close"] == half_day_last


def test_costs_round_trip():
    f = pd.DataFrame({"r13": [0.001, 0.001], "r13_late": [0.0, 0.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    res = im.pnl(f, pd.Series([1.0, 0.0], index=f.index), cost_bps=2)
    assert res.net.tolist() == pytest.approx([0.001 - 4e-4, 0.0])


def test_high_vol_days_uses_only_past(closes):
    f = im.daily_features(closes)
    day = f.index[200]
    h1 = im.high_vol_days(f, min_periods=100)
    g = f.copy()
    g.loc[g.index >= day, "r1"] *= 10                                  # da `day` in poi la volatilità esplode
    h2 = im.high_vol_days(g, min_periods=100)
    pd.testing.assert_series_equal(h1.loc[:day], h2.loc[:day])
    assert h1.iloc[:100].isna().all()
    assert not h1.loc[day:].iloc[1:].equals(h2.loc[day:].iloc[1:])
