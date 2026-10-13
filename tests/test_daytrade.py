import numpy as np
import pandas as pd
import pytest

from daytrade import engine as en
from daytrade.engine import Costs, Intent, Strategy
from daytrade.strategies import NoiseBoundary, OpeningRangeBreakout

TZ = "America/New_York"
FREE = Costs(slip_bps=0, sec_bps=0, taf_per_share=0)


def day_bars(day, closes, opens=None, highs=None, lows=None, start="09:30"):
    closes = np.asarray(closes, float)
    opens = closes if opens is None else np.asarray(opens, float)
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="1min", tz=TZ)
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) if highs is None else highs,
                         "low": np.minimum(opens, closes) if lows is None else lows,
                         "close": closes, "volume": 1000.0}, index=idx)


class Script(Strategy):
    """Strategia di test: emette intent prefissati a certi indici di barra."""

    def __init__(self, plan):
        self.plan = plan

    def on_bar(self, bars, side):
        return self.plan.get(len(bars.close) - 1)


def test_execution_at_next_open_and_moc_exit():
    df = day_bars("2024-03-01", closes=[100, 101, 102, 103], opens=[100, 100.5, 101.5, 102.5])
    res = en.run(df, Script({0: Intent(1, 1.0)}), equity=10_000, costs=FREE)
    (t,) = res.trades.itertuples()
    assert t.entry == 100.5 and t.exit == 103 and t.reason == "chiusura"     # ingresso all'apertura di barra 1
    assert t.shares == round(10_000 / 100.5)
    assert res.daily.gross.iloc[0] == pytest.approx(t.shares * (103 - 100.5) / 10_000)


def test_stop_gap_fills_at_open_and_stop_beats_target_in_same_bar():
    df = day_bars("2024-03-01", closes=[100, 100, 95, 95], opens=[100, 100, 96, 95], lows=[100, 100, 94, 95],
                  highs=[100, 100, 120, 95])
    res = en.run(df, Script({0: Intent(1, 1.0, stop=98, take=110)}), equity=10_000, costs=FREE)
    (t,) = res.trades.itertuples()
    assert t.reason == "stop" and t.exit == 96                               # apertura sotto lo stop: eseguito a 96


def test_take_profit():
    df = day_bars("2024-03-01", closes=[100, 100, 104, 104], highs=[100, 100, 106, 104])
    res = en.run(df, Script({0: Intent(1, 1.0, stop=90, take=105)}), equity=10_000, costs=FREE)
    assert res.trades.reason.tolist() == ["target"] and res.trades.exit.iloc[0] == 105


def test_short_and_flip():
    df = day_bars("2024-03-01", closes=[100, 100, 98, 99, 101])
    res = en.run(df, Script({0: Intent(-1, 1.0), 2: Intent(1, 1.0)}), equity=10_000, costs=FREE)
    assert res.trades.side.tolist() == [-1, 1]
    assert res.trades.exit.iloc[0] == res.trades.entry.iloc[1] == 99         # inversione in un'unica esecuzione
    assert res.daily.trade.iloc[0] == 2


def test_leverage_cap_and_costs():
    df = day_bars("2024-03-01", closes=[100, 100, 100])
    res = en.run(df, Script({0: Intent(1, 10.0)}), equity=10_000, max_lev=4,
                 costs=Costs(slip_bps=1, sec_bps=0, taf_per_share=0))
    assert res.trades.shares.iloc[0] == 400                                  # 10x richiesti, 4x concessi
    assert res.daily.net.iloc[0] == pytest.approx(-2 * 400 * 100 * 1e-4 / 10_000)


def test_daily_loss_limit_stops_trading():
    df = day_bars("2024-03-01", closes=[100, 100, 96, 95, 110, 111])
    res = en.run(df, Script({0: Intent(1, 1.0), 4: Intent(1, 1.0)}), equity=10_000, max_daily_loss=0.03,
                 costs=FREE)
    assert res.trades.reason.tolist() == ["limite perdita"]                  # -4% a fine barra 2 -> esce a barra 3
    assert res.trades.exit.iloc[0] == 95 and len(res.trades) == 1            # niente nuovi ingressi dopo


def test_strategy_cannot_see_future():
    class Spy(Strategy):
        def __init__(self):
            self.seen = []

        def on_bar(self, bars, side):
            self.seen.append(len(bars.close))
            assert bars.close[-1] == bars.close[len(bars.close) - 1]

    df = day_bars("2024-03-01", closes=[1, 2, 3, 4])
    s = Spy()
    en.run(df, s)
    assert s.seen == [1, 2, 3]                                               # mai sull'ultima barra, mai oltre


def trending_days(n_days, drift_bp, seed=0, start="2024-01-02"):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=n_days):
        rets = rng.normal(drift_bp / 1e4, 0.0005, 390)
        closes = px * np.cumprod(1 + rets)
        opens = np.r_[px, closes[:-1]]
        frames.append(day_bars(d.date(), closes, opens=opens))
        px = closes[-1]
    return pd.concat(frames)


def test_noise_boundary_long_on_up_trend_days_and_uses_only_past():
    df = trending_days(30, drift_bp=0)
    up = trending_days(1, drift_bp=3, seed=99, start="2024-02-13")
    up = up * (df.close.iloc[-1] / up.open.iloc[0])
    up["volume"] = 1000.0
    res = en.run(pd.concat([df, up]), NoiseBoundary(lookback=14), costs=FREE)
    last = res.trades[res.trades.date == res.daily.index[-1]]
    assert len(last) and (last.side == 1).all()                             # giorno in trend: long
    assert res.trades[res.trades.date < res.daily.index[14]].empty           # niente trade prima di 14 giorni


def test_orb_direction_stop_and_size():
    closes = [100, 100.2, 100.4, 100.6, 101, 101.5, 102]
    df = day_bars("2024-03-01", closes=closes, opens=[100] + closes[:-1], lows=[99.5] + closes[1:])
    res = en.run(df, OpeningRangeBreakout(risk=0.01, max_lev=4), equity=100_000, costs=FREE)
    (t,) = res.trades.itertuples()
    assert t.side == 1 and t.entry == 101                                    # candela rialzista -> long alla 9:35
    # rischio 1% con stop a 99.5 (minimo della candela): leva = 0.01 / (1.5/101) = 0.67
    assert t.shares == round(0.01 / (1.5 / 101) * 100_000 / 101)
