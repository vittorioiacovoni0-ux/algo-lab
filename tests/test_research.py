import numpy as np
import pandas as pd
import pytest

import research_pairs as rp

BETA = 1.5


def intraday_index(days, minutes=15):
    times = pd.timedelta_range("09:30:00", "15:45:00", freq=f"{minutes}min")
    dates = pd.bdate_range("2026-01-05", periods=days)
    return pd.DatetimeIndex([d + t for d in dates for t in times]).tz_localize("America/New_York")


def ou(n, theta, sigma, rng):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] + sigma * rng.normal()
    return x


@pytest.fixture(scope="module")
def synthetic():
    """A e B cointegrate con beta noto, C e D random walk indipendenti."""
    rng = np.random.default_rng(42)
    idx = intraday_index(120)
    n = len(idx)
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.002, n))
    log_a = 0.5 + BETA * log_b + ou(n, 0.05, 0.002, rng)
    log_c = np.log(80) + np.cumsum(rng.normal(0, 0.002, n))
    log_d = np.log(30) + np.cumsum(rng.normal(0, 0.002, n))
    return pd.DataFrame(np.exp(np.c_[log_a, log_b, log_c, log_d]), index=idx, columns=list("ABCD"))


def test_hedge_ratio_recovers_beta(synthetic):
    lg = np.log(synthetic)
    assert rp.hedge_ratio(lg.A, lg.B) == pytest.approx(BETA, abs=0.05)


def test_coint_pvalue(synthetic):
    lg = np.log(synthetic)
    assert rp.coint_pvalue(lg.A, lg.B) < 0.01
    assert rp.coint_pvalue(lg.C, lg.D) > 0.05


def test_session_flat_last_two_bars_of_each_day(synthetic):
    flat = pd.Series(rp.session_flat(synthetic.index, 15), index=synthetic.index)
    times = flat[flat].index.strftime("%H:%M").unique().tolist()
    assert sorted(times) == ["15:30", "15:45"]


def test_backtest_profitable_on_true_pair_and_costs_reduce_return(synthetic):
    s = synthetic
    m = rp.backtest(s.A, s.B, BETA, s.index[0], s.index[-1] + pd.Timedelta(minutes=15))
    assert m["trade"] > 10
    assert m["ret_lordo_%"] > 0
    assert m["ret_%"] < m["ret_lordo_%"]


def test_backtest_no_lookahead(synthetic):
    """Cambiare i prezzi dopo la fine del periodo valutato non cambia il risultato."""
    s = synthetic
    end = s.index[len(s) // 2]
    m1 = rp.backtest(s.A, s.B, BETA, s.index[0], end)
    a2 = s.A.copy()
    a2[a2.index >= end] *= 1.3
    m2 = rp.backtest(a2, s.B, BETA, s.index[0], end)
    assert m1 == m2


def test_backtest_ignores_overnight_gap(synthetic):
    """Un gap in apertura su entrambe le gambe (lo spread non cambia) non tocca il PnL: si è flat la notte."""
    s = synthetic.copy()
    day2 = s.index.normalize().unique()[60]
    gap = s.index >= day2
    gapped = s.copy()
    gapped.loc[gap, "B"] *= 1.10
    gapped.loc[gap, "A"] *= 1.10 ** BETA          # lo spread log resta identico
    end = s.index[-1] + pd.Timedelta(minutes=15)
    m1 = rp.backtest(s.A, s.B, BETA, s.index[0], end)
    m2 = rp.backtest(gapped.A, gapped.B, BETA, s.index[0], end)
    assert m1["trade"] == m2["trade"]
    assert m2["ret_lordo_%"] == pytest.approx(m1["ret_lordo_%"], rel=0.15)   # solo l'effetto scala dopo il gap


def test_run_selects_true_pair(synthetic):
    res, best, test = rp.run(synthetic)
    assert {best.a, best.b} == {"A", "B"}
    assert set(test) >= {"sharpe", "ret_%", "ret_lordo_%", "max_dd_%", "trade"}
