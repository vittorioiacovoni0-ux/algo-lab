import numpy as np
import pandas as pd
import pytest

import backtest as bt

IDX = pd.bdate_range("2020-01-01", periods=6)


def frame(data, cols=("A", "B")):
    return pd.DataFrame(data, index=IDX[: len(data)], columns=list(cols))


def test_lag_controls_first_return_earned():
    rets = frame([[0.0, 0], [0.10, 0], [0.20, 0], [0.30, 0]])
    tgt = frame([[1.0, 0]])                                  # decisione a fine giorno 0
    r0 = bt.run(tgt, rets, cost_bps=0, lag=0)
    r1 = bt.run(tgt, rets, cost_bps=0, lag=1)
    assert r0.gross.tolist() == pytest.approx([0, 0.10, 0.20, 0.30])
    assert r1.gross.tolist() == pytest.approx([0, 0, 0.20, 0.30])    # eseguo a fine giorno 1


def test_weights_drift_like_buy_and_hold():
    rets = frame([[0, 0], [1.0, 0], [0, 0], [0.5, -0.5]])
    tgt = frame([[0.5, 0.5]])
    r = bt.run(tgt, rets, cost_bps=0, lag=0)
    # buy & hold: 0.5 * 2 * 1.5 + 0.5 * 1 * 0.5 = 1.75
    assert np.prod(1 + r.gross) == pytest.approx(1.75)
    assert r.turnover.sum() == pytest.approx(1.0)                       # solo l'acquisto iniziale


def test_rebalance_turnover_is_from_drifted_weights():
    rets = frame([[0, 0], [1.0, 0], [0, 0]])
    tgt = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=[IDX[0], IDX[1]], columns=["A", "B"])
    r = bt.run(tgt, rets, cost_bps=0, lag=0)
    # dopo il +100% di A i pesi sono 2/3 e 1/3: riportarli a 1/2 costa |1/6| + |1/6|
    assert r.turnover.iloc[1] == pytest.approx(1 / 3)


def test_costs_and_borrow():
    rets = frame([[0, 0], [0, 0], [0, 0]])
    tgt = frame([[1.0, -1.0]])
    r = bt.run(tgt, rets, cost_bps=10, lag=0, borrow_bps=252, periods_per_year=252)
    assert r.net.iloc[0] == pytest.approx(-2 * 10 / 1e4)               # turnover 2 all'ingresso
    assert r.net.iloc[1] == pytest.approx(-1 / 1e4)                     # 252 bps annui su short di 1, per giorno


def test_excess_returns_over_cash():
    rf = pd.Series(0.001, index=IDX[:3])
    rets = frame([[0.001, 0], [0.001, 0], [0.001, 0]])
    r = bt.run(frame([[1.0, 0]]), rets, rf=rf, cost_bps=0, lag=0)
    assert r.gross.abs().max() == pytest.approx(0)                     # rende quanto il cash: excess nullo
    m = bt.metrics(r, periods_per_year=252)
    assert m["cagr_%"] == pytest.approx(100 * (1.001 ** 252 - 1), rel=1e-6)


def test_future_targets_do_not_affect_past():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=50)
    rets = pd.DataFrame(rng.normal(0, 0.01, (50, 2)), index=idx, columns=["A", "B"])
    t1 = pd.DataFrame([[1.0, 0], [0, 1.0]], index=idx[[5, 30]], columns=["A", "B"])
    t2 = t1.copy()
    t2.iloc[1] = [-1.0, 0.5]
    r1, r2 = bt.run(t1, rets, lag=1), bt.run(t2, rets, lag=1)
    pd.testing.assert_frame_equal(r1.iloc[:31], r2.iloc[:31])           # niente cambia prima dell'esecuzione (31)
    pd.testing.assert_series_equal(r1.gross.iloc[:32], r2.gross.iloc[:32])  # il 31 cambia solo il costo
    assert r1.net.iloc[31] != r2.net.iloc[31]
    assert not r1.gross.iloc[32:].equals(r2.gross.iloc[32:])


def test_breakeven_cost():
    rets = frame([[0, 0], [0.01, 0], [0, 0]])
    r = bt.run(frame([[1.0, 0]]), rets, cost_bps=0, lag=0)
    assert bt.metrics(r)["costo_pareggio_bps"] == pytest.approx(100)    # 1% di lordo su turnover 1
