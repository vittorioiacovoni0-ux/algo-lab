import numpy as np
import pandas as pd
import pytest

import walkforward as wf
from test_research import BETA, intraday_index, ou

WIN = dict(formation_days=30, trading_days=10)


@pytest.fixture(scope="module")
def synthetic():
    """A e B cointegrate con beta noto, C, D, E random walk indipendenti."""
    rng = np.random.default_rng(7)
    idx = intraday_index(90)
    n = len(idx)
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.002, n))
    log_a = 0.5 + BETA * log_b + ou(n, 0.05, 0.002, rng)
    others = np.log([80, 30, 120]) + np.cumsum(rng.normal(0, 0.002, (n, 3)), axis=0)
    return pd.DataFrame(np.exp(np.c_[log_a, log_b, others]), index=idx, columns=list("ABCDE"))


def test_windows_formation_precedes_trading_and_trading_windows_tile(synthetic):
    w = wf.windows(synthetic.index, **WIN)
    assert len(w) == 6
    for f0, t0, t1 in w:
        assert f0 < t0 < t1
    for (_, _, t1), (_, t0_next, _) in zip(w, w[1:]):
        assert t1 == t0_next                       # finestre di trading contigue, senza buchi né sovrapposizioni


def test_no_lookahead(synthetic):
    """Cambiare i prezzi dopo una finestra non cambia i risultati di quella finestra né delle precedenti."""
    w = wf.windows(synthetic.index, **WIN)
    cut = w[2][2]
    r1 = wf.walk_forward(synthetic, **WIN)
    moved = synthetic.copy()
    moved.loc[moved.index >= cut, "A"] *= 1.5
    r2 = wf.walk_forward(moved, **WIN)
    for k in ("coint", "tutte"):
        pd.testing.assert_frame_equal(r1[k]["rows"].iloc[:3], r2[k]["rows"].iloc[:3])
        before = r1[k]["series"].index < cut
        pd.testing.assert_frame_equal(r1[k]["series"][before], r2[k]["series"][before])


def test_out_of_sample_series_has_no_duplicate_bars(synthetic):
    r = wf.walk_forward(synthetic, **WIN)
    for k in ("coint", "tutte"):
        assert r[k]["series"].index.is_unique


def test_true_pair_beats_null(synthetic):
    r = wf.walk_forward(synthetic, **WIN)
    s_coint = wf.summary(r["coint"]["series"], r["coint"]["rows"])
    s_all = wf.summary(r["tutte"]["series"], r["tutte"]["rows"])
    assert s_coint["ret_lordo_%"] > 0
    assert s_coint["sharpe"] > s_all["sharpe"]
