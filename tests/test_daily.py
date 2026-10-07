import numpy as np
import pandas as pd
import pytest

from daily import strategies as st

IDX = pd.bdate_range("2016-01-01", "2019-12-31")


def gbm(drift, vol, seed, idx=IDX):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift / 252, vol / np.sqrt(252), len(idx))
    return pd.Series(100 * np.cumprod(1 + r), index=idx)


@pytest.fixture(scope="module")
def prices():
    return pd.DataFrame({"UP": gbm(0.30, 0.10, 1), "DOWN": gbm(-0.30, 0.10, 2), "VOLATILE": gbm(0.30, 0.40, 3)})


@pytest.fixture(scope="module")
def cash():
    return pd.Series(100 * 1.0001 ** np.arange(len(IDX)), index=IDX)


def test_month_ends():
    idx = pd.bdate_range("2020-01-27", "2020-03-03")
    assert list(st.month_ends(idx)) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"),
                                        pd.Timestamp("2020-03-03")]


def test_tsmom_signs_and_inverse_vol_sizing(prices, cash):
    w = st.tsmom(prices, cash)
    last = w.iloc[-1]
    assert last.UP > 0 and last.DOWN < 0 and last.VOLATILE > 0
    assert last.UP == pytest.approx(4 * last.VOLATILE, rel=0.35)       # vol 10% contro 40%: peso ~4 volte
    assert (w.abs().sum(axis=1) <= 2.0 + 1e-9).all()


def test_tsmom_leverage_cap(prices, cash):
    w = st.tsmom(prices, cash, asset_vol=5.0, max_gross=2.0)
    assert w.abs().sum(axis=1).max() == pytest.approx(2.0)


@pytest.mark.parametrize("fn", [
    lambda p, c: st.tsmom(p, c),
    lambda p, c: st.xsmom(p, top=1),
    lambda p, c: st.vol_target(p[["UP"]]),
])
def test_no_lookahead(prices, cash, fn):
    """Cambiare i prezzi dopo una data non cambia i pesi decisi fino a quella data."""
    cut = pd.Timestamp("2018-06-29")
    w1 = fn(prices, cash)
    moved = prices.copy()
    after = moved.index > cut
    moved.loc[after, "DOWN"] *= np.linspace(1, 20, after.sum())      # solo un asset: cambia anche la classifica
    moved.loc[after, "UP"] *= np.linspace(1, 0.5, after.sum())
    w2 = fn(moved, cash)
    pd.testing.assert_frame_equal(w1.loc[:cut], w2.loc[:cut])
    assert not w1.loc[cut:].iloc[1:].equals(w2.loc[cut:].iloc[1:])


def test_xsmom_skips_last_month():
    """Un asset con un salto solo nell'ultimo mese non viene scelto: conta il rendimento da t-12 a t-1."""
    idx = pd.bdate_range("2016-01-01", "2017-06-30")
    steady = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    flat = pd.Series(100.0, index=idx)
    jump = flat.copy()
    jump.iloc[-10:] = 200.0
    w = st.xsmom(pd.DataFrame({"STEADY": steady, "FLAT": flat, "JUMP": jump}), top=1)
    assert w.iloc[-1].to_dict() == {"STEADY": 1.0, "FLAT": 0.0, "JUMP": 0.0}


def test_vol_target_inverse_to_vol_and_capped():
    idx = pd.bdate_range("2016-01-01", "2016-12-31")
    calm, wild = gbm(0.1, 0.05, 4, idx), gbm(0.1, 0.60, 5, idx)
    w_calm = st.vol_target(calm.to_frame("X"), target_vol=0.15, max_lev=1.5)
    w_wild = st.vol_target(wild.to_frame("X"), target_vol=0.15, max_lev=1.5)
    assert (w_calm.X == 1.5).all()                                      # vol 5%: vorrebbe 3x, cap a 1.5
    assert w_wild.X.median() == pytest.approx(0.25, rel=0.3)            # 15% / 60%
