import numpy as np
import pandas as pd
import pytest

from strategy import leg_weights, must_be_flat, next_state, positions, zscore

Z = dict(z_in=2.0, z_out=0.5, z_stop=4.0)


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.3, -0.8])
def test_leg_weights_unit_gross_exposure(beta):
    wa, wb = leg_weights(beta)
    assert abs(wa) + abs(wb) == pytest.approx(1.0)
    assert wa > 0
    assert np.sign(wb) == -np.sign(beta)
    assert wb / wa == pytest.approx(-beta)


def test_zscore_uses_only_past():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=300).cumsum())
    z = zscore(s, 50)
    s2 = s.copy()
    s2.iloc[200:] += 100                       # cambio il futuro
    z2 = zscore(s2, 50)
    pd.testing.assert_series_equal(z.iloc[:200], z2.iloc[:200])
    assert z.iloc[:49].isna().all()


@pytest.mark.parametrize("state,z,expected", [
    (0, 2.5, -1),        # spread alto: short spread
    (0, -2.5, 1),        # spread basso: long spread
    (0, 1.9, 0),         # sotto soglia
    (0, 4.5, 0),         # oltre lo stop: non si entra
    (0, -4.5, 0),
    (1, -1.0, 1),        # ancora lontano dalla media: tengo
    (1, -0.4, 0),        # rientrato: esco
    (-1, 0.4, 0),
    (-1, 1.0, -1),
    (1, -4.0, 0),        # stop
    (-1, 4.2, 0),
    (1, 3.0, 0),         # mai long -> short in un passo
    (-1, -3.0, 0),       # mai short -> long in un passo
    (1, np.nan, 1),      # dato mancante: nessuna decisione
    (0, np.nan, 0),
])
def test_next_state(state, z, expected):
    assert next_state(state, z, **Z) == expected


def test_positions_never_flip_directly():
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(scale=3, size=5000))
    pos = positions(z, **Z)
    assert not ((pos.diff().abs() == 2).any())


def test_positions_flat_overrides_state():
    z = pd.Series([2.5, 2.5, 2.5, 2.5, 2.5])
    flat = [False, False, True, False, False]
    assert positions(z, **Z, flat=flat).tolist() == [-1, -1, 0, -1, -1]


@pytest.mark.parametrize("close,expected", [("15:30", False), ("15:45", True), ("16:00", True)])
def test_must_be_flat_regular_session(close, expected):
    day = "2026-10-07 "
    assert must_be_flat(pd.Timestamp(day + close), pd.Timestamp(day + "16:00"), 15) is expected


def test_must_be_flat_half_day():
    assert must_be_flat(pd.Timestamp("2026-11-27 12:45"), pd.Timestamp("2026-11-27 13:00"), 15)
