import numpy as np

from experiments.loss_audit import swap_witness
from ms.engine import Config, Minefield


def _field(mines):
    cfg = Config(4, 5, len(mines))
    rng = np.random.default_rng(0)
    mf = Minefield(cfg, rng)
    mask = np.zeros(mf.n, dtype=bool)
    mask[mines] = True
    mf.load(mask)
    mf.click(0)
    return mf


def test_swap_certificate_has_same_observation_and_all_common_safe_digits():
    mf = _field([12, 17, 18])
    assert mf.click(18) == 'boom'
    witness = swap_witness(mf, dead=18, opening=0)
    assert witness is not None
    mask = mf.mines.copy()
    mask[[18, witness]] ^= True
    other = Minefield(mf.cfg, np.random.default_rng(1))
    other.load(mask)
    other.click(0)
    np.testing.assert_array_equal(mf.visible, other.visible)
    commons = ~(mask | mf.mines)
    np.testing.assert_array_equal(mf.adjacent[commons], other.adjacent[commons])


def test_exposed_distinguishing_neighbor_removes_certificate():
    mf = _field([12, 18])
    assert swap_witness(mf, dead=18, opening=0) is None
