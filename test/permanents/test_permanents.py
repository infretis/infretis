import numpy as np
import pytest

from infretis.classes.repex import REPEX_state

W_MATRIX1 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
)
P_MATRIX1 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    ]
)

W_MATRIX2 = np.array(
    [
        [3.519e03, 3.437e03, 3.324e03, 3.263e03, 3.226e03, 3.214e03],
        [1.470e02, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
        [1.470e02, 1.470e02, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
        [1.540e02, 8.500e01, 3.400e01, 1.800e01, 4.000e00, 1.000e00],
        [1.090e02, 9.200e01, 7.000e01, 4.500e01, 2.600e01, 1.100e01],
        [1.390e02, 1.120e02, 6.900e01, 2.900e01, 9.000e00, 1.000e00],
    ]
)
P_MATRIX2 = np.array(
    [
        [0.0, 0.0, 0.03325386, 0.06703179, 0.21515415, 0.68456019],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.37105625, 0.40273855, 0.1679537, 0.0582515],
        [0.0, 0.0, 0.15654483, 0.20783939, 0.40917538, 0.2264404],
        [0.0, 0.0, 0.43914505, 0.32239027, 0.20771676, 0.03074791],
    ]
)

W_MATRIX3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 14, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 450, 352, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1518, 854, 67, 57, 37, 3, 0, 0, 0, 0, 0, 0],
[0, 1, 1345, 821, 15, 8, 2, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 1128, 983, 23, 11, 3, 0, 0, 0, 0, 0, 0, 0],
[0, 1, 725, 324, 44, 42, 33, 21, 2, 0, 0, 0, 0, 0],
[0, 1, 1129, 970, 762, 717, 635, 368, 54, 1, 0, 0, 0, 0],
[0, 1, 58, 37, 24, 24, 19, 17, 12, 7, 1, 0, 0, 0],
[0, 1, 127, 113, 90, 89, 79, 49, 39, 31, 15, 3, 0, 0],
[0, 1, 2452, 1898, 704, 702, 700, 700, 23904, 21818, 18418, 6382, 2322, 988],
[0, 1, 2452, 1898, 704, 702, 700, 700, 23904, 21818, 18418, 6382, 2322, 988],
[0, 1, 444, 444, 372, 350, 322, 258, 192, 108, 90, 202, 190, 164]], dtype=np.float64)

PERMANENT1 = 4.0
PERMANENT2 = 10508395762604.0
# I guess this should really be positive but.. maybe in the future with better permanent algos
PERMANENT3 = -4.456605869717995727e+25


def test_matrix1():
    """Test ..."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0, "restarted_from": -1},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps": 10},
        }
    )
    p_matrix = state.permanent_prob(W_MATRIX1)
    permanent = state.fast_glynn_perm(W_MATRIX1)
    assert pytest.approx(p_matrix) == P_MATRIX1
    assert permanent == PERMANENT1


def test_matrix2():
    """Test ..."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0, "restarted_from": -1},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps": 10},
        }
    )
    p_matrix = state.permanent_prob(W_MATRIX2)
    permanent = state.fast_glynn_perm(W_MATRIX2)
    assert pytest.approx(p_matrix) == P_MATRIX2
    assert permanent == PERMANENT2


def test_matrix3(caplog):
    """This w matrix technically give negative number in the p matrix.
     But we check here that no zeros are present.

    """
    import logging
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps": 10},
        },
        minus=True
    )
    locks = np.zeros(W_MATRIX3.shape[0])
    with caplog.at_level(logging.INFO):
        # this versions makes negative numbers zero
        p_matrix1 = state.inf_retis(W_MATRIX3, locks)

    p_matrix2 = state.permanent_prob(W_MATRIX3)
    assert np.sum(np.abs(p_matrix1-p_matrix2)) < 10**(-5)
    assert np.sum(p_matrix1<0) == 0
    assert np.sum(p_matrix2<0) > 1
    assert np.abs(p_matrix2[p_matrix2<0])[0] < 10**(-10)
    assert "errors in the P-matrix," in caplog.text

    permanent = state.fast_glynn_perm(W_MATRIX3)
    assert pytest.approx(permanent) == PERMANENT3

