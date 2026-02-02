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

PERMANENT1 = 4.0
PERMANENT2 = 10508395762604.0


def test_matrix1():
    """Test ..."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0},
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
            "current": {"size": 1, "cstep": 0},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps": 10},
        }
    )
    p_matrix = state.permanent_prob(W_MATRIX2)
    permanent = state.fast_glynn_perm(W_MATRIX2)
    assert pytest.approx(p_matrix) == P_MATRIX2
    assert permanent == PERMANENT2


def test_matrix3(caplog):
    """This matrix technically give negative number. but we check here that no zeros are present.

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
    locks = np.zeros(W_MATRIX2.shape[0])
    with caplog.at_level(logging.INFO):
        # this versions makes negative numbers zero
        p_matrix1 = state.inf_retis(W_MATRIX2, locks)

    # the underlying permanent calculation:
    p_matrix2 = state.permanent_prob(W_MATRIX2)

    assert np.sum(p_matrix1<0) == 0
    assert np.sum(p_matrix2<0) == 1
    assert "Numerical instability detected in permanent calculation!" in caplog.text
