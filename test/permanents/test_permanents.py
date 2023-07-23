from infretis.classes.repex import REPEX_state
import numpy as np
import unittest

W_MATRIX1 = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 0., 0., 0.],
    [0., 1., 1., 1., 1., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1.],
    [0., 1., 1., 1., 1., 1., 1., 1.]])
P_MATRIX1 = np.array([
    [1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
    [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
    [0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
    [0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. ],
    [0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. ],
    [0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. ],
    [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5],
    [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5]])

W_MATRIX2 = np.array([
    [3.519e+03, 3.437e+03, 3.324e+03, 3.263e+03, 3.226e+03, 3.214e+03],
    [1.470e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
    [1.470e+02, 1.470e+02, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00],
    [1.540e+02, 8.500e+01, 3.400e+01, 1.800e+01, 4.000e+00, 1.000e+00],
    [1.090e+02, 9.200e+01, 7.000e+01, 4.500e+01, 2.600e+01, 1.100e+01],
    [1.390e+02, 1.120e+02, 6.900e+01, 2.900e+01, 9.000e+00, 1.000e+00]])
P_MATRIX2 = np.array([
    [0.        , 0.        , 0.03325386, 0.06703179, 0.21515415, 0.68456019],
    [1.        , 0.        , 0.        , 0.        , 0.        , 0.        ],
    [0.        , 1.        , 0.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.37105625, 0.40273855, 0.1679537 , 0.0582515 ],
    [0.        , 0.        , 0.15654483, 0.20783939, 0.40917538, 0.2264404 ],
    [0.        , 0.        , 0.43914505, 0.32239027, 0.20771676, 0.03074791]])

PERMANENT1 = 4.0
PERMANENT2 = 10508395762604.0

class PermanentTest(unittest.TestCase):
    def setUp(self):
        self.state = REPEX_state({'current':{'size':1},
            'dask':{'workers':1}})

    def test_matrix1(self):
       p_matrix=self.state.permanent_prob(W_MATRIX1)
       permanent=self.state.fast_glynn_perm(W_MATRIX1)
       self.assertTrue(np.allclose(p_matrix,P_MATRIX1))
       self.assertTrue(permanent == PERMANENT1)

    def test_matrix2(self):
       p_matrix=self.state.permanent_prob(W_MATRIX2)
       permanent=self.state.fast_glynn_perm(W_MATRIX2)
       self.assertTrue(np.allclose(p_matrix,P_MATRIX2))
       self.assertTrue(permanent == PERMANENT2)
