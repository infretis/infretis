# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""This file defines the order parameter used for the GROMACS example."""
import logging
import numpy as np
from numpy import average, rint, dot, sqrt
from infretis.classes.orderparameter import OrderParameter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class RingDiffusion(OrderParameter):
    """RingDiffusion(OrderParameter).

    This class defines the ring diffusion order parameter for
    the hydrate example.

    Attributes
    ----------
    name : string
        A human-readable name for the order parameter.
    index : integer
        This selects the particle to use for the order parameter.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the position or not.

    """

    def __init__(self):
        """Initialise the order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        super().__init__(description='Ring diffusion for hydrate')
        self.idx1 = np.array([56, 64, 104, 112, 200, 208], dtype=np.int16)
        # convert to atom index:
        self.idx1 *= 4
        self.idx1 -= 3
        # convert to 0 index:
        self.idx1 -= 1
        self.idx2 = np.array([56, 64, 72, 80, 104, 112, 136, 152, 168, 176,
                              200, 208, 232, 248, 264, 272, 296, 304, 328,
                              336, 344, 352, 360, 368], dtype=np.int16)
        # convert to atom index:
        self.idx2 *= 4
        self.idx2 -= 3
        # convert to 0 index:
        self.idx2 -= 1
        self.idxd = 1472  # index for diffusing atom

    def calculate(self, system):
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like `System` from `pyretis.core.system`
            This object is used for the actual calculation, typically
            only `system.particles.pos` and/or `system.particles.vel`
            will be used. In some cases `system.forcefield` can also be
            used to include specific energies for the order parameter.

        Returns
        -------
        out : float
            The order parameter.

        """
        pos = system.particles.pos
        resl = 1.0e3
        cm1 = average(rint(pos[self.idx1] * resl) / resl, axis=0)
        cm2 = average(rint(pos[self.idx2] * resl) / resl, axis=0)
        cmvec = cm2 - cm1
        molvec = rint(pos[self.idxd] * resl) / resl
        molvec -= cm1
        orderp = dot(cmvec, molvec) / sqrt(dot(cmvec, cmvec))
        return [-1.0 * orderp]
