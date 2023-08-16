# -*- coding: utf-8 -*-
# Copyright (c) 2019, PyRETIS Development Team.
"""This is a 2D example potential."""
import logging
import numpy as np
from pyretis.forcefield.potential import PotentialFunction

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class PotentialTwoChannels(PotentialFunction):
    r"""Hyst2D(PotentialFunction).

    This class defines a 2D dimensional potential with two
    reactions channels.
    The potential energy (:math:`V_\text{pot}`) is given by

    TODO -- adapt.... TODO

    .. math::

       V_\text{pot}(x, y) = \exp(-cx^2)
         ( Vmin/4 + Vmin2/4 + Vmax/2

         + (Vmin2-Vmin1)/2 \sin(2\pi y/L)
         + (Vmax/2-Vmin1/2-Vmin2/2) \cos(4\pi y/L) )

    where :math:`x` and :math:`y` gives the positions and :math:`\gamma_1`,
    :math:`\gamma_2`, :math:`\gamma_3`, :math:`\alpha_1`, :math:`\alpha_2`,
    :math:`\beta_1`, :math:`\beta_2`, :math:`x_0` and :math:`y_0` are
    potential parameters.

    Attributes
    ----------
    params : dict
        Contains the parameters. The keys are:

        * `gamma1`: The :math:`\gamma_1` parameter for the potential.
        * `gamma2`: The :math:`\gamma_2` parameter for the potential.
        * `gamma3`: The :math:`\gamma_3` parameter for the potential.
        * `alpha1`: The :math:`\alpha_1` parameter for the potential.
        * `alpha2`: The :math:`\alpha_2` parameter for the potential.
        * `beta1`: The :math:`\beta_1` parameter for the potential.
        * `beta2`: The :math:`\beta_2` parameter for the potential.
        * `x0`: The :math:`x_0` parameter for the potential.
        * `y0`: The :math:`y_0` parameter for the potential.

    """

    def __init__(self, desc="2D surface with 2 reaction channels"):
        """Set up the potential.

        Parameters
        ----------
        a : float, optional
            Parameter for the potential.
        b : float, optional
            Parameter for the potential.
        c : float, optional
            Parameter for the potential.
        desc : string, optional
            Description of the force field.

        """
        super().__init__(dim=2, desc=desc)
        self.params = {
            "Vmin1": 0.0,
            "Vmin2": 0.0,
            "Vmax": 0.0,
            "L": 0.0,
            "c": 0.0,
        }

    def potential(self, system):
        """Evaluate the potential.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out : float
            The potential energy.

        """
        # x = system.particles.pos[:, 0]  # pylint: disable=invalid-name
        # y = system.particles.pos[:, 1]  # pylint: disable=invalid-name
        # I need to do WRAPPING!
        pos = system.box.pbc_wrap(system.particles.pos)
        x = pos[:, 0]  # pylint: disable=invalid-name
        y = pos[:, 1]  # pylint: disable=invalid-name
        Vmin1 = self.params["Vmin1"]
        Vmin2 = self.params["Vmin2"]
        Vmax = self.params["Vmax"]
        c = self.params["c"]
        L = self.params["L"]

        A = (Vmin2 - Vmin1) / 2
        B = Vmax / 2 - Vmin1 / 4 - Vmin2 / 4

        f1 = A * np.sin(2 * np.pi * y / L)
        f2 = B * np.cos(4 * np.pi * y / L)  # different period
        f3 = np.exp(-c * x**2)

        # V_\text{pot}(x, y) = \exp(-cx^2)
        # ( Vmin/4 + Vmin2/4 + Vmax/2
        # + (Vmin2-Vmin1)/2 \sin(2\pi y/L)
        # + (Vmax/2-Vmin1/2-Vmin2/2) \cos(4\pi y/L) )
        v_pot = (Vmin1 + A + B + f1 + f2) * f3

        return v_pot.sum()

    def force(self, system):
        """Evaluate forces.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : numpy.array
            The calculated force.
        out[1] : numpy.array
            The virial, currently not implemented for this potential.

        """
        pos = system.box.pbc_wrap(system.particles.pos)
        x = pos[:, 0]  # pylint: disable=invalid-name
        y = pos[:, 1]  # pylint: disable=invalid-name
        Vmin1 = self.params["Vmin1"]
        Vmin2 = self.params["Vmin2"]
        Vmax = self.params["Vmax"]
        c = self.params["c"]
        L = self.params["L"]

        A = (Vmin2 - Vmin1) / 2
        B = Vmax / 2 - Vmin1 / 4 - Vmin2 / 4

        f1 = A * np.sin(2 * np.pi * y / L)
        f2 = B * np.cos(4 * np.pi * y / L)  # different period
        f3 = np.exp(-c * x**2)

        # V_\text{pot}(x, y) = \exp(-cx^2)
        # ( Vmin/4 + Vmin2/4 + Vmax/2
        # + (Vmin2-Vmin1)/2 \sin(2\pi y/L)
        # + (Vmax/2-Vmin1/2-Vmin2/2) \cos(4\pi y/L) )
        # v_pot = (Vmin1 + A + B + f1 + f2)*f3
        forces = np.zeros_like(system.particles.pos)
        forces[:, 0] = (
            2.0 * c * x * (Vmin1 + A + B + f1 + f2) * f3
        )  # -( -2*c*x) = +2*c*x
        forces[:, 1] = -f3 * (
            A * 2 * np.pi / L * np.cos(2 * np.pi * y / L)
            - B * 4 * np.pi / L * np.sin(4 * np.pi * y / L)
        )

        virial = np.zeros((self.dim, self.dim))  # just return zeros here
        return forces, virial

    def potential_and_force(self, system):
        """Evaluate the potential and the force.

        Parameters
        ----------
        system : object like `System`
            The system we evaluate the potential for. Here, we
            make use of the positions only.

        Returns
        -------
        out[0] : float
            The potential energy as a float.
        out[1] : numpy.array
            The force as a numpy.array of the same shape as the
            positions in `particles.pos`.
        out[2] : numpy.array
            The virial, currently not implemented for this potential.

        """
        virial = np.zeros((self.dim, self.dim))  # just return zeros here
        vpot = self.potential(system)
        forces, virial = self.force(system)

        return vpot, forces, virial
