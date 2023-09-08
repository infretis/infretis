"""Define the OrderParameter class."""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from infretis.core.core import create_external, generic_factory

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.engines.enginebase import EngineBase
    from infretis.classes.system import System

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def pbc_dist_coordinate(
    distance: np.ndarray, box_lengths: np.ndarray
) -> np.ndarray:
    """Apply periodic boundaries to a distance.

    This will apply periodic boundaries to a distance. Note that the
    distance can be a vector, but not a matrix of distance vectors.

    Parameters
    ----------
    distance : numpy.array with shape `(self.dim,)`
        A distance vector.

    Returns
    -------
    out : numpy.array, same shape as the `distance` parameter
        The periodic-boundary wrapped distance vector.

    """
    box_ilengths = 1.0 / box_lengths
    pbcdist = np.zeros(distance.shape)
    for i, (length, ilength) in enumerate(zip(box_lengths, box_ilengths)):
        if np.abs(distance[i]) > 0.5 * length:
            pbcdist[i] = distance[i] - np.rint(distance[i] * ilength) * length
        else:
            pbcdist[i] = distance[i]
    return pbcdist


class OrderParameter:
    """Base class for order parameters.

    This class represents an order parameter and other collective
    variables. The order parameter is assumed to be a function
    that can uniquely be determined by the system object and its
    attributes.

    Attributes
    ----------
    description : string
        This is a short description of the order parameter.
    velocity_dependent : boolean
        This flag indicates whether or not the order parameter
        depends on the velocity direction. If so, we need to
        recalculate the order parameter when reversing trajectories.

    """

    def __init__(
        self,
        description: str = "Generic order parameter",
        velocity: bool = False,
    ):
        """Initialise the OrderParameter object.

        Parameters
        ----------
        description : string
            Short description of the order parameter.

        """
        self.description = description
        self.velocity_dependent = velocity
        if self.velocity_dependent:
            logger.debug(
                'Order parameter "%s" was marked as velocity dependent.',
                self.description,
            )

    @abstractmethod
    def calculate(self, system: System) -> list[float]:
        """Calculate the main order parameter and return it.

        All order parameters should implement this method as
        this ensures that the order parameter can be calculated.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This object contains the information needed to calculate
            the order parameter.

        Returns
        -------
        out : list of floats
        ￼
            The order parameter(s). The first order parameter returned
            is used as the progress coordinate in path sampling
            simulations!

        """

    def __str__(self) -> str:
        """Return a simple string representation of the order parameter."""
        msg = [
            f'Order parameter: "{self.__class__.__name__}"',
            f"{self.description}",
        ]
        if self.velocity_dependent:
            msg.append("This order parameter is velocity dependent.")
        return "\n".join(msg)

    @abstractmethod
    def load_restart_info(self, info: dict[str, str]):
        """Load the orderparameter restart info."""

    @abstractmethod
    def restart_info(self) -> dict[str, str]:
        """Save any mutatable parameters for the restart."""


class Distancevel(OrderParameter):
    """A rate of change of the distance order parameter.

    This class defines a very simple order parameter which is just
    the time derivative of the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index: tuple[int, int], periodic: bool = True):
        """Initialise the order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = (
            f"{pbc} rate-of-change-distance, particles {index[0]} and "
            f"{index[1]}"
        )
        super().__init__(description=txt, velocity=True)
        self.periodic = periodic
        self.index = index

    def calculate(self, system: System) -> list[float]:
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The rate-of-change of the distance order parameter.

        """
        delta = system.pos[self.index[1]] - system.pos[self.index[0]]
        if self.periodic:
            delta = pbc_dist_coordinate(delta, system.box)
        lamb = np.sqrt(np.dot(delta, delta))
        # Add the velocity as an additional collective variable:
        delta_v = system.vel[self.index[1]] - system.vel[self.index[0]]
        cv1 = np.dot(delta, delta_v) / lamb
        return [cv1]


class Position(OrderParameter):
    """Position order parameter.


    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index: tuple[int, int], periodic: bool = True):
        """Initialise order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = f"{pbc} distance, particles {index[0]} and {index[1]}"
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        if periodic:
            raise NotImplementedError("Can't use pbc for distance order yet")
        self.index = index

    def calculate(self, system: System) -> list[float]:
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The distance order parameter.

        """
        pos = system.pos[self.index[0], self.index[1]]
        return [pos]


class Distance(OrderParameter):
    """A distance order parameter.

    This class defines a very simple order parameter which is just
    the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index: tuple[int, int], periodic: bool = True):
        """Initialise order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = f"{pbc} distance, particles {index[0]} and {index[1]}"
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        self.index = index

    def calculate(self, system: System) -> list[float]:
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The distance order parameter.

        """
        box = np.array(system.box[:3])
        delta = system.pos[self.index[1]] - system.pos[self.index[0]]
        if self.periodic:
            delta = pbc_dist_coordinate(delta, box)
        lamb = np.sqrt(np.dot(delta, delta))
        return [lamb]


class Velocity(OrderParameter):
    """Initialise the order parameter.

    This class defines a very simple order parameter which is just
    the velocity of a given particle.

    Attributes
    ----------
    index : integer
        This is the index of the atom which will be used, i.e.
        ``system.particles.vel[index]`` will be used.
    dim : integer
        This is the dimension of the coordinate to use.
        0, 1 or 2 for 'x', 'y' or 'z'.

    """

    def __init__(self, index: int, dim: str = "x"):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the velocity of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.

        """
        txt = f"Velocity of particle {index} (dim: {dim})"
        super().__init__(description=txt, velocity=True)
        self.index = index
        self.dim = {"x": 0, "y": 1, "z": 2}.get(dim, None)
        if self.dim is None:
            logger.critical("Unknown dimension %s requested", dim)
            raise ValueError

    def calculate(self, system: System) -> list[float]:
        """Calculate the velocity order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the velocities.

        Returns
        -------
        out : list of floats
            The velocity order parameter.

        """
        return [system.vel[self.index][self.dim]]


def create_orderparameters(
    engines: dict[str, EngineBase], settings: dict[str, Any]
):
    """Create orderparameters."""
    for engine in engines.keys():
        engines[engine].order_function = create_orderparameter(settings)


def create_orderparameter(settings: dict[str, Any]) -> OrderParameter | None:
    """Create order parameters from settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    out : object like :py:class:`.OrderParameter`
        This object represents the order parameter.

    """
    order_map = {
        "orderparameter": {"cls": OrderParameter},
        "position": {"cls": Position},
        "velocity": {"cls": Velocity},
        "distance": {"cls": Distance},
        "dihedral": {"cls": Dihedral},
        "distancevel": {"cls": Distancevel},
    }

    if settings["orderparameter"]["class"].lower() not in order_map:
        return create_external(
            settings["orderparameter"], "orderparameter", ["calculate"]
        )

    main_order = generic_factory(
        settings["orderparameter"], order_map, name="engine"
    )

    if main_order is None:
        logger.info("No order parameter created")
        print("omg..")
        return None
    logger.info("Created main order parameter:\n%s", main_order)
    return main_order


def _verify_pair(index: tuple[int, int]):
    """Check that the given index contains a pair."""
    try:
        if len(index) != 2:
            msg = (
                "Wrong number of atoms for pair definition. "
                f"Expected 2 got {len(index)}"
            )
            logger.error(msg)
            raise ValueError(msg)
    except TypeError as err:
        msg = "Atom pair should be defined as a tuple/list of integers."
        logger.error(msg)
        raise TypeError(msg) from err


class Dihedral(OrderParameter):
    """Calculates the dihedral angle defined by 4 atoms.

    The angle definition is given by Blondel and Karplus,
    J. Comput. Chem., vol. 17, 1996, pp. 1132--1141. If we
    label the 4 atoms A, B, C and D, then the angle is given by
    the vectors u = A - B, v = B - C, w = D - C

    Attributes
    ----------
    index : list/tuple of integers
        These are the indices for the atoms to use in the
        definition of the dihedral angle.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the position or not.

    """

    def __init__(
        self, index: tuple[int, int, int, int], periodic: bool = False
    ):
        """Initialise the order parameter.

        Parameters
        ----------
        index : list/tuple of integers
            This list gives the indices for the atoms to use in the
            definition of the dihedral angle.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the distance vectors.

        """
        try:
            if len(index) != 4:
                msg = (
                    "Wrong number of atoms for dihedral definition. "
                    f"Expected 4 got {len(index)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        except TypeError as err:
            msg = "Dihedral should be defined as a tuple/list of integers!"
            logger.error(msg)
            raise TypeError(msg) from err
        self.index = [int(i) for i in index]
        txt = (
            "Dihedral angle between particles "
            f"{index[0]}, {index[1]}, {index[2]} and {index[3]}"
        )
        super().__init__(description=txt)
        self.periodic = periodic

    def calculate(self, system: System) -> list[float]:
        """Calculate the dihedral angle.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the information we need to calculate
            the order parameter.

        Returns
        -------
        out : list of float
            The order parameter.

        """
        pos = system.pos
        box = np.array(system.box[:3])
        vector1 = pos[self.index[0]] - pos[self.index[1]]
        vector2 = pos[self.index[1]] - pos[self.index[2]]
        vector3 = pos[self.index[3]] - pos[self.index[2]]

        if self.periodic:
            vector1 = pbc_dist_coordinate(vector1, box)
            vector2 = pbc_dist_coordinate(vector2, box)
            vector3 = pbc_dist_coordinate(vector3, box)
        # Norm to simplify formulas:
        vector2 /= np.linalg.norm(vector2)
        denom = np.dot(vector1, vector3) - np.dot(vector1, vector2) * np.dot(
            vector2, vector3
        )
        numer = np.dot(np.cross(vector1, vector2), vector3)
        angle = np.arctan2(numer, denom)
        return [angle]
