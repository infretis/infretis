"""Define the OrderParameter class."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple, Dict, List

import numpy as np

from infretis.core.core import create_external, generic_factory

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.system import System

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def pbc_dist_coordinate(
    distance: np.ndarray, box_lengths: np.ndarray
) -> np.ndarray:
    """Apply periodic boundaries to a distance.

    Apply periodic boundaries to a distance vector.

    Args:
        distance: A distance vector.
        box_lengths: The box lengths (one for each dimension.)

    Returns:
        The periodic-boundary wrapped distance vector (a numpy.array
            with the same shape as the `distance` parameter).

    Note:
        This method assumes an orthogonal box.
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
    uniquely determined by the system and its attributes.

    Attributes
        description: A short (textual) description of the order parameter.
        velocity_dependent: This flag indicates whether or not the
            order parameter depends on the velocity direction. If so,
            the order parameter *must* be recalculated when reversing
            trajectories.
    """

    def __init__(
        self,
        description: str = "Generic order parameter",
        velocity: bool = False,
    ):
        """Initialize the OrderParameter object.

        Args:
            description: Short description of the order parameter.
            velocity: If True, the order parameter is flagged as
                velocity dependent.
        """
        self.description = description
        self.velocity_dependent = velocity
        if self.velocity_dependent:
            logger.debug(
                'Order parameter "%s" was marked as velocity dependent.',
                self.description,
            )

    @abstractmethod
    def calculate(self, system: System) -> List[float]:
        """Calculate the order parameter.

        All order parameters **must** implement this method.

        Args:
            system: This object contains the information
                needed to calculate the order parameter.

        Returns:
            A list of floats containing the order parameter(s). The
                first item in this list is used as the progress
                coordinate in path sampling simulations.
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


class Distancevel(OrderParameter):
    """A rate of change of the distance order parameter.

    This order parameter is the time derivative of the scalar
    distance between two particles.

    Attributes:
        index: These are the indices used for the two particles. The
            distance is calculated using `system.pos[index[0]]`
            and `system.pos[index[1]]`.
        periodic: If True, apply periodic boundaries to the distance.
    """

    def __init__(self, index: Tuple[int, int], periodic: bool = True):
        """Initialize the order parameter.

        Args:
            index: The indices of the particles to use for the distance.
            periodic: If True, apply periodic boundary conditions.

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

    def calculate(self, system: System) -> List[float]:
        """Calculate the order parameter.

        Args:
            system: The object containing the positions
                and box used for the calculation.

        Returns:
            A list where the first item is the rate of change of the
            distance.
        """
        delta = system.pos[self.index[1]] - system.pos[self.index[0]]
        if self.periodic and system.box is not None:
            delta = pbc_dist_coordinate(delta, system.box)
        lamb = np.sqrt(np.dot(delta, delta))
        # Add the velocity as an additional collective variable:
        delta_v = system.vel[self.index[1]] - system.vel[self.index[0]]
        cv1 = np.dot(delta, delta_v) / lamb
        return [cv1]


class Position(OrderParameter):
    """The position of a particle in a dimension.

    Attributes:
        index: Tuple of integers where the first integer
            selects the particle and the second integer
            selects the dimension.
        periodic: True, if we should apply periodic boundaries
            to the position.
    """

    def __init__(self, index: Tuple[int, int], periodic: bool = True):
        """Initialize the position order parameter.

        Args:
            index: This tuple is used to select the particle (`index[0]`)
                and the dimension (`index[1]`) to use.
            periodic: If True, periodic boundary conditions should be
                applied to the position.

        Note:
            The periodic setting is currently not supported.
        """
        _verify_pair(index)
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = f"{pbc} distance, particles {index[0]} and {index[1]}"
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        if self.periodic:
            raise NotImplementedError("Can't use pbc for position order yet")
        self.index = index

    def calculate(self, system: System) -> List[float]:
        """Calculate the order parameter."""
        return [system.pos[self.index[0], self.index[1]]]


class Distance(OrderParameter):
    """The scalar distance between two particles.

    Attributes:
        index: A tuple of integers that selects the two
            particles.
        periodic: This determines if periodic boundaries
            should be applied to the distance or not.
    """

    def __init__(self, index: Tuple[int, int], periodic: bool = True):
        """Initialize the order parameter.

        Args:
            index: A tuple of ints that selects the two particles.
            periodic: This determines if periodic boundary conditions
                should be applied to the position.
        """
        _verify_pair(index)
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = f"{pbc} distance, particles {index[0]} and {index[1]}"
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        self.index = index

    def calculate(self, system: System) -> List[float]:
        """Calculate the order parameter."""
        delta = system.pos[self.index[1]] - system.pos[self.index[0]]
        if self.periodic and system.box is not None:
            box = np.array(system.box[:3])
            delta = pbc_dist_coordinate(delta, box)
        lamb = np.sqrt(np.dot(delta, delta))
        return [lamb]


class Velocity(OrderParameter):
    """The velocity of a given particle in a given dimension.

    Attributes:
        index: This is the index of the particle which will be
            used, i.e., `system.particles.vel[index]` will be used.
        dim: A integer representing the dimension to use:
            0, 1 or 2 for 'x', 'y' or 'z'.
    """

    def __init__(self, index: int, dim: str = "x"):
        """Initialize the order parameter.

        Args:
            index: This is the index of the particle we will use
                the velocity of.
        dim: A string ("x", "y", "z") that selects the
            dimension of the velocity to use.
        """
        dim = dim.lower()
        txt = f"Velocity of particle {index} (dim: {dim})"
        super().__init__(description=txt, velocity=True)
        self.index = index
        self.dim = {"x": 0, "y": 1, "z": 2}.get(dim, None)
        if self.dim is None:
            logger.critical("Unknown dimension %s requested", dim)
            raise ValueError

    def calculate(self, system: System) -> List[float]:
        """Calculate the velocity order parameter."""
        return [system.vel[self.index][self.dim]]


def create_orderparameters(
    engines: Dict[str, List],
    settings: Dict[str, Any],
):
    """Create orderparameters."""
    for engine_key in engines.keys():
        for engine in engines[engine_key]:
            engine.order_function = create_orderparameter(settings)



def create_orderparameter(settings: Dict[str, Any]) -> Optional[OrderParameter]:
    """Create order parameters from settings.

    Args:
        settings: A dictionary with simulation settings. This
        method will use the `"orderparameter"` section of the settings.

    Returns:
        The created order parameter.
    """
    order_map = {
        "orderparameter": {"class": OrderParameter},
        "position": {"class": Position},
        "velocity": {"class": Velocity},
        "distance": {"class": Distance},
        "dihedral": {"class": Dihedral},
        "distancevel": {"class": Distancevel},
        "puckering": {"class": Puckering},
    }

    if settings["orderparameter"]["class"].lower() not in order_map:
        return create_external(
            settings["orderparameter"], "orderparameter", ["calculate"]
        )

    main_order = generic_factory(
        settings["orderparameter"], order_map, name="engine"
    )
    logger.info("Created main order parameter:\n%s", main_order)
    return main_order


def _verify_pair(index: Tuple[int, int]):
    """Check that the given index contains a pair."""
    try:
        if len(index) != 2:
            msg = (
                "Wrong number of particles for pair definition. "
                f"Expected 2 got {len(index)}"
            )
            logger.error(msg)
            raise ValueError(msg)
    except TypeError as err:
        msg = "Particle pair should be defined as a tuple/list of integers."
        logger.error(msg)
        raise TypeError(msg) from err


class Dihedral(OrderParameter):
    """Calculate the dihedral angle defined by 4 particles.

    The angle definition is given by Blondel and Karplus,
    J. Comput. Chem., vol. 17, 1996, pp. 1132--1141. If we
    label the 4 particles A, B, C and D, then the angle is given by
    the vectors u = A - B, v = B - C, w = D - C

    Attributes:
        index: A list/tuple of integers that defines the 4
            particles in the dihedral angle.
        periodic: If True, we apply periodic boundaries.
    """

    def __init__(
        self, index: Tuple[int, int, int, int], periodic: bool = False
    ):
        """Initialize the order parameter.

        Args:
            index: The indices of the 4 particles in the dihedral.
            periodic: This determines if periodic boundary conditions
                should be applied to distance vectors.
        """
        try:
            if len(index) != 4:
                msg = (
                    "Wrong number of particles for dihedral definition. "
                    f"Expected 4 got {len(index)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        except TypeError as err:
            msg = "Dihedral should be defined as a tuple/list of integers!"
            logger.error(msg)
            raise TypeError(msg) from err
        self.index = tuple(int(i) for i in index)
        txt = (
            "Dihedral angle between particles "
            f"{index[0]}, {index[1]}, {index[2]} and {index[3]}"
        )
        super().__init__(description=txt)
        self.periodic = periodic

    def calculate(self, system: System) -> List[float]:
        """Calculate the dihedral angle."""
        pos = system.pos
        vector1 = pos[self.index[0]] - pos[self.index[1]]
        vector2 = pos[self.index[1]] - pos[self.index[2]]
        vector3 = pos[self.index[3]] - pos[self.index[2]]

        if self.periodic and system.box is not None:
            box = np.array(system.box[:3])
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


class Puckering(OrderParameter):
    """Calculate puckering coordinates for a 6-ring.

    The puckering coordinates are described by Cremer and Pople in
    J. Am. Chem. Soc. 1975, 97, 6, 1354–1358.

    For six-membered rings, there are 3 puckering
    degrees of freedom. They can be described as a spherical polar
    set given by (theta, phi, Qampl). The poles of this sphere
    (given by theta = 0 and theta = 180) are the well-known 1C4 or 4C1
    chair conformations.

    For carbohydrates, the indices convention is
        O5:0, C1:1, C2:2,..., C5:5

    Attributes:
        index: A list/tuple of integers used to select the
            particles to use for the puckering coordinates.
        periodic: This determines if periodic boundaries should
            be applied to the position or not.
    """

    def __init__(
        self,
        index: Tuple[int, int, int, int, int, int],
        periodic: bool = False,
    ):
        """Initialize the order parameter.

        Args:
            index: This list gives the indices for the particles
                to use in the definition of the puckering coordinates.
            periodic: If True, apply periodic boundary conditions to
                distance vectors.
        """
        try:
            if len(index) != 6:
                msg = (
                    "Wrong number of particles for 6-ring puckering "
                    f"definition. Expected 6 got {len(index)}"
                )
                logger.error(msg)
                raise ValueError(msg)
        except TypeError as err:
            msg = "Puckering particles should be defined as a \
                    tuple/list of integers!"
            logger.error(msg)
            raise TypeError(msg) from err
        self.index = tuple(int(i) for i in index)
        self.periodic = periodic
        txt = (
            "Puckering coordinates between particles "
            f"{index[0]}, {index[1]}, {index[2]}, \
                    {index[3]}, {index[4]} and {index[5]}."
        )
        super().__init__(description=txt)

    def calculate(self, system: System) -> List[float]:
        """Calculate the puckering angle."""
        pos = system.pos[list(self.index)]
        if self.periodic and system.box is not None:
            box = np.array(system.box[:3])
            # make 6-ring whole around atom 0
            for i in range(1, 6):
                pos[i, :] = pbc_dist_coordinate(pos[i, :] - pos[0, :], box)
            # set atom 0 position to the origin
            pos[0, :] *= 0
        # geometric center of the molecule
        center = np.mean(pos, axis=0)
        # translate origin of molecule to geometric center
        for i in range(6):
            pos[i, :] -= center
        # get the R1 = R' and R2 = R'' vectors
        R1 = np.zeros(3)
        R2 = np.zeros(3)
        for i in range(6):
            R1 += pos[i, :] * np.sin(2 * np.pi * (i - 6) / 6)
            R2 += pos[i, :] * np.cos(2 * np.pi * (i - 6) / 6)
        # get the molecular z-axis defined by the vector n perpendicular
        # to the mean plane of the ring
        n = np.cross(R1, R2)
        n = n / np.linalg.norm(n)
        # displacements from the mean plane of the ring
        z = np.zeros(6)
        for i in range(6):
            z[i] = np.dot(pos[i, :], n)
        # get the generalized ring puckering coordinates (q2, phi2, q3)
        # (from eq. 12, 13 and 14 in the article) which will be used to
        # get the sphetical (theta, phi, Q) ring puckering coordinates
        h1 = 0.0
        h2 = 0.0
        q3 = 0.0
        for i in range(6):
            h1 += np.sqrt(2 / 6) * z[i] * np.cos(2 * np.pi * 2 * i / 6)
            h2 += -np.sqrt(2 / 6) * z[i] * np.sin(2 * np.pi * 2 * i / 6)
            q3 += np.sqrt(1 / 6) * (-1) ** (i) * z[i]
        q2 = np.sqrt(h1**2 + h2**2)
        theta = np.arctan2(q2, q3)
        phi = np.arctan2(h2, h1)
        # map to -180,+180 to 0,360 deg
        if phi < 0:
            phi += np.pi * 2
        Qampl = np.sqrt(np.sum(z**2))
        return [np.rad2deg(theta), np.rad2deg(phi), Qampl]
