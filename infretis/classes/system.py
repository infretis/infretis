"""Defines the snapshot system class."""
import logging
from copy import copy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class System:
    """System class."""

    config = None
    order = None
    pos = None

    def __init__(self):
        """Initiate class."""
        self.vel = None
        self.vel_rev = None
        self.ekin = None
        self.vpot = None
        self.box = None

    def copy(self):
        """Return a copy of the system.

        This copy is useful for storing snapshots obtained during
        a simulation.

        Returns
        -------
        out : object like :py:class:`.System`
            A copy of the system.

        """
        system_copy = copy(self)
        return system_copy

    def set_pos(self, pos):
        """Set positions for the particles."""
        self.config = (pos[0], pos[1])

    def set_vel(self, rev_vel):
        """Set velocities for the particles.

        Here we store information which tells if the
        velocities should be reversed or not.

        Parameters
        ----------
        rev_vel : boolean
            The velocities to set. If True, the velocities should
            be reversed before used.

        """
        self.vel_rev = rev_vel
