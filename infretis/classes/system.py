"""Defines the snapshot system class."""
from copy import copy
import logging
from infretis.core.core import compare_objects
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

    def __eq__(self, other):
        """Test whether two systems are equal."""
        attrs = self.__dict__.keys()
        return compare_objects(self, other, attrs)

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


def system_from_snapshot(system, snapshot):
    """Create a system from a given snapshot."""
    system_copy = system.copy()
    system_copy.particles.ekin = snapshot.get('ekin', None)
    system_copy.particles.vpot = snapshot.get('vpot', None)
    system_copy.order = snapshot.get('order', None)
    system_copy.particles.set_pos(snapshot.get('pos', None))
    system_copy.particles.set_vel(snapshot.get('vel', None))
    return system_copy
