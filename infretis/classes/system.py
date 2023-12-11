"""Defines the snapshot system class."""
from __future__ import annotations

import logging
from copy import copy

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
print('System')


class System:
    """System class."""

    config: tuple[None, None] = (None, None)
    order: list[float] | None = None
    pos: np.ndarray = np.zeros(0)

    def __init__(self) -> None:
        """Initiate class."""
        self.vel: np.ndarray = np.zeros(0)
        self.vel_rev: bool = False
        self.ekin: float | None = None
        self.vpot: float | None = None
        self.box: np.ndarray | None = np.zeros((3, 3))
        self.temperature: dict[str, float] = {}

    def copy(self) -> System:
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

    def set_pos(self, pos: tuple[str, int]) -> None:
        """Set positions for the particles."""
        self.config = (pos[0], pos[1])

    def set_vel(self, rev_vel: bool) -> None:
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
