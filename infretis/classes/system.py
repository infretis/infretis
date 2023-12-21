"""Defines the snapshot system class."""
from __future__ import annotations

import logging
from copy import copy

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class System:
    """Define the representation of System snapshots.

    The system is intended for sharing the current configuration (of a
    frame in a path) between different methods.
    """

    def __init__(self) -> None:
        """Initiate class."""
        self.config: tuple[str, int] = ("", -1)
        self.order: list[float] = [-float("nan")]
        self.pos: np.ndarray = np.zeros(0)
        self.vel: np.ndarray = np.zeros(0)
        self.vel_rev: bool = False
        self.ekin: float | None = None
        self.vpot: float | None = None
        self.box: np.ndarray | None = np.zeros((3, 3))
        self.temperature: dict[str, float] = {}

    def copy(self) -> System:
        """Return a copy of this system."""
        system_copy = copy(self)
        return system_copy

    def set_pos(self, pos: tuple[str, int]) -> None:
        """Set positions for the particles."""
        self.config = (pos[0], pos[1])
