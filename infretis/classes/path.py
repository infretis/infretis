"""Define the path class."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from infretis.classes.formatter import (
    EnergyPathFile,
    OrderPathFile,
    PathExtFile,
)
from infretis.classes.system import System

if TYPE_CHECKING:  # pragma: no cover
    from numpy.random import Generator

    from infretis.classes.orderparameter import OrderParameter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


DEFAULT_MAXLEN: int = 100_000


class Path:
    """Define a Path class to store trajectories."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN, time_origin: int = 0):
        """Initiate a new path.

        Args:
            maxlen: The maximum length of the path.
            time_origin: The time origin for the path. Used to keep
                track of when the path was created.
        """
        self.maxlen = maxlen
        self.status: str = ""
        self.generated: Optional[Union[Tuple[str, float, int, int], str]] = (
            None
        )
        self.path_number = None
        self.weights: Optional[Tuple[float, ...]] = None
        self.phasepoints: List[System] = []
        self.time_origin = time_origin

    @property
    def length(self) -> int:
        """Compute the length of the path."""
        return len(self.phasepoints)

    @property
    def ordermin(self) -> Tuple[float, np.intp]:
        """Compute the minimum order parameter of the path."""
        idx = np.argmin([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def ordermax(self) -> Tuple[float, np.intp]:
        """Compute the maximum order parameter of the path."""
        idx = np.argmax([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def adress(self) -> set[Any]:
        """Get all configurations belonging to the trajectory."""
        adresses = set(i.config[0] for i in self.phasepoints)
        return adresses

    def check_interfaces(
        self, interfaces: List[float]
    ) -> Tuple[Optional[str], Optional[str], str, List[bool]]:
        """Check interfaces."""
        if self.length < 1:
            logger.warning("Path is empty!")
            return None, None, "*", [False] * len(interfaces)
        ordermax, ordermin = self.ordermax[0], self.ordermin[0]
        cross = [ordermin < interpos <= ordermax for interpos in interfaces]
        left, right = min(interfaces), max(interfaces)
        # Check end & start point:
        end = self.get_end_point(left, right)
        start = self.get_start_point(left, right)
        middle = "M" if cross[1] else "*"
        return start, end, middle, cross

    def get_end_point(
        self, left: float, right: Optional[float] = None
    ) -> Optional[str]:
        """Return the end point of the path as a string.

        The end point is either to the left of the `left` interface or
        to the right of the `right` interface, or somewhere in between.

        Args:
            left: The left interface.
            right: The right interface, equal to left if not specified.

        Returns:
            A string representing where the end point is ('L' - left,
                'R' - right or None).

        """
        if right is None:
            right = left
        assert left <= right

        if self.phasepoints[-1].order[0] <= left:
            end = "L"
        elif self.phasepoints[-1].order[0] >= right:
            end = "R"
        else:
            end = None
            logger.debug("Undefined end point.")
        return end

    def get_start_point(
        self, left: float, right: Optional[float] = None
    ) -> str:
        """Return the start point of the path as a string.

        The start point is either to the left of the `left` interface or
        to the right of the `right` interface.

        Args:
            left: The left interface.
            right: The right interface, equal to left if not specified.

        Returns:
            A string representing where the start point is ('L' - left,
                'R' - right or None).
        """
        if right is None:
            right = left
        assert left <= right
        if self.phasepoints[0].order[0] <= left:
            start = "L"
        elif self.phasepoints[0].order[0] >= right:
            start = "R"
        else:
            start = "?"
            logger.debug("Undefined starting point.")
        return start

    def get_shooting_point(self, rgen: Generator) -> Tuple[System, int]:
        """Pick a random shooting point from the path."""
        ### TODO: probably need an unittest for this to check if correct.
        ### idx = rgen.random_integers(1, self.length - 2)
        idx = int(rgen.integers(1, self.length - 1))
        order = self.phasepoints[idx].order[0]
        logger.debug(f"Selected point with orderp {order}")
        return self.phasepoints[idx], idx

    def append(self, phasepoint: System) -> bool:
        """Append a new phase point to the path."""
        if self.maxlen is None or self.length < self.maxlen:
            self.phasepoints.append(phasepoint)
            return True
        logger.debug("Max length exceeded. Could not append to path.")
        return False

    def get_move(self) -> Optional[str]:
        """Return the move used to generate the path."""
        if self.generated is None:
            return None
        return self.generated[0]

    def success(self, target_interface: float) -> bool:
        """Check if the path is successful.

        A path is successful if the maximum order parameter is greater
        than the given `target_interface`.

        Args:
            target_interface: The value for which the path is successful.

        Returns:
            True if the path is successful.
        """
        return self.ordermax[0] > target_interface

    def __iadd__(self, other: Path) -> Path:
        """Append phase points from another path, `self += other`.

        Args:
            other: The object to add path data from.

        Returns:
            The updated path object (self).
        """
        for phasepoint in other.phasepoints:
            app = self.append(phasepoint.copy())
            if not app:
                logger.warning(
                    "Truncated path at %d while adding paths", self.length
                )
                return self
        return self

    def copy(self) -> Path:
        """Return a copy of this path."""
        new_path = self.empty_path(maxlen=self.maxlen)
        for phasepoint in self.phasepoints:
            new_path.append(phasepoint.copy())
        new_path.status = self.status
        new_path.time_origin = self.time_origin
        new_path.generated = self.generated
        new_path.maxlen = self.maxlen
        new_path.path_number = self.path_number
        new_path.weights = self.weights
        return new_path

    def reverse_velocities(self, system: System) -> None:
        """Reverse the velocities in the system."""
        system.vel_rev = not system.vel_rev

    def reverse(
        self, order_function: Optional[OrderParameter], rev_v: bool = True
    ) -> Path:
        """Reverse a path and return the reverse path as a new path.

        Args:
            order_function : The order parameter function to use for
                recalculating the order parameter (in case it depends
                on the velocity).
            rev_v : If True, also the velocities are reversed.
                If False, the velocities for each frame are not altered.

        Returns:
            The time reversed path.
        """
        new_path = self.empty_path(maxlen=self.maxlen)
        new_path.weights = self.weights
        for phasepoint in reversed(self.phasepoints):
            new_point = phasepoint.copy()
            if rev_v:
                self.reverse_velocities(new_point)
            new_path.append(new_point)
        if order_function is None:
            return new_path
        if order_function.velocity_dependent and rev_v:
            for phasepoint in new_path.phasepoints:
                phasepoint.order = order_function.calculate(phasepoint)
        return new_path

    def empty_path(self, maxlen=DEFAULT_MAXLEN, **kwargs) -> Path:
        """Return an empty path of same class as the current one."""
        time_origin = kwargs.get("time_origin", 0)
        return self.__class__(maxlen=maxlen, time_origin=time_origin)

    def __eq__(self, other) -> bool:
        """Check if two paths are equal."""
        if self.__class__ != other.__class__:
            logger.debug("%s and %s.__class__ differ", self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug("%s and %s.__dict__ differ", self, other)
            return False

        # Compare phasepoints:
        if not len(self.phasepoints) == len(other.phasepoints):
            return False
        for i, j in zip(self.phasepoints, other.phasepoints):
            if not i == j:
                return False
        if self.phasepoints:
            # Compare other attributes:
            for key in (
                "maxlen",
                "time_origin",
                "status",
                "generated",
                "length",
                "ordermax",
                "ordermin",
                "path_number",
            ):
                attr_self = hasattr(self, key)
                attr_other = hasattr(other, key)
                if attr_self ^ attr_other:  # pragma: no cover
                    logger.warning(
                        'Failed comparing path due to missing "%s"', key
                    )
                    return False
                if not attr_self and not attr_other:
                    logger.warning(
                        'Skipping comparison of missing path attribute "%s"',
                        key,
                    )
                    continue
                if getattr(self, key) != getattr(other, key):
                    return False
        return True

    def __ne__(self, other) -> bool:
        """Check if two paths are not equal."""
        return not self == other

    def update_energies(
        self,
        ekin: Union[np.ndarray, List[float]],
        vpot: Union[np.ndarray, List[float]],
        etot: Union[np.ndarray, List[float]],
        temp: Union[np.ndarray, List[float]],
    ) -> None:
        """Update the energies for the phase points.

        This method is useful in cases where the energies are
        read from external engines and returned as a list of
        floats.

        Args:
            ekin : The kinetic energies to set.
            vpot : The potential energies to set.
            etot : The total energies to set.
            temp : The temperature to set.
        """
        energies = [ekin, vpot, etot, temp]
        names = ["ekin", "vpot", "etot", "temp"]

        len_p = len(self.phasepoints)
        for ene, name in zip(energies, names):
            len_e = len(ene)
            if len_e != len_p:
                logger.debug(
                    f"Length of {name} and phasepoints differ {len_e}!={len_p}"
                )
            for i, phasepoint in enumerate(self.phasepoints):
                try:
                    enei = ene[i]
                except IndexError:
                    logger.warning(f"Ran out of {name}, setting to None.")
                    enei = None
                setattr(phasepoint, name, enei)


def paste_paths(
    path_back: Path,
    path_forw: Path,
    overlap: bool = True,
    maxlen: Optional[int] = None,
) -> Path:
    """Merge a backward with a forward path into a new path.

    The resulting path is equal to the two paths stacked, in correct
    time. Note that the ordering is important here so that:
    ``paste_paths(path1, path2) != paste_paths(path2, path1)``.

    There are two things we need to take care of here:

    - `path_back` must be iterated in reverse (it is assumed to be a
      backward trajectory).
    - we may have to remove one point in `path_forw` (if the paths overlap).

    Args:
        path_back: The backward trajectory.
        path_forw: The forward trajectory.
        overlap: If True, `path_back` and `path_forw` have a common
            starting-point; the first point in `path_forw` is
            identical to the first point in `path_back`. In time-space, this
            means that the *first* point in `path_forw` is identical to the
            *last* point in `path_back` (the backward and forward path
            started at the same location in space).
        maxlen: This is the maximum length for the new path.
            If it's not given, it will just be set to the largest of
            the `maxlen` of the two given paths.

    Returns:
        The resulting path from the merge.

    Note:
        Some information about the path will not be set here. This must be
        set elsewhere. This includes how the path was generated
        (`path.generated`) and the status of the path (`path.status`).
    """
    if maxlen is None:
        if path_back.maxlen == path_forw.maxlen:
            maxlen = path_back.maxlen
        else:
            # They are unequal and both is not None, just pick the largest.
            # In case one is None, the other will be picked.
            # Note that now there is a chance of truncating the path while
            # pasting!
            maxlen = max(path_back.maxlen, path_forw.maxlen)
            msg = f"Unequal length: Using {maxlen} for the new path!"
            logger.warning(msg)
    time_origin = path_back.time_origin - path_back.length + 1
    new_path = path_back.empty_path(maxlen=maxlen, time_origin=time_origin)
    for phasepoint in reversed(path_back.phasepoints):
        app = new_path.append(phasepoint)
        if not app:
            msg = "Truncated while pasting backwards at: {}"
            msg = msg.format(new_path.length)
            logger.warning(msg)
            return new_path
    first = True
    for phasepoint in path_forw.phasepoints:
        if first and overlap:
            first = False
            continue
        app = new_path.append(phasepoint)
        if not app:
            msg = f"Truncated path at: {new_path.length}"
            logger.warning(msg)
            return new_path
    return new_path


def load_path(pdir: str) -> Path:
    """Load a path from the given directory."""
    trajtxt = os.path.join(pdir, "traj.txt")
    ordertxt = os.path.join(pdir, "order.txt")
    assert os.path.isfile(trajtxt)
    assert os.path.isfile(ordertxt)

    # load trajtxt
    with PathExtFile(trajtxt, "r") as trajfile:
        # Just get the first trajectory:
        traj = next(trajfile.load())

        # Update trajectory to use full path names:
        for i, snapshot in enumerate(traj["data"]):
            config = os.path.join(pdir, "accepted", snapshot[1])
            traj["data"][i][1] = config
            reverse = int(snapshot[3]) == -1
            idx = int(snapshot[2])
            traj["data"][i][2] = idx
            traj["data"][i][3] = reverse

        for config in set(frame[1] for frame in traj["data"]):
            assert os.path.isfile(config)

    # load ordertxt
    with OrderPathFile(ordertxt, "r") as orderfile:
        orderdata = next(orderfile.load())["data"][:, 1:]

    path = Path()
    for snapshot, order in zip(traj["data"], orderdata):
        frame = System()
        frame.order = order
        frame.config = (snapshot[1], snapshot[2])
        frame.vel_rev = snapshot[3]
        path.phasepoints.append(frame)
    _load_energies_for_path(path, pdir)
    # TODO: CHECK PATH SOMEWHERE .acc, sta = _check_path(path, path_ensemble)
    return path


def _load_energies_for_path(path: Path, dirname: str) -> None:
    """Load energy data for a path.

    Args:
        path: The path we are to set up/fill.
        dirname: The path to the directory with the input files.
    """
    energy_file_name = os.path.join(dirname, "energy.txt")
    try:
        with EnergyPathFile(energy_file_name, "r") as energyfile:
            energy = next(energyfile.load())
            path.update_energies(
                energy["data"]["ekin"],
                energy["data"]["vpot"],
                energy["data"]["etot"],
                energy["data"]["temp"],
            )
    except FileNotFoundError:
        pass


def load_paths_from_disk(config: Dict[str, Any]) -> List[Path]:
    """Load paths from disk."""
    load_dir = config["simulation"]["load_dir"]
    paths = []
    for pnumber in config["current"]["active"]:
        new_path = load_path(os.path.join(load_dir, str(pnumber)))
        status = "re" if "restarted_from" in config["current"] else "ld"
        ### TODO: important for shooting move if 'ld' is set. need a smart way
        ### to remember if status is 'sh' or 'wf' etc. maybe in the toml file.
        new_path.generated = (status, float("nan"), 0, 0)
        new_path.maxlen = config["simulation"]["tis_set"]["maxlength"]
        paths.append(new_path)
        # assign pnumber
        paths[-1].path_number = pnumber
    return paths
