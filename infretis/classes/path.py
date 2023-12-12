"""Define the path class."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

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
    """Define Path class."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN, time_origin: int = 0):
        """Initiate Path class."""
        self.maxlen = maxlen
        self.status: str = ""
        self.generated: tuple[str, float, int, int] | str | None = None
        self.path_number = None
        self.weights: tuple[float, ...] | None = None
        self.weight: float = 0.0
        self.phasepoints: list[System] = []
        self.min_valid = None
        self.time_origin = time_origin

    @property
    def length(self) -> int:
        """Compute the length of the path."""
        return len(self.phasepoints)

    @property
    def ordermin(self) -> tuple[float, np.intp]:
        """Compute the minimum order parameter of the path."""
        idx = np.argmin([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def ordermax(self) -> tuple[float, np.intp]:
        """Compute the maximum order parameter of the path."""
        idx = np.argmax([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def adress(self) -> set[Any]:
        """Compute the maximum order parameter of the path."""
        adresses = set(i.config[0] for i in self.phasepoints)
        return adresses

    def check_interfaces(
        self, interfaces: list[float]
    ) -> tuple[str | None, str | None, str, list[bool]]:
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
        self, left: float, right: float | None = None
    ) -> str | None:
        """Return the end point of the path as a string.

        The end point is either to the left of the `left` interface or
        to the right of the `right` interface, or somewhere in between.

        Args:
            left: The left interface.
            right: The right interface, equal to left if not specified.

        Returns:
            out: A string representing where the end point is ('L' - left,
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

    def get_start_point(self, left: float, right: float | None = None) -> str:
        """Return the start point of the path as a string.

        The start point is either to the left of the `left` interface or
        to the right of the `right` interface.

        Args:
            left: The left interface.
            right: The right interface, equal to left if not specified.

        Returns:
            out: A string representing where the start point is ('L' - left,
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

    def get_shooting_point(self, rgen: Generator) -> tuple[System, int]:
        ### TODO: probably need an unittest for this to check if correct.
        ### idx = rgen.random_integers(1, self.length - 2)
        idx = rgen.integers(1, self.length - 1)
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

    def get_path_data(
        self, status: str, interfaces: list[float]
    ) -> dict[str, Any]:
        """Return information about the path.

        Args:
            status: The current status of the path.
            interfaces:: The interfaces for the simulation.

        Returns:
            path_info: A dict with information about the path.

        """
        path_info: dict[str, Any] = {
            "generated": self.generated,
            "status": status,
            "length": self.length,
            "ordermax": self.ordermax,
            "ordermin": self.ordermin,
            "weights": self.weights,
        }

        start, end, middle, _ = self.check_interfaces(interfaces)
        path_info["interface"] = (start, middle, end)
        return path_info

    def get_move(self) -> str | None:
        """Return the move used to generate the path."""
        if self.generated is None:
            return None
        return self.generated[0]

    def success(self, target_interface: float) -> bool:
        """Check if the path is successful.

        The check is based on the maximum order parameter and the value
        of `target_interface`. It is successful if the maximum order parameter
        is greater than `target_interface`.

        Parameters
        ----------
        target_interface : float
            The value for which the path is successful, i.e. the
            "target_interface" interface.

        """
        return self.ordermax[0] > target_interface

    def __iadd__(self, other: Path) -> Path:
        """Append phase points from another path, `self += other`.

        Args:
            other: The object to add path data from.

        Returns:
            self: The updated path object.

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
        """Return a copy of the path."""
        new_path = self.empty_path()
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
        # TODO: The path should not modify the system?
        system.vel_rev = not system.vel_rev

    def reverse(
        self, order_function: OrderParameter | None, rev_v: bool = True
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
        new_path = self.empty_path()
        new_path.weights = self.weights
        new_path.maxlen = self.maxlen
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

    def empty_path(self, **kwargs) -> Path:
        """Return an empty path of same class as the current one."""
        maxlen = kwargs.get("maxlen", DEFAULT_MAXLEN)
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

    def delete(self, idx: int) -> None:
        """Remove the specified phase point from the path."""
        del self.phasepoints[idx]

    def sorting(
        self, key: str, reverse: bool = False
    ):  # TODO: Can this be removed?
        """Re-order the phase points according to the given key.

        Args:
            key: The attribute we will sort according to.
            reverse: If this is False, the sorting is from big to small.

        Yields:
            The ordered phase points from the path.

        """
        if key in ("ekin", "vpot"):
            # TODO: Particles are gone, remove this method
            sort_after = [getattr(i, key)[0] for i in self.phasepoints]
        elif key == "order":
            sort_after = [getattr(i, key)[0] for i in self.phasepoints]
        else:
            sort_after = [getattr(i, key) for i in self.phasepoints]
        idx = np.argsort(sort_after)
        if reverse:
            idx = idx[::-1]
        self.phasepoints = [self.phasepoints[i] for i in idx]

    def update_energies(
        self, ekin: np.ndarray | list[float], vpot: np.ndarray | list[float]
    ) -> None:
        """Update the energies for the phase points.

        This method is useful in cases where the energies are
        read from external engines and returned as a list of
        floats.

        Args:
            ekin : The kinetic energies to set.
            vpot : The potential energies to set.

        """
        if len(ekin) != len(vpot):
            logger.debug(
                "Kinetic and potential energies have different length."
            )
        if len(ekin) != len(self.phasepoints):
            logger.debug(
                "Length of kinetic energy and phase points differ %d != %d.",
                len(ekin),
                len(self.phasepoints),
            )
        if len(vpot) != len(self.phasepoints):
            logger.debug(
                "Length of potential energy and phase points differ %d != %d.",
                len(vpot),
                len(self.phasepoints),
            )
        for i, phasepoint in enumerate(self.phasepoints):
            try:
                vpoti = vpot[i]
            except IndexError:
                logger.warning(
                    "Ran out of potential energies, setting to None."
                )
                vpoti = None
            try:
                ekini = ekin[i]
            except IndexError:
                logger.warning("Ran out of kinetic energies, setting to None.")
                ekini = None
            phasepoint.vpot = vpoti
            phasepoint.ekin = ekini


def paste_paths(
    path_back: Path,
    path_forw: Path,
    overlap: bool = True,
    maxlen: int | None = None,
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
        new_path: The resulting path from the merge.

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


def load_trajtxt(dirname: str) -> dict[str, Any]:  # TODO: CAN THIS BE REMOVED?
    """Load the information in traj.txt."""
    traj_file_name = os.path.join(dirname, "traj.txt")
    with PathExtFile(traj_file_name, "r") as trajfile:
        # Just get the first trajectory:
        traj = next(trajfile.load())

        # Update trajectory to use full path names:
        for i, snapshot in enumerate(traj["data"]):
            config = os.path.join(dirname, snapshot[1])
            traj["data"][i][1] = config
            reverse = int(snapshot[3]) == -1
            idx = int(snapshot[2])
            traj["data"][i][2] = idx
            traj["data"][i][3] = reverse
        return traj


def load_ordertxt(
    dirname: str,
) -> dict[str, np.ndarray]:  # TODO: CAN THIS BE REMOVED?
    """Load order_txt."""
    order_file_name = os.path.join(dirname, "order.txt")
    with OrderPathFile(order_file_name, "r") as orderfile:
        order = next(orderfile.load())
        return order["data"][:, 1:]


def load_path(pdir: str) -> Path:
    """Load path."""
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


def _check_path(
    path: Path, path_ensemble: Any, warning: bool = True
) -> tuple[bool, str]:
    """Run some checks for the path.

    Parameters
    ----------
    path : object like :py:class:`.PathBase`
        The path we are to set up/fill.
    path_ensemble : object like :py:class:`.PathEnsemble`
        The path ensemble the path could be added to.
    warning : boolean, optional
        If True, it output warnings, else only debug info.

    """
    start, end, _, cross = path.check_interfaces(path_ensemble.interfaces)
    accept = True
    status = "ACC"
    msg = "Initial path for %s is accepted."
    if start is None or start not in path_ensemble.start_condition:
        msg = "Initial path for %s starts at the wrong interface!"
        status = "SWI"
        accept = False
    if end not in ("R", "L"):
        msg = "Initial path for %s ends at the wrong interface!"
        status = "EWI"
        accept = False
    if not cross[1]:
        msg = "Initial path for %s does not cross the middle interface!"
        status = "NCR"
        accept = False

    if not accept:
        if warning:
            logger.critical(msg, path_ensemble.ensemble_name)
        else:
            logger.debug(msg, path_ensemble.ensemble_name)

    path.status = status
    return accept, status


def _load_energies_for_path(path: Path, dirname: str) -> None:
    """Load energy data for a path.

    Parameters
    ----------
    path : object like :py:class:`.PathBase`
        The path we are to set up/fill.
    dirname : string
        The path to the directory with the input files.

    Returns
    -------
    None, but may add energies to the path.

    """
    # Get energies if any:
    energy_file_name = os.path.join(dirname, "energy.txt")
    try:
        with EnergyPathFile(energy_file_name, "r") as energyfile:
            energy = next(energyfile.load())
            path.update_energies(
                energy["data"]["ekin"], energy["data"]["vpot"]
            )
    except FileNotFoundError:
        pass


def load_paths_from_disk(config: dict[str, Any]) -> list[Path]:
    """Load paths from disk."""
    load_dir = config["simulation"]["load_dir"]
    paths = []
    for pnumber in config["current"]["active"]:
        new_path = load_path(os.path.join(load_dir, str(pnumber)))
        status = "re" if "restarted_from" in config["current"] else "ld"
        ### TODO: important for shooting move if 'ld' is set. need a smart way
        ### to remember if status is 'sh' or 'wf' etc. maybe in the toml file.
        new_path.generated = (status, float("nan"), 0, 0)
        paths.append(new_path)
        # assign pnumber
        paths[-1].path_number = pnumber
    return paths
