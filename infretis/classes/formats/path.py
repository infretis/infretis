import logging
import os

from infretis.classes.formats.formatter import (
    FileIO,
    OutputFormatter,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


__all__ = [
    "PathExtFormatter",
    "PathExtFile",
]


class PathExtFormatter(OutputFormatter):
    """A class for formatting external trajectories.

    The external trajectories as stored as files and this path
    formatter includes the location of these files.

    Attributes
    ----------
    FMT : string
        The string to use for the formatting.

    """

    FMT = "{:>10}  {:>20s}  {:>10}  {:>5}"

    def __init__(self):
        """Initialise the PathExtFormatter formatter."""
        header = {
            "labels": ["Step", "Filename", "index", "vel"],
            "width": [10, 20, 10, 5],
            "spacing": 2,
        }

        super().__init__("PathExtFormatter", header=header)
        self.print_header = False

    def format(self, step, data):
        """Format path data for external paths.

        Parameters
        ----------
        step : integer
            The current simulation step.
        data : list
            Here, ``data[0]`` is assumed to be an object like
            :py:class:`.Path`` and ``data[1]`` a string containing the
            status of this path.

        Yields
        ------
        out : string
            The trajectory as references to files.

        """
        path, status = data[0], data[1]
        if not path:  # E.g. when null-moves are False.
            return
        yield f"# Cycle: {step}, status: {status}"
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            filename, idx = phasepoint.particles.get_pos()
            filename_short = os.path.basename(filename)
            if idx is None:
                idx = 0
            vel = -1 if phasepoint.particles.get_vel() else 1
            yield self.FMT.format(i, filename_short, idx, vel)

    @staticmethod
    def parse(line):
        """Parse the line data by splitting text on spaces.

        Parameters
        ----------
        line : string
            The line to parse.

        Returns
        -------
        out : list
            The columns of data.

        """
        return [i for i in line.split()]


class PathExtFile(FileIO):
    """A class for writing path data."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the path writer with correct format for external paths."""
        super().__init__(
            filename, file_mode, PathExtFormatter(), backup=backup
        )
