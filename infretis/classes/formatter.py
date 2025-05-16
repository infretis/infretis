"""Handle formatting of files."""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from infretis.core.core import make_dirs

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())

LOG_FMT = "[%(levelname)s]: %(message)s"
LOG_DEBUG_FMT = (
    "[%(levelname)s] [%(name)s, %(funcName)s() at"
    " line %(lineno)d]: %(message)s"
)
if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable

    from infretis.classes.path import Path as InfPath


def _read_line_data(
    ncol: int, stripline: str, line_parser: Callable[[str], Any]
) -> Tuple[Any, bool, int]:
    """Read data for :py:func:`.read_some_lines.`.

    Args
        ncol: The expected number of columns to read. If this is less than 1
            it is not yet set. Note that we skip data which appear
            inconsistent. A warning will be issued about this.
        stripline: The line to read. Note that we assume that leading and
            trailing spaces have been removed.
        line_parser: A method for parsing a single line.

    Returns:
        A tuple containing:
            - The line read.
            - A boolean flag, True if data was read successfully.
            - The number of columns read. If this is -1 the line
                was malformed.
    """
    if line_parser is None:
        # Just return data without any parsing:
        return stripline, True, ncol
    try:
        linedata = line_parser(stripline)
    except (ValueError, IndexError):
        return None, False, -1
    newcol = len(linedata)
    if ncol == -1:  # first item
        ncol = newcol
    if newcol == ncol:
        return linedata, True, ncol
    # We assume that this is line is malformed --- skip it!
    return None, False, -1


def read_some_lines(
    filename: str, line_parser: Callable[[str], Any], block_label: str = "#"
) -> Iterable[Dict[str, Any]]:
    """Open a file and try to read as many lines as possible.

    This method will read a file using the given `line_parser`.
    If the given `line_parser` fails at a line in the file,
    `read_some_lines` will stop here. Further, this method
    will read data in blocks and yield a block when a new
    block is found. A special string (`block_label`) is assumed to
    identify the start of blocks.

    Args:
        filename: This is the name/path of the file to open and read.
        line_parser: This is a function which knows how to translate
            a given line to a desired internal format. If not given,
            a simple `float()` will be used.
        block_label: This string is used to identify blocks.

    Yields:
        The data read from the file, arranged in a dictionary.
        The dictionary has a key for comments and a key for the
        data.
    """
    ncol = -1  # The number of columns
    new_block: Dict[str, Any] = {"comment": [], "data": []}
    yield_block = False
    read_comment = False
    with open(filename, encoding="utf-8") as fileh:
        for i, line in enumerate(fileh):
            stripline = line.strip()
            if stripline.startswith(block_label):
                # this is a comment, then a new block will follow,
                # unless this is a multi-line comment.
                if read_comment:  # part of multi-line comment...
                    new_block["comment"].append(stripline)
                else:
                    if yield_block:
                        # Yield the current block
                        yield_block = False
                        yield new_block
                    new_block = {"comment": [stripline], "data": []}
                    yield_block = True  # Data has been added
                    ncol = -1
                    read_comment = True
            else:
                read_comment = False
                data, _yieldb, _ncol = _read_line_data(
                    ncol, stripline, line_parser
                )
                if data:
                    new_block["data"].append(data)
                    ncol = _ncol
                    yield_block = _yieldb
                else:
                    logger.warning(
                        'Skipped malformed data in "%s", line: %i',
                        filename,
                        i,
                    )
    # if the block has not been yielded, yield it
    if yield_block:
        yield_block = False
        yield new_block


def _make_header(labels: List[str], width: List[int], spacing: int = 1) -> str:
    """Format a table header with the given labels.

    Args
        labels: The strings to use for the table header.
        width: The widths to use for the table.
        spacing: The spacing between the columns in the table

    Returns
        A header for the table.
    """
    heading = []
    for i, col in enumerate(labels):
        try:
            wid = width[i]
        except IndexError:
            wid = width[-1]
        if i == 0:
            fmt = f"# {{:>{wid - 2}s}}"
        else:
            fmt = f"{{:>{wid}s}}"
        heading.append(fmt.format(col))
    str_white = " " * spacing
    return str_white.join(heading)


class OutputFormatter:
    """A generic class for formatting output.

    Attributes:
        name: A string which identifies the formatter.
        header: A string representing the header (or table heading)
            with information about the output data.
        print_header: If True, the header is printed. This is useful
            for classes making use of the formatter to determine if
            they should write out the header or not.
    """

    _FMT = "{}"

    def __init__(self, name: str, header: Optional[Dict[str, Any]] = None):
        """Initialise the formatter.

        Args:
            name: A string that identifies the output type of this formatter.
            header: A dictionary representing the header for the output data.
        """
        self.name = name
        self._header = "infretis output"
        self.print_header = True
        if header is not None:
            if "width" in header and "labels" in header:
                self._header = _make_header(
                    header["labels"],
                    header["width"],
                    spacing=header.get("spacing", 1),
                )
            else:
                self._header = header.get("text", "infretis output")
        else:
            self.print_header = False

    @property
    def header(self) -> str:
        """Define the header as a property."""
        return self._header

    @header.setter
    def header(self, value: str) -> None:
        """Set the header."""
        self._header = value

    def format(self, step: int, data: Any) -> Iterable[str]:
        """Use the formatter to generate output.

        Args:
            step: The current step number for generating the output.
            data: The data to format; assumed to be iterable.
        """
        out = [f"{step}"]
        for i in data:
            out.append(self._FMT.format(i))
        yield " ".join(out)

    @staticmethod
    def parse(line: str) -> Union[List[int], List[float], List[str]]:
        """Parse formatted data to numbers.

        This method is intended to be the "inverse" of the :py:meth:`.format`
        method. It assume that we read floats from columns in a file.
        One input line corresponds to a "row" of data.

        Args:
            line: The string we will parse.

        Returns:
            The parsed input data.
        """
        return [
            int(col) if i == 0 else float(col)
            for i, col in enumerate(line.split())
        ]

    def load(self, filename: str) -> Iterable[Dict[str, Any]]:
        """Read generic data from a file.

        This method will read entire blocks of data from a file into
        memory. This will be slow for large files.

        Args:
            filename: The path to the file to read.

        Yields:
            This is the data contained in the file. It is return as
            a dictionary where the key `"data"` contains the data
            read.

        See Also:
            :py:func:`.read_some_lines`.
        """
        for blocks in read_some_lines(filename, self.parse):
            data_dict = {"comment": blocks["comment"], "data": blocks["data"]}
            yield data_dict

    def __str__(self) -> str:
        """Return basic info about the formatter."""
        return self.name


class OrderFormatter(OutputFormatter):
    """A class for formatting order parameter data.

    The format for the order file is column-based and the columns are:

    1) Time.

    2) Main order parameter.

    3) Collective variable 1

    4) Collective variable 2

    5) ...

    """

    # Format for order files. Note that we don't know how many parameters
    # we need to format yet.
    ORDER_FMT = ["{:>10d}", "{:>12.6f}"]

    def __init__(self, name: str = "OrderFormatter"):
        """Initialise the formatter."""
        header = {"labels": ["Time", "Orderp"], "width": [10, 12]}
        super().__init__(name, header=header)

    def format_data(self, step: int, orderdata: List[float]) -> str:
        """Format order parameter data.

        Args:
            step: This is the current step number.
            orderdata: These are the order parameters to format.

        Yields:
            The formatted data as a string.

        """
        towrite = [self.ORDER_FMT[0].format(step)]
        for orderp in orderdata:
            towrite.append(self.ORDER_FMT[1].format(orderp))
        out = " ".join(towrite)
        return out

    def format(self, step: int, data: List[float]) -> Iterable[str]:
        """Yield formatted order parameters. See :py:meth:`.format_data`."""
        yield self.format_data(step, data)

    def load(self, filename: str) -> Iterable[Dict[str, Any]]:
        """Read order parameter data from the given file."""
        for blocks in read_some_lines(filename, self.parse):
            data_dict = {
                "comment": blocks["comment"],
                "data": np.array(blocks["data"]),
            }
            yield data_dict


class OrderPathFormatter(OrderFormatter):
    """A class for formatting order parameter data for paths."""

    def __init__(self) -> None:
        """Initialise the formatter."""
        super().__init__(name="OrderPathFormatter")
        self.print_header = False

    def format(self, step: int, data: List[Any]) -> Iterable[str]:
        """Format the order parameter dat for a path.

        Args:
            step: The cycle number we are creating output for.
            data: A tuple on the form `(Path, status)` where `Path` is
                the :py:class:`.PathBase` to extract order parameters for
                and `status` is the string representing the status of the path.

        Yields:
            The formatted order parameters.
        """
        path, status = data[0], data[1]
        if not path:  # E.g. when null-moves are False.
            return
        move = path.generated
        yield f"# Cycle: {step}, status: {status}, move: {move}"
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            yield self.format_data(i, phasepoint.order)


class OutputBase(metaclass=ABCMeta):
    """A generic class for handling output.

    Attributes:
        formatter: The object responsible for formatting output.
        first_write: Determines if we have written something yet, or
            if this is the first write.
    """

    def __init__(self, formatter: OutputFormatter):
        """Create the object and attach a formatter."""
        self.formatter = formatter
        self.first_write = True

    def output(self, step: int, data: List[Any]) -> Any:
        """Use the formatter to write data to the file.

        Args:
            step: The current step number.
            data: The data we are going to format and write.
        """
        if self.first_write and self.formatter.print_header:
            self.first_write = False
            self.write(self.formatter.header)
        for line in self.formatter.format(step, data):
            self.write(line)

    @abstractmethod
    def write(self, towrite: str, end: str = "\n") -> bool:
        """Write a string to the output defined by this class.

        Parameters:
            towrite: The string to write.
            end: A "terminator" for the given string.

        Returns:
            True if we managed to write, False otherwise.
        """

    def __str__(self) -> str:
        """Return basic info about the formatter."""
        return f"{self.__class__.__name__}\n\t* Formatter: {self.formatter}"


class FileIO(OutputBase):
    """A generic class for handling IO with files.

    This should define how InfRETIS is handling input and output.

    Attributes:
        filename: Path to the file to read or write.
        file_mode: Specifies the mode in which the file is opened.
        backup: If False, overwrite existing files. If True, backup
            existing files before writing.
        fileh: The file handle we are interacting with.
        last_flush: The previous time for flushing to the file.
        FILE_FLUSH: The interval (in seconds) for flushing to the file.
            That is, we will flush if the time since the last flush is
            larger than this value. This is only checked in relation to
            writing.
    """

    target = "file"
    FILE_FLUSH = 1  # Interval for flushing files in seconds.

    def __init__(
        self,
        filename: str,
        file_mode: str,
        formatter: OutputFormatter,
        backup: bool = True,
    ):
        """Set up the FileIO object.

        Args:
            filename: The path to the file to open or read.
            file_mode: Specifies the mode for opening the file.
            formatter: The object responsible for formatting output.
            backup: If True, create backup files. Otherwise, overwrite.
        """
        super().__init__(formatter)
        self.filename = filename
        self.file_mode = file_mode
        if backup not in (True, False):
            logger.info("Setting backup to default: True")
            self.backup = True
        else:
            self.backup = backup
        self.fileh: Union[IO[Any], None] = None
        if self.file_mode.startswith("a") and self.formatter is not None:
            self.formatter.print_header = False
        self.last_flush: Optional[datetime] = None

    def open_file_read(self) -> Optional[IO[Any]]:
        """Open a file for reading."""
        if not self.file_mode.startswith("r"):
            raise ValueError(
                f'Inconsistent file mode "{self.file_mode}" ' "for reading"
            )
        try:
            self.fileh = open(self.filename, self.file_mode)
        except OSError as error:
            if "energy" not in self.filename:
                logger.critical(
                    'Could not open file "%s" for reading', self.filename
                )
                logger.critical(
                    "I/O error ({%d}): {%s}", error.errno, error.strerror
                )
        return self.fileh

    def open_file_write(self) -> Optional[IO[Any]]:
        """Open a file for writing and handle backup."""
        if self.file_mode[0] not in ("a", "w"):
            raise ValueError(
                f'Inconsistent file mode "{self.file_mode}" ' "for writing"
            )
        try:
            if os.path.isfile(self.filename):
                if self.file_mode.startswith("a"):
                    logger.info(
                        'Appending to existing file "%s"', self.filename
                    )
                else:
                    logger.debug(
                        'Overwriting existing file "%s"', self.filename
                    )
            self.fileh = open(self.filename, self.file_mode)
        except OSError as error:  # pragma: no cover
            logger.critical(
                'Could not open file "%s" for writing', self.filename
            )
            logger.critical("I/O error (%d): %d", error.errno, error.strerror)
        return self.fileh

    def open(self) -> Optional[IO[Any]]:
        """Open a file for reading or writing."""
        if self.fileh is not None:
            logger.debug(
                "%s asked to open file, but it has already opened a file.",
                self.__class__.__name__,
            )
            return self.fileh
        if self.file_mode[0] in ("r",):
            return self.open_file_read()
        if self.file_mode[0] in ("a", "w"):
            return self.open_file_write()
        raise ValueError(f'Unknown file mode "{self.file_mode}"')

    def load(self) -> Iterable[Dict[str, Any]]:
        """Read blocks or lines from the file."""
        return self.formatter.load(self.filename)

    def write(self, towrite: str, end: str = "\n") -> bool:
        """Write a string to the file.

        Args:
            towrite: The string to output to the file.
            end: A string appended to `towrite` when writing.
                Can be used to print a new line after the input `towrite`.

        Returns:
            True if we managed to write, False otherwise.
        """
        status = False
        if towrite is None:
            return status
        if self.fileh is not None and not self.fileh.closed:
            try:
                if end is not None:
                    self.fileh.write(f"{towrite}{end}")
                    status = True
                else:
                    self.fileh.write(towrite)
                    status = True
            except OSError as error:  # pragma: no cover
                msg = f"Write I/O error ({error.errno}): {error.strerror}"
                logger.critical(msg)
            if self.last_flush is None:
                self.flush()
                self.last_flush = datetime.now()
            delta = datetime.now() - self.last_flush
            if delta.total_seconds() > self.FILE_FLUSH:  # pragma: no cover
                self.flush()
                self.last_flush = datetime.now()
            return status
        if self.fileh is not None and self.fileh.closed:
            logger.warning("Ignored writing to closed file %s", self.filename)
        if self.fileh is None:
            logger.critical(
                "Attempting to write to empty file handle for file %s",
                self.filename,
            )
        return status

    def close(self) -> None:
        """Close the file."""
        if self.fileh is not None and not self.fileh.closed:
            try:
                self.flush()
            finally:
                self.fileh.close()

    def flush(self) -> None:
        """Flush file buffers to file."""
        if self.fileh is not None and not self.fileh.closed:
            self.fileh.flush()
            os.fsync(self.fileh.fileno())

    def output(self, step: int, data: List[Any]) -> Any:
        """Open file before first write."""
        if self.first_write:
            self.open()
        return super().output(step, data)

    def __del__(self) -> None:
        """Close the file in case the object is deleted."""
        self.close()

    def __enter__(self):
        """Context manager for opening the file."""
        self.open()
        return self

    def __exit__(self, *args) -> None:
        """Context manager for closing the file."""
        self.close()

    def __iter__(self):
        """Make it possible to iterate over lines in the file."""
        return self

    def __next__(self):
        """Let the file object handle the iteration."""
        if self.fileh is None:
            raise StopIteration
        if self.fileh.closed:
            raise StopIteration
        return next(self.fileh)

    def __str__(self) -> str:
        """Return info about this class as a string."""
        msg = [f'FileIO (file: "{self.filename}")']
        if self.fileh is not None and not self.fileh.closed:
            msg += ["\t* File is open"]
            msg += [f"\t* Mode: {self.fileh.mode}"]
        msg += [f"\t* Formatter: {self.formatter}"]
        return "\n".join(msg)


class OrderPathFile(FileIO):
    """A class for handling order parameter path files."""

    def __init__(self, filename: str, file_mode: str, backup: bool = True):
        """Create the order path file with correct formatter."""
        super().__init__(
            filename, file_mode, OrderPathFormatter(), backup=backup
        )


class EnergyFormatter(OutputFormatter):
    """A class for formatting energy data.

    This class handles formatting of energy data.
    The data is formatted in 5 columns:

    1) Time, i.e. the step number.

    2) Potential energy.

    3) Kinetic energy.

    4) Total energy, should equal the sum of the two previous columns.

    5) Temperature.
    """

    # Format for the energy files:
    ENERGY_FMT = ["{:>10d}"] + 5 * ["{:>14.6f}"]
    ENERGY_TERMS = ("vpot", "ekin", "etot", "temp")
    HEADER = {
        "labels": ["Time", "Potential", "Kinetic", "Total", "Temperature"],
        "width": [10, 14],
    }

    def __init__(self, name: str = "EnergyFormatter"):
        """Initialise the formatter for energy."""
        super().__init__(name, header=self.HEADER)

    def apply_format(self, step: int, energy: Any) -> str:
        """Apply the energy format.

        Args:
            step: The current simulation step.
            energy: A dictionary with energy terms to format.

        Returns:
            A string with the formatted energy data.
        """
        towrite = [self.ENERGY_FMT[0].format(step)]
        for i, key in enumerate(self.ENERGY_TERMS):
            value = energy.get(key, None)
            if value is None:
                towrite.append(self.ENERGY_FMT[i + 1].format(float("nan")))
            else:
                towrite.append(self.ENERGY_FMT[i + 1].format(float(value)))
        return " ".join(towrite)

    def format(self, step: int, data: Any) -> Iterable[str]:
        """Yield formatted energy data. See :py:meth:.`apply_format`."""
        yield self.apply_format(step, data)

    def load(self, filename: str) -> Iterable[Dict[str, Any]]:
        """Load entire energy blocks into memory.

        Args:
            filename: The path to the file to open.

        Yields:
            The energy data read from the file, stored in a dictionary.
            This is for convenience so that each energy term can be accessed
            using keys on the form: `data_dict["data"][key]`.
        """
        for blocks in read_some_lines(filename, line_parser=self.parse):
            data = np.array(blocks["data"])
            col = tuple(data.shape)
            col_max = min(col[1], len(self.ENERGY_TERMS) + 1)
            data_dict = {
                "comment": blocks["comment"],
                "data": {"time": data[:, 0]},
            }
            for i in range(col_max - 1):
                data_dict["data"][self.ENERGY_TERMS[i]] = data[:, i + 1]
            yield data_dict


class EnergyPathFormatter(EnergyFormatter):
    """A class for formatting energy data for paths."""

    HEADER = {
        "labels": ["Time", "Potential", "Kinetic", "Total", "Temperature"],
        "width": [10, 14]
        }

    def __init__(self):
        """Initialise."""
        super().__init__(name="EnergyPathFormatter")
        self.print_header = False

    def format(self, step: int, data: Any) -> Iterable[str]:
        """Format the order parameter data from a path.

        Args:
            step: The cycle number we are creating output for.
            data: A tuple containing the path (as an object like
                :py:class:`.PathBase`) as the first element and a string
                with the status for the path as the second element.

        Yields:
            The strings to be written.
        """
        path, status = data[0], data[1]
        if not path:  # when nullmoves = False
            return
        move = path.generated
        yield f"# Cycle: {step}, status: {status}, move: {move}"
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            energy = {}
            for key in self.ENERGY_TERMS:
                energy[key] = getattr(phasepoint, key, None)
            yield self.apply_format(i, energy)


class EnergyPathFile(FileIO):
    """A class for handling energy path files."""

    def __init__(self, filename: str, file_mode: str, backup: bool = True):
        """Create the file object and attach the energy formatter."""
        super().__init__(
            filename, file_mode, EnergyPathFormatter(), backup=backup
        )


class PathExtFormatter(OutputFormatter):
    """A class for formatting trajectories.

    The trajectories as stored as files and this formatter
    creates a file that includes the location to these files.
    """

    FMT = "{:>10}  {:>20s}  {:>10}  {:>5}"  # For formatting the paths.

    def __init__(self):
        """Initialise the PathExtFormatter formatter."""
        header = {
            "labels": ["Step", "Filename", "index", "vel"],
            "width": [10, 20, 10, 5],
            "spacing": 2,
        }

        super().__init__("PathExtFormatter", header=header)
        self.print_header = False

    def format(self, step: int, data: List[Any]) -> Iterable[str]:
        """Format path data for external paths.

        Args:
            step: The current simulation step.
            data : A tuple where `data[0]` is assumed to be the path
                (as an object like :py:class:`.Path`) and `data[1]`
                a string containing the status of this path.

        Yields:
            The trajectory as references to files.
        """
        path, status = data[0], data[1]
        if not path:  # E.g. when null-moves are False.
            return
        yield f"# Cycle: {step}, status: {status}"
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            filename, idx = phasepoint.config
            filename_short = os.path.basename(filename)
            if idx is None:
                idx = 0
            vel = -1 if phasepoint.vel_rev else 1
            yield self.FMT.format(i, filename_short, idx, vel)

    @staticmethod
    def parse(line: str) -> List[str]:
        """Parse the line data by splitting the given text on spaces."""
        return [i for i in line.split()]


class PathExtFile(FileIO):
    """A class for writing path data."""

    def __init__(self, filename: str, file_mode: str, backup: bool = True):
        """Create the path writer with correct format for external paths."""
        super().__init__(
            filename, file_mode, PathExtFormatter(), backup=backup
        )


class FormattersEntry(TypedDict):
    """To store formatters and output files together."""

    fmt: OutputFormatter  # The formatter to use.
    file: str  # The file to write.


class PathStorage(OutputBase):
    """A class for handling storage of external trajectories.

    Attributes:
        target: Determines the target for this output class.
            Here it will be a file archive (i.e., a directory
            based collection of files).
        formatters: This dict contains the formatters for writing path data,
            with default filenames used for them.
        out_dir_fmt: A format to use for creating directories within the
            archive. This one is applied to the step number for the output.
    """

    target = "file-archive"
    formatters: Dict[str, FormattersEntry] = {
        "order": {"fmt": OrderPathFormatter(), "file": "order.txt"},
        "energy": {"fmt": EnergyPathFormatter(), "file": "energy.txt"},
        "traj": {"fmt": PathExtFormatter(), "file": "traj.txt"},
    }
    out_dir_fmt = "{}"

    def __init__(self, keep_traj_fnames: list = []):
        """Set up the storage.

        Note:
            No formatters are passed to the parent class. This is because
            this class is less flexible and only intended to do one
            thing: write path data for external trajectories.
        """
        formatter = OutputFormatter("empty formatter", header=None)
        super().__init__(formatter)
        self.keep_traj_fnames = keep_traj_fnames

    def output_path_files(
        self, step: int, data: List[Any], target_dir: str
    ) -> List[Tuple[str, str]]:
        """Write the output files for energy, path and order parameter.

        Args:
            step: The current simulation step.
            data: A tuple containing:
                - The path as an object like :py:class:`.Path`.
                - A string string containing the status of this path.
            target_dir: The path to where we archive the files.

        Returns:
            The files created as a list of tuples. Each tuple contain:
                - The full path to the file.
                - A relative path to the file. The relative path is useful
                  for organizing internally in archives.
        """
        path, status = data[0], data[1]
        files = []
        for key, val in self.formatters.items():
            logger.debug("Storing: %s", key)
            fmt = val["fmt"]
            full_path = os.path.join(target_dir, val["file"])
            relative_path = os.path.join(
                self.out_dir_fmt.format(step), val["file"]
            )
            files.append((full_path, relative_path))
            with open(full_path, mode="w", encoding="utf8") as output:
                for line in fmt.format(step, (path, status)):
                    output.write(f"{line}\n")
        return files

    @staticmethod
    def _move_path(
        path: InfPath,
        target_dir: str,
        keep_traj_fnames: list,
        prefix: Optional[str] = None,
    ) -> InfPath:
        """Copy a path to a given target directory.

        Args:
            path: The path to copy.
            target_dir: The location where we are moving the path to.
            keep_traj_fnames: A list file extensions that are matched aginst
              the source directories in which the trajectories are stored.
              File extensions that match the pattern are also stored.
            prefix: A prefix for the file names of copied files.

        Returns:
            A copy of the input path.
        """
        path_copy = path.copy()
        new_pos, source = _generate_file_names(
            path_copy, target_dir, prefix=prefix
        )
        # keep any files where extension match the patterns in keep_traj_fnames
        if keep_traj_fnames:
            for source_file in source.copy().keys():
                source_dir, source_fname = os.path.split(source_file)
                traj_name, traj_ext = os.path.splitext(source_fname)
                for ext in keep_traj_fnames:
                    new_fname = traj_name + ext
                    fpath = os.path.join(source_dir, new_fname)
                    if os.path.isfile(fpath):
                        source[fpath] = os.path.join(target_dir, new_fname)
        # Update positions:
        for pos, phasepoint in zip(new_pos, path_copy.phasepoints):
            phasepoint.config = (pos[0], pos[1])
            # phasepoint.particles.set_pos(pos)
        for src, dest in source.items():
            if src != dest:
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        logger.debug("Removing %s as it exists", dest)
                        os.remove(dest)
                logger.debug("Copy %s -> %s", src, dest)
                shutil.move(src, dest)
        return path_copy

    def output(self, step: int, data: Any) -> InfPath:
        """Format the path data and store the path.

        Args:
            step: The current simulation step.
            data: A dictionary containing the path and the directory to
                write to.

        Returns:
            A copy of the path (moved to the new directory).
        """
        # path_ensemble = data
        # path = path_ensemble.last_path
        # home_dir = path_ensemble.directory['home_dir'] + '/trajs'
        path = data["path"]
        home_dir = data["dir"]
        # This is the path on form: /path/to/000/traj/11
        archive_path = os.path.join(
            home_dir,
            f"{path.path_number}",
        )

        # To organize things we create a subfolder for storing the
        # files. This is on form: /path/to/000/traj/11/traj
        traj_dir = os.path.join(archive_path, "accepted")
        # Create the needed directories:
        make_dirs(traj_dir)
        # Write order, energy and traj files to the archive:
        _ = self.output_path_files(step, [path, "ACC"], archive_path)
        path = self._move_path(path, traj_dir, self.keep_traj_fnames)
        return path

    def write(self, towrite: str, end: str = "\n") -> bool:
        """We do not need the write method for this object."""
        logger.critical(
            '%s does *not* support the "write" method!',
            self.__class__.__name__,
        )
        return False

    def __str__(self) -> str:
        """Return basic info."""
        return f"{self.__class__.__name__} - archive writer."


def get_log_formatter(level: int) -> LogFormatter:
    """Select a log format based on a given level.

    Here, it is just used to get a slightly more verbose format for
    the debug level.

    Args:
        level: This integer defines the log level.

    Returns:
        An object that can be used as a formatter for a logger.
    """
    if level <= logging.DEBUG:
        return LogFormatter(LOG_DEBUG_FMT)
    return LogFormatter(LOG_FMT)


class LogFormatter(logging.Formatter):  # pragma: no cover
    """Hard-coded formatter for the log file."""

    def format(self, record):
        """Apply the log format."""
        out = logging.Formatter.format(self, record)
        return out


def _generate_file_names(
    path: InfPath, target_dir: str, prefix: Optional[str] = None
) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
    """Generate new file names for moving or copying paths.

    Args:
        path: The path object we are going to store.
        target_dir: The location where we are moving the path to.
        prefix: The prefix can be used to prefix the name of the files.

    Returns:
        A tuple containing:
            - A list with new file names.
            - A dict which defines the unique "source -> destination" for
              the copy/move operations.
    """
    source = {}
    new_pos = []
    for phasepoint in path.phasepoints:
        pos_file, idx = phasepoint.config
        if pos_file not in source:
            localfile = os.path.basename(pos_file)
            if prefix is not None:
                localfile = f"{prefix}{localfile}"
            dest = os.path.join(target_dir, localfile)
            source[pos_file] = dest
        dest = source[pos_file]
        new_pos.append((dest, idx))
    return new_pos, source
