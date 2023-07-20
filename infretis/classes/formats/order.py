import logging
import numpy as np
from infretis.classes.formats.formatter import OutputFormatter, FileIO, read_some_lines
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


__all__ = [
    'OrderFormatter',
    'OrderPathFormatter',
    'OrderFile',
    'OrderPathFile',
]


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
    ORDER_FMT = ['{:>10d}', '{:>12.6f}']

    def __init__(self, name='OrderFormatter'):
        """Initialise a `OrderFormatter` formatter."""
        header = {'labels': ['Time', 'Orderp'], 'width': [10, 12]}
        super().__init__(name, header=header)

    def format_data(self, step, orderdata):
        """Format order parameter data.

        Parameters
        ----------
        step : int
            This is the current step number.
        orderdata : list of floats
            These are the order parameters.

        Yields
        ------
        out : string
            The strings to be written.

        """
        towrite = [self.ORDER_FMT[0].format(step)]
        for orderp in orderdata:
            towrite.append(self.ORDER_FMT[1].format(orderp))
        out = ' '.join(towrite)
        return out

    def format(self, step, data):
        """Yield formatted order parameters. See :py:meth:`.format_data`."""
        yield self.format_data(step, data)

    def load(self, filename):
        """Read order parameter data from a file.

        Since this class defines how the data is formatted it is also
        convenient to have methods for reading the data defined here.
        This method will read entire blocks of data from a file into
        memory. This will be slow for large files and this method
        could be converted to also yield the individual "rows" of
        the blocks, rather than the full blocks themselves.

        Parameters
        ----------
        filename : string
            The path/file name of the file we want to open.

        Yields
        ------
        data_dict : dict
            This is the order parameter data in the file.

        See Also
        --------
        :py:func:`.read_some_lines`.

        """
        for blocks in read_some_lines(filename, self.parse):
            data_dict = {'comment': blocks['comment'],
                         'data': np.array(blocks['data'])}
            yield data_dict


class OrderPathFormatter(OrderFormatter):
    """A class for formatting order parameter data for paths."""

    def __init__(self):
        """Initialise."""
        super().__init__(name='OrderPathFormatter')
        self.print_header = False

    def format(self, step, data):
        """Format the order parameter data from a path.

        Parameters
        ----------
        step : int
            The cycle number we are creating output for.
        data : tuple or list
            Here, data[0] contains a object
            like :py:class:`.PathBase` which is the path we are
            creating output for. data[1] contains the status for
            this path.

        Yields
        ------
        out : string
            The strings to be written.

        """
        path, status = data[0], data[1]
        if not path:  # E.g. when null-moves are False.
            return
        move = path.generated
        yield '# Cycle: {}, status: {}, move: {}'.format(step, status, move)
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            yield self.format_data(i, phasepoint.order)


class OrderFile(FileIO):
    """A class for handling PyRETIS order parameter files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the order file with correct formatter."""
        super().__init__(filename, file_mode, OrderFormatter(), backup=backup)


class OrderPathFile(FileIO):
    """A class for handling PyRETIS order parameter path files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the order path file with correct formatter."""
        super().__init__(filename, file_mode, OrderPathFormatter(),
                         backup=backup)
