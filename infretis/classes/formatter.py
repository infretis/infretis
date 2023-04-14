from infretis.classes.fileio import FileIO, OutputBase
from infretis.classes.pathensemble import PathEnsemble

import os
import numpy as np
import errno
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LOG_FMT = '[%(levelname)s]: %(message)s'
LOG_DEBUG_FMT = ('[%(levelname)s] [%(name)s, %(funcName)s() at'
                 ' line %(lineno)d]: %(message)s')

def _make_header(labels, width, spacing=1):
    """Format a table header with the given labels.

    Parameters
    ----------
    labels : list of strings
        The strings to use for the table header.
    width : list of ints
        The widths to use for the table.
    spacing : int
        The spacing between the columns in the table

    Returns
    -------
    out : string
        A header for the table.

    """
    heading = []
    for i, col in enumerate(labels):
        try:
            wid = width[i]
        except IndexError:
            wid = width[-1]
        if i == 0:
            fmt = '# {{:>{}s}}'.format(wid - 2)
        else:
            fmt = '{{:>{}s}}'.format(wid)
        heading.append(fmt.format(col))
    str_white = ' ' * spacing
    return str_white.join(heading)


def apply_format(value, fmt):
    """Apply a format string to a given float value.

    Here we check the formatting of a float. We are *forcing* a
    *maximum length* on the resulting string. This is to avoid problems
    like: '{:7.2f}'.format(12345.7) which returns '12345.70' with a
    length 8 > 7. The intended use of this function is to avoid such
    problems when we are formatting numbers for tables. Here it is done
    by switching to an exponential notation. But note however that this
    will have implications for how many decimal places we can show.

    Parameters
    ----------
    value : float
        The float to format.
    fmt : string
        The format to use.

    Note
    ----
    This function converts numbers to have a fixed length. In some
    cases this may reduce the number of significant digits. Remember
    to also output your numbers without this format in case a specific
    number of significant digits is important!

    """
    maxlen = fmt.split(':')[1].split('.')[0]
    align = ''
    if not maxlen[0].isalnum():
        align = maxlen[0]
        maxlen = maxlen[1:]
    maxlen = int(maxlen)
    str_fmt = fmt.format(value)
    if len(str_fmt) > maxlen:  # switch to exponential:
        if value < 0:
            deci = maxlen - 7
        else:
            deci = maxlen - 6
        new_fmt = '{{:{0}{1}.{2}e}}'.format(align, maxlen, deci)
        return new_fmt.format(value)
    else:
        return str_fmt


def format_number(number, minf, maxf, fmtf='{0:<16.9f}', fmte='{0:<16.9e}'):
    """Format a number based on its size.

    Parameters
    ----------
    number : float
        The number to format.
    minf : float
        If the number is smaller than `minf` then apply the
        format with scientific notation.
    maxf : float
        If the number is greater than `maxf` then apply the
        format with scientific notation.
    fmtf : string, optional
        Format to use for floats.
    fmte : string, optional
        Format to use for scientific notation.

    Returns
    -------
    out : string
        The formatted number.

    """
    if minf <= number <= maxf:
        return fmtf.format(number)
    return fmte.format(number)



class OutputFormatter:
    """A generic class for formatting output from PyRETIS.

    Attributes
    ----------
    name : string
        A string which identifies the formatter.
    header : string
        A header (or table heading) with information about the
        output data.
    print_header : boolean
        Determines if we are to print the header or not. This is useful
        for classes making use of the formatter to determine if they
        should write out the header or not.

    """

    _FMT = '{}'

    def __init__(self, name, header=None):
        """Initialise the formatter.

        Parameters
        ----------
        name : string
            A string which identifies the output type of this formatter.
        header : dict, optional
            The header for the output data

        """
        self.name = name
        self._header = None
        self.print_header = True
        if header is not None:
            if 'width' in header and 'labels' in header:
                self._header = _make_header(header['labels'],
                                            header['width'],
                                            spacing=header.get('spacing', 1))
            else:
                self._header = header.get('text', None)
        else:
            self.print_header = False

    @property
    def header(self):
        """Define the header as a property."""
        return self._header

    @header.setter
    def header(self, value):
        """Set the header."""
        self._header = value

    def format(self, step, data):
        """Use the formatter to generate output.

        Parameters
        ----------
        step : integer
            This is assumed to be the current step number for
            generating the output.
        data : list, dict or similar
            This is the data we are to format. Here we assume that
            this is something we can iterate over.

        """
        out = ['{}'.format(step)]
        for i in data:
            out.append(self._FMT.format(i))
        yield ' '.join(out)

    @staticmethod
    def parse(line):
        """Parse formatted data.

        This method is intended to be the "inverse" of the :py:meth:`.format`
        method. In this particular case, we assume that we read floats from
        columns in a file. One input line corresponds to a "row" of data.

        Parameters
        ----------
        line : string
            The string we will parse.

        Returns
        -------
        out : list of floats
            The parsed input data.

        """
        return [int(col) if i == 0 else
                float(col) for i, col in enumerate(line.split())]

    def load(self, filename):
        """Read generic data from a file.

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
        data : list of tuples of int
            This is the data contained in the file. The columns are the
            step number, interface number and direction.

        See Also
        --------
        :py:func:`.read_some_lines`.

        """
        for blocks in read_some_lines(filename, self.parse):
            data_dict = {'comment': blocks['comment'],
                         'data': blocks['data']}
            yield data_dict

    def __str__(self):
        """Return basic info about the formatter."""
        return self.name


class PyretisLogFormatter(logging.Formatter):  # pragma: no cover
    """Hard-coded formatter for the PyRETIS log file.

    This formatter will just adjust multi-line messages to have some
    indentation.
    """

    def format(self, record):
        """Apply the PyRETIS log format."""
        out = logging.Formatter.format(self, record)
        if '\n' in out:
            heading, _ = out.split(record.message)
            if len(heading) < 12:
                out = out.replace('\n', '\n' + ' ' * len(heading))
            else:
                out = out.replace('\n', '\n' + ' ' * 4)
        return out


class EnergyFormatter(OutputFormatter):
    """A class for formatting energy data from PyRETIS.

    This class handles formatting of energy data.
    The data is formatted in 5 columns:

    1) Time, i.e. the step number.

    2) Potential energy.

    3) Kinetic energy.

    4) Total energy, should equal the sum of the two previous columns.

    5) Temperature.

    """

    # Format for the energy files:
    ENERGY_FMT = ['{:>10d}'] + 5*['{:>14.6f}']
    ENERGY_TERMS = ('vpot', 'ekin', 'etot', 'temp')
    HEADER = {'labels': ['Time', 'Potential', 'Kinetic', 'Total',
                         'Temperature'],
              'width': [10, 14]}

    def __init__(self, name='EnergyFormatter'):
        """Initialise the formatter for energy."""
        super().__init__(name, header=self.HEADER)

    def apply_format(self, step, energy):
        """Apply the energy format.

        Parameters
        ----------
        step : int
            The current simulation step.
        energy : dict
            A dict with energy terms to format.

        Returns
        -------
        out : string
            A string with the formatted energy data.

        """
        towrite = [self.ENERGY_FMT[0].format(step)]
        for i, key in enumerate(self.ENERGY_TERMS):
            value = energy.get(key, None)
            if value is None:
                towrite.append(self.ENERGY_FMT[i + 1].format(float('nan')))
            else:
                towrite.append(self.ENERGY_FMT[i + 1].format(float(value)))
        return ' '.join(towrite)

    def format(self, step, data):
        """Yield formatted energy data. See :py:meth:.`apply_format`."""
        yield self.apply_format(step, data)

    def load(self, filename):
        """Load entire energy blocks into memory.

        Parameters
        ----------
        filename : string
            The path/file name of the file we want to open.

        Yields
        ------
        data_dict : dict
            This is the energy data read from the file, stored in
            a dict. This is for convenience so that each energy term
            can be accessed by `data_dict['data'][key]`.

        """
        for blocks in read_some_lines(filename, line_parser=self.parse):
            data = np.array(blocks['data'])
            col = tuple(data.shape)
            col_max = min(col[1], len(self.ENERGY_TERMS) + 1)
            data_dict = {'comment': blocks['comment'],
                         'data': {'time': data[:, 0]}}
            for i in range(col_max-1):
                data_dict['data'][self.ENERGY_TERMS[i]] = data[:, i+1]
            yield data_dict


class EnergyPathFormatter(EnergyFormatter):
    """A class for formatting energy data for paths."""

    ENERGY_TERMS = ('vpot', 'ekin')
    HEADER = {'labels': ['Time', 'Potential', 'Kinetic'],
              'width': [10, 14]}

    def __init__(self):
        """Initialise."""
        super().__init__(name='EnergyPathFormatter')
        self.print_header = False

    def format(self, step, data):
        """Format the order parameter data from a path.

        Parameters
        ----------
        step : int
            The cycle number we are creating output for.
        data : tuple
            Here we assume that ``data[0]`` contains an object
            like :py:class:`.PathBase` and that ``data[1]`` is a
            string with the status for the path.

        Yields
        ------
        out : string
            The strings to be written.

        """
        path, status = data[0], data[1]
        if not path:  # when nullmoves = False
            return
        move = path.generated
        yield '# Cycle: {}, status: {}, move: {}'.format(step, status, move)
        yield self.header
        for i, phasepoint in enumerate(path.phasepoints):
            energy = {}
            for key in self.ENERGY_TERMS:
                energy[key] = getattr(phasepoint.particles, key, None)
            yield self.apply_format(i, energy)


class EnergyFile(FileIO):
    """A class for handling PyRETIS energy files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the file object and attach the energy formatter."""
        super().__init__(filename, file_mode, EnergyFormatter(), backup=backup)


class EnergyPathFile(FileIO):
    """A class for handling PyRETIS energy path files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the file object and attach the energy formatter."""
        super().__init__(filename, file_mode, EnergyPathFormatter(),
                         backup=backup)

def get_log_formatter(level):
    """Select a log format based on a given level.

    Here, it is just used to get a slightly more verbose format for
    the debug level.

    Parameters
    ----------
    level : integer
        This integer defines the log level.

    Returns
    -------
    out : object like :py:class:`logging.Formatter`
        An object that can be used as a formatter for a logger.

    """
    if level <= logging.DEBUG:
        return PyretisLogFormatter(LOG_DEBUG_FMT)
    return PyretisLogFormatter(LOG_FMT)

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

class CrossFormatter(OutputFormatter):
    """A class for formatting crossing data from flux simulations.

    This class handles formatting of crossing data. The format for
    the crossing file is three columns:

    1) The first column is the step number (an integer).

    2) The second column is the interface number (an integer). These are
       numbered from 1 (_NOT_ from 0).

    3) The direction we are moving in - `+` for the positive direction
       or `-` for the negative direction. Internally this is converted
       to an integer (`+1` or `-1`).

    """

    # Format for crossing files:
    CROSS_FMT = '{:>10d} {:>4d} {:>3s}'

    def __init__(self):
        """Initialise the crossing formatter."""
        header = {'labels': ['Step', 'Int', 'Dir'], 'width': [10, 4, 3]}
        super().__init__('CrossFormatter', header=header)

    def format(self, step, data):
        """Generate output data to be written to a file or screen.

        This will format the crossing data in a space delimited form.


        Parameters
        ----------
        step : int
            This is the current step number. It is only used here for
            debugging and can possibly be removed. However, it's useful
            to have here since this gives a common interface for all
            formatters.
        data : list of tuples
            The tuples are crossing with interfaces (if any) on the form
            `(timestep, interface, direction)` where the direction
            is '-' or '+'.

        Yields
        ------
        out : string
            The line(s) to be written.

        See Also
        --------
        :py:meth:`.check_crossing` in :py:mod:`pyretis.core.path`
        calculates the tuple `cross` which is used in this routine.

        Note
        ----
        We add 1 to the interface number here. This is for
        compatibility with the old FORTRAN code where the interfaces
        are numbered 1, 2, ... rather than 0, 1, ... .

        """
        msgtxt = 'Generating crossing data at step: {}'.format(step)
        logger.debug(msgtxt)
        for cro in data:
            if cro:
                yield self.CROSS_FMT.format(cro[0], cro[1] + 1, cro[2])

    @staticmethod
    def parse(line):
        """Parse crossing data.

        The format is described in :py:meth:`.format`, this method will
        parse this format for a _single_ row of data.

        Parameters
        ----------
        line : string
            A line to parse.

        Returns
        -------
        out[0] : integer
            The step number.
        out[1] : integer
            The interface number.
        out[2] : integer
            The direction, left (-1) or right (1).

        Note
        ----
        A '1' will be subtracted from the interface in analysis. This is
        just for backward compatibility with the old FORTRAN code.

        """
        linessplit = line.strip().split()
        step, inter = int(linessplit[0]), int(linessplit[1])
        direction = -1 if linessplit[2] == '-' else 1
        return step, inter, direction


class CrossFile(FileIO):
    """A class for handling PyRETIS crossing files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the cross file with correct formatter."""
        super().__init__(filename, file_mode, CrossFormatter(), backup=backup)


def txt_save_columns(outputfile, header, variables, backup=False):
    """Save variables to a text file using ``numpy.savetxt``.

    Note that the variables are assumed to be numpy.arrays of equal
    shape and that the output file may also be a compressed file in
    gzip format (this is selected by letting the output file name
    end with '.gz').

    Parameters
    ----------
    outputfile : string
        Name of the output file to create.
    header : string
        A string that will be written at the beginning of the file.
    variables : tuple or list of numpy.arrays
        These are the variables that will be saved to the text file.
    backup : boolean, optional
        Determines if we should backup old files or not.

    """
    if backup:
        msg = create_backup(outputfile)
        if msg:
            logger.warning(msg)
    nvar = len(variables)
    mat = np.zeros((len(variables[0]), nvar))
    for i, vari in enumerate(variables):
        try:
            mat[:, i] = vari
        except ValueError:
            msg = 'Could not align variables, skipping (writing zeros)'
            logger.warning(msg)
    np.savetxt(outputfile, mat, header=header)


def _fill_list(the_list, length, fillvalue=None):
    """Fill a list to a specified length.

    Parameters
    ----------
    the_list : list
        The list to fill.
    length : int
        The required length.
    fillvalue : optional
        The value to insert. If None is given the last item in the list
        is repeated.

    """
    if fillvalue is None:
        fillvalue = the_list[-1]
    while len(the_list) < length:
        the_list.append(fillvalue)


class TxtTableFormatter(OutputFormatter):
    """A class for generating table output.

    This class handles formatting of output data to a table-like
    format.

    Attributes
    ----------
    variables : list of strings
        These are the variables we will use in the table.
    fmt : string
        The format to apply to the columns.
    row_fmt : list of strings
        A list of strings used for formatting, used to construct `fmt`.
    title : string
        A title for the table.

    Example
    -------
    For creating a new table, a dictionary is convenient for
    grouping the settings:

    >>> tabl = {
    ...    'title': 'Energy output',
    ...    'var': ['step', 'temp', 'vpot'
    ...            'ekin', 'etot', 'press'],
    ...    'format': {'labels': ['Step', 'Temp', 'Pot',
    ...               'Kin', 'Tot', 'Press'],
    ...               'width': (10, 12),
    ...               'spacing': 2,
    ...               'row_fmt': ['{:> 10d}', '{:> 12.6g}']
    ...              }
    ...    }
    >>> table = TxtTableFormatter(tabl['var'], tabl['title'], **tabl['format'])


    """

    def __init__(self, variables, title, **kwargs):
        """Initialise the TxtTable object.

        Parameters
        ----------
        variables : list of strings
            These are the variables we will use in the table. If the
            header is not specified, then we will create one using
            these variables.
        title : string
            A title for the table.
        kwargs : dict
            Additional settings for the formatter. This may contain:

            * width : list of ints
                The (maximum) width of the columns. If the number of
                items in this list is smaller than the number of
                variables, we simply repeat the last width for the
                number of times we need.
            * labels : list of strings
                Table headers to use for the columns.
            * spacing : integer
                The separation between columns. The default value is 1.
            * row_fmt : list of strings
                The format to apply to the columns. If the number of
                items in this list is smaller than the number of
                variables, we simply repeat the last width for the
                number of times we need.

        """
        spacing = kwargs.get('spacing', 1)
        header = {'spacing': spacing,
                  'labels': kwargs.get('labels', list(variables))}
        width = kwargs.get('width', None)
        if width is None:
            header['width'] = [max(12, len(i)) for i in header['labels']]
        else:
            header['width'] = list(width)

        _fill_list(header['width'], len(header['labels']))

        super().__init__('TxtTableFormatter', header=header)
        self.title = title
        self.variables = variables
        row_fmt = kwargs.get('row_fmt', None)
        if row_fmt is None:
            self.row_fmt = []
            for wid in header['width']:
                if wid - 6 <= 0:
                    self.row_fmt.append(f'{{:> {wid}}}')
                else:
                    self.row_fmt.append(f'{{:> {wid}.{wid-6}g}}')
        else:
            self.row_fmt = row_fmt
        _fill_list(self.row_fmt, len(self.variables))
        str_white = ' ' * spacing
        self.fmt = str_white.join(self.row_fmt)

    def format(self, step, data):
        """Generate output from a dictionary using the requested variables.

        Parameters
        ----------
        step : int
            This is the current step number or a cycle number in a
            simulation.
        data : dict
            This is assumed to a dictionary containing the row items we
            want to format.

        Yields
        ------
        out : string
            A line with the formatted output.

        """
        var = []
        for i in self.variables:
            if i == 'step':
                var.append(step)
            else:
                var.append(data.get(i, float('nan')))
        txt = self.fmt.format(*var)
        yield txt

    def __str__(self):
        """Return a string with some info about the TxtTableFormatter."""
        msg = [f'TxtTableFormatter: "{self.title}"']
        msg += [f'\t* Variables: {self.variables}']
        msg += [f'\t* Format: {self.fmt}']
        return '\n'.join(msg)


class PathTableFormatter(TxtTableFormatter):
    """A special table output class for path ensembles.

    This object will return a table of text with a header and with
    formatted rows for a path ensemble. The table rows will contain
    data from the `PathEnsemble.nstats` variable. This table is just
    meant as output to the screen during a path ensemble simulation.
    """

    def __init__(self):
        """Initialise the PathTableFormatter."""
        title = 'Path Ensemble Statistics'
        var = ['step', 'ACC', 'BWI',
               'BTL', 'FTL', 'BTX', 'FTX']
        table_format = {'labels': ['Cycle', 'Accepted', 'BWI', 'BTL', 'FTL',
                                   'BTX', 'FTX'],
                        'width': (10, 12),
                        'spacing': 2,
                        'row_fmt': ['{:> 10d}', '{:> 12d}']}
        super().__init__(var, title, **table_format)

    def format(self, step, data):
        """Generate the output for the path table.

        Here we overload the :py:meth:`.TxtTableFormatter.format` method
        in order to write path ensemble statistics to (presumably)
        the screen.

        Parameters
        ----------
        step : int
            This is the current step number or a cycle number in a
            simulation.
        data : object like :py:class:`.PathEnsemble`
            This is the path ensemble we are generating output for.

        Yield
        -----
        out : string
            The formatted output.

        """
        row = {}
        for key in self.variables:
            if key == 'step':
                value = step
            else:
                value = data.nstats.get(key, 0)
            row[key] = value
        var = [row.get(i, float('nan')) for i in self.variables]
        yield self.fmt.format(*var)


class ThermoTableFormatter(TxtTableFormatter):
    """A special text table for energy output.

    This object will return a table of text with a header and with
    formatted rows for energy output. Typical use is in MD simulation
    where we want to display energies at different steps in the
    simulations.
    """

    def __init__(self):
        """Initialise the ThermoTableFormatter."""
        title = 'Energy Output'
        var = ['step', 'temp', 'vpot', 'ekin', 'etot', 'press']
        table_format = {'labels': ['Step', 'Temp', 'Pot',
                                   'Kin', 'Tot', 'Press'],
                        'width': (10, 12),
                        'spacing': 2,
                        'row_fmt': ['{:> 10d}', '{:> 12.6g}']}
        super().__init__(var, title, **table_format)


class RETISResultFormatter(TxtTableFormatter):
    """A special table output class for path ensembles in RETIS simulations.

    This object will return a table of text with a header and with
    formatted rows for a path ensemble. The table rows will contain
    data from the `PathEnsemble.nstats` variable. This table is just
    meant as output to the screen during a path ensemble simulation.
    """

    def __init__(self):
        """Initialise the PathTableFormatter."""
        title = 'Path Ensemble Statistics'
        var = ['pathensemble', 'step', 'ACC', 'BWI',
               'BTL', 'FTL', 'BTX', 'FTX']
        table_format = {
            'labels': [
                'Ensemble', 'Cycle', 'Accepted', 'BWI', 'BTL', 'FTL',
                'BTX', 'FTX'
            ],
            'width': (8, 8, 10),
            'spacing': 2,
            'row_fmt': ['{:>10}', '{:> 8d}', '{:> 10d}']
        }
        super().__init__(var, title, **table_format)
        self.print_header = False

    def format(self, step, data):
        """Generate the output for the path table.

        Here we overload the :py:meth:`.TxtTableFormatter.format` method
        in order to write path ensemble statistics to (presumably)
        the screen.

        Parameters
        ----------
        step : int
            This is the current step number or a cycle number in a
            simulation.
        data : object like :py:class:`.PathEnsemble`
            This is the path ensemble we are generating output for.

        Yield
        -----
        out : string
            The formatted output.

        """
        row = {}
        for key in self.variables:
            if key == 'step':
                value = step
            elif key == 'pathensemble':
                value = data.ensemble_name
            else:
                value = data.nstats.get(key, 0)
            row[key] = value
        # yield (f'# Results for path ensemble {data.ensemble_name} '
        #        f'at cycle {step}:')
        path = data.paths[-1]
        move = _GENERATED_SHORT.get(path['generated'][0], 'unknown').lower()
        # yield (f'# Generated path with status "{path["status"]}", '
        #        f'move "{move}" and length {path["length"]}.')
        omax = path['ordermax']
        # yield f'# Order parameter max was: {omax[0]} at index {omax[1]}.'
        omin = path['ordermin']
        # yield f'# Order parameter min was: {omin[0]} at index {omin[1]}.'
        # yield '# Path ensemble statistics:'
        # yield self.header
        var = [row.get(i, float('nan')) for i in self.variables]
        # yield self.fmt.format(*var)
        yield '\n'
class CrossFormatter(OutputFormatter):
    """A class for formatting crossing data from flux simulations.

    This class handles formatting of crossing data. The format for
    the crossing file is three columns:

    1) The first column is the step number (an integer).

    2) The second column is the interface number (an integer). These are
       numbered from 1 (_NOT_ from 0).

    3) The direction we are moving in - `+` for the positive direction
       or `-` for the negative direction. Internally this is converted
       to an integer (`+1` or `-1`).

    """

    # Format for crossing files:
    CROSS_FMT = '{:>10d} {:>4d} {:>3s}'

    def __init__(self):
        """Initialise the crossing formatter."""
        header = {'labels': ['Step', 'Int', 'Dir'], 'width': [10, 4, 3]}
        super().__init__('CrossFormatter', header=header)

    def format(self, step, data):
        """Generate output data to be written to a file or screen.

        This will format the crossing data in a space delimited form.


        Parameters
        ----------
        step : int
            This is the current step number. It is only used here for
            debugging and can possibly be removed. However, it's useful
            to have here since this gives a common interface for all
            formatters.
        data : list of tuples
            The tuples are crossing with interfaces (if any) on the form
            `(timestep, interface, direction)` where the direction
            is '-' or '+'.

        Yields
        ------
        out : string
            The line(s) to be written.

        See Also
        --------
        :py:meth:`.check_crossing` in :py:mod:`pyretis.core.path`
        calculates the tuple `cross` which is used in this routine.

        Note
        ----
        We add 1 to the interface number here. This is for
        compatibility with the old FORTRAN code where the interfaces
        are numbered 1, 2, ... rather than 0, 1, ... .

        """
        msgtxt = 'Generating crossing data at step: {}'.format(step)
        logger.debug(msgtxt)
        for cro in data:
            if cro:
                yield self.CROSS_FMT.format(cro[0], cro[1] + 1, cro[2])

    @staticmethod
    def parse(line):
        """Parse crossing data.

        The format is described in :py:meth:`.format`, this method will
        parse this format for a _single_ row of data.

        Parameters
        ----------
        line : string
            A line to parse.

        Returns
        -------
        out[0] : integer
            The step number.
        out[1] : integer
            The interface number.
        out[2] : integer
            The direction, left (-1) or right (1).

        Note
        ----
        A '1' will be subtracted from the interface in analysis. This is
        just for backward compatibility with the old FORTRAN code.

        """
        linessplit = line.strip().split()
        step, inter = int(linessplit[0]), int(linessplit[1])
        direction = -1 if linessplit[2] == '-' else 1
        return step, inter, direction

class CrossFile(FileIO):
    """A class for handling PyRETIS crossing files."""

    def __init__(self, filename, file_mode, backup=True):
        """Create the cross file with correct formatter."""
        super().__init__(filename, file_mode, CrossFormatter(), backup=backup)

class PathExtFormatter(OutputFormatter):
    """A class for formatting external trajectories.

    The external trajectories as stored as files and this path
    formatter includes the location of these files.

    Attributes
    ----------
    FMT : string
        The string to use for the formatting.

    """

    FMT = '{:>10}  {:>20s}  {:>10}  {:>5}'

    def __init__(self):
        """Initialise the PathExtFormatter formatter."""
        header = {'labels': ['Step', 'Filename', 'index', 'vel'],
                  'width': [10, 20, 10, 5], 'spacing': 2}

        super().__init__('PathExtFormatter', header=header)
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
        yield '# Cycle: {}, status: {}'.format(step, status)
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
        super().__init__(filename, file_mode, PathExtFormatter(),
                         backup=backup)

class SnapshotFormatter(OutputFormatter):
    """Generic class for formatting system snapshots.

    Attributes
    ----------
    write_vel : boolean
        If True, we will also format velocities
    fmt : string
        Format to use for position output.
    fmt_vel : string
        Format to use for position and velocity output.

    """

    data_keys = ('atomname', 'x', 'y', 'z', 'vx', 'vy', 'vz')
    _FMT_FULL = '{} {} {} {}'
    _FMT_FULL_VEL = '{} {} {} {} {} {} {}'
    _FMT = '{:5s} {:15.9f} {:15.9f} {:15.9f}'
    _FMT_VEL = '{:5s} {:15.9f} {:15.9f} {:15.9f} {:15.9f} {:15.9f} {:15.9f}'

    def __init__(self, write_vel=True, fmt=None):
        """Initialise the formatter.

        Parameters
        ----------
        write_vel : boolean, optional
            If True, the formatter will attempt to output velocities.
        fmt : string, optional
            Selects the format to use.

        """
        super().__init__('SnapshotFormatter', header=None)
        self.print_header = False
        self.write_vel = write_vel
        if fmt == 'full':
            self.fmt = self._FMT_FULL
            self.fmt_vel = self._FMT_FULL_VEL
        else:
            self.fmt = self._FMT
            self.fmt_vel = self._FMT_VEL

    def format(self, step, data):
        """Generate the snapshot output.

        Parameters
        ----------
        step : integer
            The current step number for generating the output.
        data : object like :py:class:`.System`
            The system we are generating output for.

        """
        for lines in self.format_snapshot(step, data):
            yield lines

    def _format_without_vel(self, particles):
        """Format positions of particles for output.

        Parameters
        ----------
        particles : object like :py:class:`.Particles`
            The particles for which we will format information.

        Yields
        ------
        out : string
            The formatted output, to be written.

        """
        pos = adjust_coordinate(particles.pos)
        for namei, posi in zip(particles.name, pos):
            yield self.fmt.format(namei, *posi)

    def _format_with_vel(self, particles):
        """Format positions of particles for output.

        Parameters
        ----------
        particles : object like :py:class:`.Particles`
            The particles for which we will format information.

        Yields
        ------
        out : string
            The formatted output, to be written.

        """
        pos = adjust_coordinate(particles.pos)
        vel = adjust_coordinate(particles.vel)
        for namei, posi, veli in zip(particles.name, pos, vel):
            yield self.fmt_vel.format(namei, posi[0], posi[1], posi[2],
                                      veli[0], veli[1], veli[2])

    @staticmethod
    def parse(line):
        """Not implemented - line parser for snapshots.

        For snapshots, we use a specialized reader that will read entire
        system snapshots. To avoid confusion, we just give a warning here.

        """
        logger.warning('The line parser is not implemented for the'
                       'snapshot reader.')

    def format_snapshot(self, step, system):
        """Format the given snapshot.

        Parameters
        ----------
        step : int
            The current simulation step.
        system : object like :py:class:`.System`
            The system object with the positions to format.

        Returns
        -------
        out : list of strings
            The formatted snapshot.

        """
        npart = system.particles.npart
        buff = [
            '{}'.format(npart),
            'Snapshot, step: {} box: {}'.format(
                step,
                system.box.print_length()
            ),
        ]
        if self.write_vel:
            formatter = self._format_with_vel
        else:
            formatter = self._format_without_vel
        for lines in formatter(system.particles):
            buff += [lines]
        return buff

    def load(self, filename):
        """Read snapshots from a given file.

        Parameters
        ----------
        filename : string
            The path/filename to open.

        Yields
        ------
        out : dict
            This dict contains the snapshot.

        """
        for snapshot in read_txt_snapshots(filename,
                                           data_keys=self.data_keys):
            yield snapshot


def read_txt_snapshots(filename, data_keys=None):
    """Read snapshots from a text file.

    Parameters
    ----------
    filename : string
        The file to read from.
    data_keys : tuple of strings, optional
        This tuple determines the data we are to read. It can
        be of type ``('atomname', 'x', 'y', 'z', ...)``.

    Yields
    ------
    out : dict
        A dictionary with the snapshot.

    """
    lines_to_read = 0
    snapshot = None
    if data_keys is None:
        data_keys = ('atomname', 'x', 'y', 'z', 'vx', 'vy', 'vz')
    read_header = False
    with open(filename, 'r', encoding="utf8") as fileh:
        for lines in fileh:
            if read_header:
                snapshot = {'header': lines.strip()}
                snapshot['box'] = get_box_from_header(snapshot['header'])
                read_header = False
                continue
            if lines_to_read == 0:  # new snapshot
                if snapshot is not None:
                    yield snapshot
                try:
                    lines_to_read = int(lines.strip())
                except ValueError:
                    logger.error('Error in the input file %s', filename)
                    raise
                read_header = True
                snapshot = None
            else:
                lines_to_read -= 1
                data = lines.strip().split()
                for i, (val, key) in enumerate(zip(data, data_keys)):
                    if i == 0:
                        value = val.strip()
                    else:
                        value = float(val)
                    try:
                        snapshot[key].append(value)
                    except KeyError:
                        snapshot[key] = [value]
    if snapshot is not None:
        yield snapshot


def get_box_from_header(header):
    """Get box lengths from a text header.

    Parameters
    ----------
    header : string
        Header from which we will extract the box.

    Returns
    -------
    out : numpy.array or None
        The box lengths.

    """
    low = header.lower()
    if low.find('box:') != -1:
        txt = low.split('box:')[1].strip()
        return np.array([float(i) for i in txt.split()])
    return None


def adjust_coordinate(coord):
    """Adjust the dimensionality of coordinates.

    This is a helper method for trajectory writers.

    A lot of the different formats expects us to have 3 dimensional
    data. This method just adds dummy dimensions equal to zero.

    Parameters
    ----------
    coord : numpy.array
        The real coordinates.

    Returns
    -------
    out : numpy.array
        The "zero-padded" coordinates.

    """
    if len(coord.shape) == 1:
        npart, dim = len(coord), 1
    else:
        npart, dim = coord.shape
    if dim == 3:
        # correct dimensionality, just stop here:
        return coord
    adjusted = np.zeros((npart, 3))
    try:
        for i in range(dim):
            adjusted[:, i] = coord[:, i]
    except IndexError:
        if dim == 1:
            adjusted[:, 0] = coord
    return adjusted


class SnapshotFile(FileIO):
    """A class for a collection of snapshots."""

    def __init__(self, filename, file_mode, backup=True, format_settings=None):
        """Create the snapshot file with the possible settings."""
        if format_settings is not None:
            fmt = initiate_instance(SnapshotFormatter, format_settings)
        else:
            fmt = SnapshotFormatter()
        super().__init__(filename, file_mode, fmt, backup=backup)

class PathEnsembleFormatter(OutputFormatter):
    """A class for formatting path ensemble data.

    This class will effectively define the data which we store
    for each path ensemble. The data is stored in columns with the
    format defined below and contains:

    1) The current cycle number: `'Step'`.

    2) The number of accepted paths: `'No.-acc'`.

    3) The number of shooting moves attempted: `'No.shoot'`.

    4) Starting location (with respect to interfaces): `'l'`.
       This can be `L` if a path starts on the left side, `R` if it
       starts on the right side and `*` if it did not reach the
       interface.

    5) Marker for crossing he middle interface: (`'m'`). This is
       either `M` (when the middle interface is crossed by the path)
       or `*` (if the middle interface is no crossed).

    6) End point for the path: `'r'`. The possible values here are
       similar to the values for the starting location given above.

    7) The length of the generated path: `'Length'`.

    8) The status of the path: `'Acc'`. This is one of the possible
       path statuses defined in :py:mod:`pyretis.core.path`.

    9) The type of move executed for generating the path: `'Mc'`.
       This is one of the moves defined in :py:mod:`pyretis.core.path`.

    10) The smallest order parameter in the path: `'Min-O'`.

    11) The largest order parameter in the path: `'Max-O'`.

    12) The index in the path for the smallest order parameter: `'Idx-Min'`.

    13) The index in the path for the largest order parameter: `'Idx-Max'`.

    14) The order parameter for the shooting point, immediately after
        the shooting move: `'O-shoot'`.

    15) The index in the source path for the shooting point: `'Idx-sh'`.

    16) The index in the new path for the shooting point: `'Idx-shN'`.

    17) The statistical weight of the path: `'Weight'`.

    """

    PATH_FMT = (
        '{0:>10d} {1:>10d} {2:>10d} {3:1s} {4:1s} {5:1s} {6:>7d} '
        '{7:3s} {8:2s} {9:>16.9e} {10:>16.9e} {11:>7d} {12:>7d} '
        '{13:>16.9e} {14:>7d} {15:7d} {16:>16.9e}'
    )
    HEADER = {
        'labels': ['Step', 'No.-acc', 'No.-shoot',
                   'l', 'm', 'r', 'Length', 'Acc', 'Mc',
                   'Min-O', 'Max-O', 'Idx-Min', 'Idx-Max',
                   'O-shoot', 'Idx-sh', 'Idx-shN', 'Weight'],
        'width': [10, 10, 10, 1, 1, 1, 7, 3, 2, 16, 16, 7, 7, 16, 7, 7, 7],
    }

    def __init__(self):
        """Initialise the formatter for path ensemble data."""
        super().__init__('PathEnsembleFormatter', header=self.HEADER)
        self.print_header = True

    def format(self, step, data):
        """Format path ensemble data.

        Here we generate output based on the last path from the
        given path ensemble.

        Parameters
        ----------
        step : int
            The current cycle/step number.
        data : object like :py:class:`.PathEnsemble`
            The path ensemble we will format here.

        Yields
        ------
        out : string
            The formatted line with the data from the last path.

        """
        path_ensemble = data
        path_dict = path_ensemble.paths[-1]
        interfaces = ['*' if i is None else i for i in path_dict['interface']]
        yield self.PATH_FMT.format(
            step,
            path_ensemble.nstats['ACC'],
            path_ensemble.nstats['nshoot'],
            interfaces[0],
            interfaces[1],
            interfaces[2],
            path_dict['length'],
            path_dict['status'],
            path_dict['generated'][0],
            path_dict['ordermin'][0],
            path_dict['ordermax'][0],
            path_dict['ordermin'][1],
            path_dict['ordermax'][1],
            path_dict['generated'][1],
            path_dict['generated'][2],
            path_dict['generated'][3],
            path_dict['weight'],
        )

    def parse(self, line):
        """Parse a line to a simplified representation of a path.

        Parameters
        ----------
        line : string
            The line of text to parse.

        Returns
        -------
        out : dict
            The dict with the simplified path information.

        """
        linec = line.strip()
        if linec.startswith('#'):
            # This is probably the comment
            return None
        # stip trailing comments
        linec = linec.split('#')[0]
        data = [i.strip() for i in linec.split()]
        if len(data) < 16:
            logger.warning(
                'Incorrect number of columns in path data, skipping line.'
            )
            return None
        if len(data) == 16:
            path_info = {
                'cycle': int(data[0]),
                'generated': [str(data[8]), float(data[13]),
                              int(data[14]), int(data[15])],
                'interface': (str(data[3]), str(data[4]), str(data[5])),
                'length': int(data[6]),
                'ordermax': (float(data[10]), int(data[12])),
                'ordermin': (float(data[9]), int(data[11])),
                'status': str(data[7]),
            }
            path_info['weight'] = 1.  # For backward compatibility

        else:
            path_info = {
                'cycle': int(data[0]),
                'generated': [str(data[8]), float(data[13]),
                              int(data[14]), int(data[15])],
                'interface': (str(data[3]), str(data[4]), str(data[5])),
                'length': int(data[6]),
                'ordermax': (float(data[10]), int(data[12])),
                'ordermin': (float(data[9]), int(data[11])),
                'status': str(data[7]),
                'weight': float(data[16]),
            }

        return path_info

    def load(self, filename):
        """Yield the different paths stored in the file.

        The lines are read on-the-fly, converted and yielded one-by-one.

        Parameters
        ----------
        filename : string
            The path/filename to open.

        Yields
        ------
        out : dict
            The information for the current path.

        """
        try:
            with open(filename, 'r', encoding="utf8") as fileh:
                for line in fileh:
                    path_data = self.parse(line)
                    if path_data is not None:
                        path_data['filename'] = filename
                        yield path_data
        except IOError as error:
            logger.critical('I/O error (%d): %s', error.errno, error.strerror)
        except Exception as error:  # pragma: no cover
            logger.critical('Error: %s', error)
            raise


class PathEnsembleFile(PathEnsemble, FileIO):
    """A class for handling PyRETIS path ensemble files.

    This class inherits from the :py:class:`.PathEnsemble` and
    this makes it possible to run the analysis directly on this
    file.

    """

    def __init__(self, filename, file_mode, ensemble_settings=None,
                 backup=True):
        """Set up the file-like object.

        Parameters
        ----------
        filename : string
            The file to open.
        file_mode : string
            Determines the mode for opening the file.
        ensemble_settings : dict, optional
            Ensemble specific settings.
        backup : boolean, optional
            Determines how we handle existing files when the mode
            is set to writing.

        """
        default_settings = {
            'ensemble_number': 0,
            'interfaces': [0.0, 1.0, 2.0],
            'detect': None
        }
        settings = {}
        for key, val in default_settings.items():
            try:
                settings[key] = ensemble_settings[key]
            except (TypeError, KeyError):
                settings[key] = val
                logger.warning(
                    'No "%s" ensemble setting given for "%s". Using defaults',
                    key,
                    self.__class__
                )
        PathEnsemble.__init__(self, settings['ensemble_number'],
                              settings['interfaces'])
        FileIO.__init__(self, filename, file_mode, PathEnsembleFormatter(),
                        backup=backup)

    def get_paths(self):
        """Yield paths from the file."""
        return self.load()

class PathStorage(OutputBase):
    """A class for handling storage of external trajectories.

    Attributes
    ----------
    target : string
        Determines the target for this output class. Here it will
        be a file archive (i.e. a directory based collection of
        files).
    archive_acc : string
        Basename for the archive with accepted trajectories.
    archive_rej : string
        Basename for the archive with rejected trajectories.
    archive_traj : string
        Basename for a sub-folder containing the actual files
        for a trajectory.
    formatters : dict
        This dict contains the formatters for writing path data,
        with default filenames used for them.
    out_dir_fmt : string
        A format to use for creating directories within the archive.
        This one is applied to the step number for the output.

    """

    target = 'file-archive'
    archive_acc = 'traj-acc'
    archive_rej = 'traj-rej'
    archive_traj = 'traj'
    formatters = {
        'order': {'fmt': OrderPathFormatter(), 'file': 'order.txt'},
        'energy': {'fmt': EnergyPathFormatter(), 'file': 'energy.txt'},
        'traj': {'fmt': PathExtFormatter(), 'file': 'traj.txt'},
    }
    out_dir_fmt = '{}'

    def __init__(self):
        """Set up the storage.

        Note that we here do not pass any formatters to the parent
        class. This is because this class is less flexible and
        only intended to do one thing - write path data for external
        trajectories.

        """
        super().__init__(None)

    def archive_name_from_status(self, status):
        """Return the name of the archive to use."""
        return self.archive_acc if status == 'ACC' else self.archive_rej

    def output_path_files(self, step, data, target_dir):
        """Write the output files for energy, path and order parameter.

        Parameters
        ----------
        step : integer
            The current simulation step.
        data : list
            Here, ``data[0]`` is assumed to be an object like
            :py:class:`.Path` and `data[1]`` a string containing the
            status of this path.
        target_dir : string
            The path to where we archive the files.

        Returns
        -------
        files : list of tuple of strings
            These are the files created. The tuple contains the files as
            a full path and a relative path (to the given
            target directory). The form is
            ``files[i] = (full_path[i], relative_path[i])``.
            The relative path is useful for organizing internally in
            archives.

        """
        path, status, = data[0], data[1]
        files = []
        for key, val in self.formatters.items():
            logger.debug('Storing: %s', key)
            fmt = val['fmt']
            full_path = os.path.join(target_dir, val['file'])
            relative_path = os.path.join(
                self.out_dir_fmt.format(step), val['file']
            )
            files.append((full_path, relative_path))
            with open(full_path, 'w', encoding="utf8") as output:
                for line in fmt.format(step, (path, status)):
                    output.write('{}\n'.format(line))
        return files

    def output(self, step, data):
        """Format the path data and store the path.

        Parameters
        ----------
        step : integer
            The current simulation step.
        data : list
            Here, ``data[0]`` is assumed to be an object like
            :py:class:`.Path`, ``data[1]`` a string containing the
            status of this path and ``data[2]`` the path ensemble for
            which the path was generated.

        Returns
        -------
        files : list of tuples of strings
            The files added to the archive.

        """
        path_ensemble = data
        path = path_ensemble.last_path
        home_dir = path_ensemble.directory['home_dir'] + '/trajs'
        archive = self.archive_name_from_status('ACC')
        # This is the path on form: /path/to/000/traj/11
        archive_path = os.path.join(
            home_dir,
            f'{path.path_number}',
        )

        # To organize things we create a subfolder for storing the
        # files. This is on form: /path/to/000/traj/11/traj
        traj_dir = os.path.join(archive_path, 'accepted')
        # Create the needed directories:
        make_dirs(traj_dir)
        # Write order, energy and traj files to the archive:
        files = self.output_path_files(step, (path, 'ACC'), archive_path)
        path_ensemble.last_path = path_ensemble._copy_path(path, traj_dir)

        return files

    def write(self, towrite, end='\n'):
        """We do not need the write method for this object."""
        logger.critical(
            '%s does *not* support the "write" method!',
            self.__class__.__name__
        )

    def formatter_info(self):
        """Return info about the formatters."""
        return [val['fmt'].__class__ for val in self.formatters.values()]

    def __str__(self):
        """Return basic info."""
        return '{} - archive writer.'.format(self.__class__.__name__)

def make_dirs(dirname):
    """Create directories for path simulations.

    This function will create a folder using a specified path.
    If the path already exists and if it's a directory, we will do
    nothing. If the path exists and is a file we will raise an
    `OSError` exception here.

    Parameters
    ----------
    dirname : string
        This is the directory to create.

    Returns
    -------
    out : string
        A string with some info on what this function did. Intended for
        output.

    """
    try:
        os.makedirs(dirname)
        msg = f'Created directory: "{dirname}"'
    except OSError as err:
        if err.errno != errno.EEXIST:  # pragma: no cover
            raise err
        if os.path.isfile(dirname):
            msg = f'"{dirname}" is a file. Will abort!'
            raise OSError(errno.EEXIST, msg) from err
        if os.path.isdir(dirname):
            msg = f'Directory "{dirname}" already exist.'
    return msg
