from abc import ABCMeta, abstractmethod
import numpy as np
import os

def _read_line_data(ncol, stripline, line_parser):
    """Read data for :py:func:`.read_some_lines.`.

    Parameters
    ----------
    ncol : integer
        The expected number of columns to read. If this is less than 1
        it is not yet set. Note that we skip data which appear
        inconsistent. A warning will be issued about this.
    stripline : string
        The line to read. Note that we assume that leading and
        trailing spaces have been removed.
    line_parser : callable
        A method we use to parse a single line.

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

def read_some_lines(filename, line_parser, block_label='#'):                   
    """Open a file and try to read as many lines as possible.                  
                                                                               
    This method will read a file using the given `line_parser`.                
    If the given `line_parser` fails at a line in the file,                    
    `read_some_lines` will stop here. Further, this method                     
    will read data in blocks and yield a block when a new                      
    block is found. A special string (`block_label`) is assumed to             
    identify the start of blocks.                                              
                                                                               
    Parameters                                                                 
    ----------                                                                 
    filename : string                                                          
        This is the name/path of the file to open and read.                    
    line_parser : function, optional                                           
        This is a function which knows how to translate a given line           
        to a desired internal format. If not given, a simple float             
        will be used.                                                          
    block_label : string, optional                                             
        This string is used to identify blocks.                                
                                                                               
    Yields                                                                     
    ------                                                                     
    data : list                                                                
        The data read from the file, arranged in dicts.                        
                                                                               
    """                                                                        
    ncol = -1  # The number of columns                                         
    new_block = {'comment': [], 'data': []}                                    
    yield_block = False                                                        
    read_comment = False                                                       
    with open(filename, 'r', encoding='utf-8') as fileh:                       
        for i, line in enumerate(fileh):                                       
            stripline = line.strip()                                           
            if stripline.startswith(block_label):                              
                # this is a comment, then a new block will follow,             
                # unless this is a multi-line comment.                         
                if read_comment:  # part of multi-line comment...              
                    new_block['comment'].append(stripline)                     
                else:                                                          
                    if yield_block:                                            
                        # Yield the current block                              
                        yield_block = False                                    
                        yield new_block                                        
                    new_block = {'comment': [stripline], 'data': []}           
                    yield_block = True  # Data has been added                  
                    ncol = -1                                                  
                    read_comment = True                                        
            else:                                                              
                read_comment = False                                           
                data, _yieldb, _ncol = _read_line_data(ncol, stripline,        
                                                       line_parser)            
                if data:                                                       
                    new_block['data'].append(data)                             
                    ncol = _ncol                                               
                    yield_block = _yieldb                                      
                else:                                                          
                    logger.warning('Skipped malformed data in "%s", line: %i', 
                                   filename, i)                                
    # if the block has not been yielded, yield it                              
    if yield_block:                                                            
        yield_block = False                                                    
        yield new_block                                                        

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

class OutputBase(metaclass=ABCMeta):
    """A generic class for handling output.

    Attributes
    ----------
    formatter : object like py:class:`.OutputFormatter`
        The object responsible for formatting output.
    target : string
        Determines where the target for the output, for
        instance "screen" or "file".
    first_write : boolean
        Determines if we have written something yet, or
        if this is the first write.

    """

    target = None

    def __init__(self, formatter):
        """Create the object and attach a formatter."""
        self.formatter = formatter
        self.first_write = True

    def output(self, step, data):
        """Use the formatter to write data to the file.

        Parameters
        ----------
        step : int
            The current step number.
        data : list
            The data we are going to output.

        """
        if self.first_write and self.formatter.print_header:
            self.first_write = False
            self.write(self.formatter.header)
        for line in self.formatter.format(step, data):
            self.write(line)

    @abstractmethod
    def write(self, towrite, end='\n'):
        """Write a string to the output defined by this class.

        Parameters
        ----------
        towrite : string
            The string to write.
        end : string, optional
            A "terminator" for the given string.

        Returns
        -------
        status : boolean
            True if we managed to write, False otherwise.

        """
        return

    def formatter_info(self):
        """Return a string with info about the formatter."""
        if self.formatter is not None:
            return self.formatter.__class__
        return None

    def __str__(self):
        """Return basic info."""
        return f'{self.__class__.__name__}\n\t* Formatter: {self.formatter}'

class FileIO(OutputBase):
    """A generic class for handling IO with files.

    This class defines how PyRETIS stores and reads data.
    Formatting is handled by an object like :py:class:`.OutputFormatter`

    Attributes
    ----------
    filename : string
        Name (e.g. path) to the file to read or write.
    file_mode : string
        Specifies the mode in which the file is opened.
    backup : boolean
        Determines the behavior if we want to write to a file
        that is already existing.
    fileh : object like :py:class:`io.IOBase`
        The file handle we are interacting with.
    last_flush : object like :py:class:`datetime.datetime`
        The previous time for flushing to the file.
    FILE_FLUSH : integer
        The interval for flushing to the file. That is, we will
        flush if the time since the last flush is larger than this
        value. Note that this is only checked in relation to writing.

    """

    target = 'file'
    FILE_FLUSH = 1  # Interval for flushing files in seconds.

    def __init__(self, filename, file_mode, formatter, backup=True):
        """Set up the file object.

        Parameters
        ----------
        filename : string
            The path to the file to open or read.
        file_mode : string
            Specifies the mode for opening the file.
        formatter : object like py:class:`.OutputFormatter`
            The object responsible for formatting output.
        backup : boolean, optional
            Defines how we handle cases where we write to a
            file which is already existing.

        """
        super().__init__(formatter)
        self.filename = filename
        self.file_mode = file_mode
        if backup not in (True, False):
            logger.info('Setting backup to default: True')
            self.backup = True
        else:
            self.backup = backup
        self.fileh = None
        if self.file_mode.startswith('a') and self.formatter is not None:
            self.formatter.print_header = False
        self.last_flush = None

    def open_file_read(self):
        """Open a file for reading."""
        if not self.file_mode.startswith('r'):
            raise ValueError(
                ('Inconsistent file mode "{}" '
                 'for reading').format(self.file_mode)
            )
        try:
            self.fileh = open(self.filename, self.file_mode)
        except (OSError, IOError) as error:
            logger.critical(
                'Could not open file "%s" for reading', self.filename
            )
            logger.critical(
                'I/O error ({%d}): {%s}', error.errno, error.strerror
            )
        return self.fileh

    def open_file_write(self):
        """Open a file for writing.

        In this method, we also handle the possible backup settings.
        """
        if not self.file_mode[0] in ('a', 'w'):
            raise ValueError(
                ('Inconsistent file mode "{}" '
                 'for writing').format(self.file_mode)
            )
        msg = []
        try:
            if os.path.isfile(self.filename):
                msg = ''
                if self.file_mode.startswith('a'):
                    logger.info(
                        'Appending to existing file "%s"', self.filename
                    )
                else:
                    if self.backup:
                        msg = create_backup(self.filename)
                        logger.debug(msg)
                    else:
                        logger.debug(
                            'Overwriting existing file "%s"', self.filename
                        )
            self.fileh = open(self.filename, self.file_mode)
        except (OSError, IOError) as error:  # pragma: no cover
            logger.critical(
                'Could not open file "%s" for writing', self.filename
            )
            logger.critical(
                'I/O error (%d): %d', error.errno, error.strerror
            )
        return self.fileh

    def open(self):
        """Open a file for reading or writing."""
        if self.fileh is not None:
            logger.debug(
                '%s asked to open file, but it has already opened a file.',
                self.__class__.__name__
            )
            return self.fileh
        if self.file_mode[0] in ('r',):
            return self.open_file_read()
        if self.file_mode[0] in ('a', 'w'):
            return self.open_file_write()
        raise ValueError('Unknown file mode "{}"'.format(self.file_mode))

    def load(self):
        """Read blocks or lines from the file."""
        return self.formatter.load(self.filename)

    def write(self, towrite, end='\n'):
        """Write a string to the file.

        Parameters
        ----------
        towrite : string
            The string to output to the file.
        end : string, optional
            Appended to `towrite` when writing, can be used to print a
            new line after the input `towrite`.

        Returns
        -------
        status : boolean
            True if we managed to write, False otherwise.

        """
        status = False
        if towrite is None:
            return status
        if self.fileh is not None and not self.fileh.closed:
            try:
                if end is not None:
                    self.fileh.write('{}{}'.format(towrite, end))
                    status = True
                else:
                    self.fileh.write(towrite)
                    status = True
            except (OSError, IOError) as error:  # pragma: no cover
                msg = 'Write I/O error ({}): {}'.format(error.errno,
                                                        error.strerror)
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
            logger.warning('Ignored writing to closed file %s', self.filename)
        if self.fileh is None:
            logger.critical(
                'Attempting to write to empty file handle for file %s',
                self.filename
            )
        return status

    def close(self):
        """Close the file."""
        if self.fileh is not None and not self.fileh.closed:
            try:
                self.flush()
            finally:
                self.fileh.close()

    def flush(self):
        """Flush file buffers to file."""
        if self.fileh is not None and not self.fileh.closed:
            self.fileh.flush()
            os.fsync(self.fileh.fileno())

    def output(self, step, data):
        """Open file before first write."""
        if self.first_write:
            self.open()
        return super().output(step, data)

    def __del__(self):
        """Close the file in case the object is deleted."""
        self.close()

    def __enter__(self):
        """Context manager for opening the file."""
        self.open()
        return self

    def __exit__(self, *args):
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

    def __str__(self):
        """Return basic info."""
        msg = ['FileIO (file: "{}")'.format(self.filename)]
        if self.fileh is not None and not self.fileh.closed:
            msg += ['\t* File is open']
            msg += ['\t* Mode: {}'.format(self.fileh.mode)]
        msg += ['\t* Formatter: {}'.format(self.formatter)]
        return '\n'.join(msg)

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
