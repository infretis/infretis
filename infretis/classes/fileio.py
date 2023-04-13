from abc import ABCMeta, abstractmethod
from datetime import datetime
import os


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

