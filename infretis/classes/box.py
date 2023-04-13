from abc import ABCMeta, abstractmethod
from copy import copy as copy0
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BoxBase(metaclass=ABCMeta):
    """Class for a generic simulation box.

    This class defines a generic simulation box.

    Attributes
    ----------
    low : numpy.array
        1D array containing the lower bounds of the cell.
    high : numpy.array
        1D array containing the higher bounds of the cell.
    length : numpy.array
        1D array containing the length of the sides of the
        simulation box.
    ilength : numpy.array
        1D array containing the inverse box lengths for the
        simulation box.
    periodic : list of boolean
        If `periodic[i]` then we should apply periodic boundaries
        to dimension `i`.
    box_matrix : numpy.array
        2D matrix, representing the simulation cell.
    cell : numpy.array
        1D array representing the simulation cell (flattened
        version of the 2D box matrix).

    """

    def __init__(self, low, high, length, periodic, cell):
        """Initialise the BoxBase class."""
        self.low = low 
        self.high = high
        self.length = length
        self.periodic = periodic
        self.cell = cell
        # Create box matrix from the given cell:
        self.box_matrix = array_to_box_matrix(self.cell)
        self.ilength = 1.0 / self.length
        self.dim = len(self.length)

    def update_size(self, new_size):
        """Update the box size.

        Parameters
        ----------
        new_size : list, tuple, numpy.array, or other iterable
            The new box size.

        """
        if new_size is None:
            logger.warning(
                'Box update ignored: Tried to update with empty size!'
            )
        else:
            try:
                size = new_size.size
            except AttributeError:
                size = len(new_size)
            if size <= 3:
                if size == len(self.cell):
                    for i in range(self.dim):
                        self.length[i] = new_size[i]
                        self.high[i] = self.low[i] + new_size[i]
                        self.cell[i] = new_size[i]
                    self.ilength = 1.0 / self.length
            else:
                try:
                    self.box_matrix = array_to_box_matrix(new_size)
                    self.cell = [i for i in new_size]
                    self.length = np.array([float(i) for i in self.cell[:3]])
                    self.high = self.low + self.length
                    self.ilength = 1.0 / self.length
                except ValueError:
                    logger.critical('Box update failed!')

    def bounds(self):
        """Return the boundaries of the box (low, high) as an array."""
        return [(i, j) for i, j in zip(self.low, self.high)]

    @abstractmethod
    def calculate_volume(self):
        """Return the volume of the box."""
        return

    @abstractmethod
    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only.

        For the given positions, this function will apply periodic
        boundary conditions to one dimension only. This can be useful
        for instance in connection with order parameters.

        Parameters
        ----------
        pos : float
            Coordinate to wrap.
        dim : int
            This selects the dimension to consider.

        """
        return

    @abstractmethod
    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position.

        Parameters
        ----------
        pos : nump.array
            Positions to apply periodic boundaries to.

        Returns
        -------
        out : numpy.array, same shape as parameter `pos`
            The periodic-boundary wrapped positions.

        """
        return
    @abstractmethod
    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector.

        Parameters
        ----------
        distance : numpy.array
            The distance vectors.

        Returns
        -------
        out : numpy.array, same shape as parameter `distance`
            The periodic-boundary-wrapped distances.

        """
        return

    @abstractmethod
    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance.

        This will apply periodic boundaries to a distance. Note that the
        distance can be a vector, but not a matrix of distance vectors.

        Parameters
        ----------
        distance : numpy.array with shape `(self.dim,)`
            A distance vector.

        Returns
        -------
        out : numpy.array, same shape as parameter `distance`
            The periodic-boundary wrapped distance vector.

        """
        return

    def print_length(self, fmt=None):
        """Return a string with box lengths. Can be used for output."""
        if fmt is None:
            return ' '.join(('{}'.format(i) for i in self.cell))
        return ' '.join((fmt.format(i) for i in self.cell))

    def restart_info(self):
        """Return a dictionary with restart information."""
        info = {
            'length': self.length,
            'periodic': self.periodic,
            'low': self.low,
            'high': self.high,
            'cell': self.cell,
        }
        return info
    def copy(self):
        """Return a copy of the box.

        Returns
        -------
        out : object like :py:class:`.BoxBase`
            A copy of the box.

        """
        box_copy = self.__class__(
            np.copy(self.low),
            np.copy(self.high),
            np.copy(self.length),
            copy0(self.periodic),
            np.copy(self.cell)
        )
        return box_copy

    def load_restart_info(self, info):
        """Read the restart information."""
        self.length = info.get('length')
        self.periodic = info.get('periodic')
        self.low = info.get('low')
        self.high = info.get('high')
        self.cell = info.get('cell')

    def __str__(self):
        """Return a string describing the box.

        Returns
        -------
        out : string
            String with the type of box, the extent of the box and
            information about the periodicity.

        """
        boxstr = []
        if len(self.cell) <= 3:
            boxstr.append('Orthogonal box:')
        else:
            boxstr.append('Triclinic box:')
        for i, periodic in enumerate(self.periodic):
            low = self.low[i]
            high = self.high[i]
            msg = 'Dim: {}, Low: {}, high: {}, periodic: {}'
            boxstr.append(msg.format(i, low, high, periodic))
        cell = self.print_length()
        boxstr.append('Cell: {}'.format(cell))
        return '\n'.join(boxstr)

    def __eq__(self, other):
        """Compare two box objects."""
        attrs = {'low', 'high', 'length', 'ilength', 'box_matrix', 'cell',
                 'periodic', 'dim'}
        numpy_attrs = {'low', 'high', 'length', 'ilength', 'box_matrix',
                       'cell'}
        return compare_objects(self, other, attrs, numpy_attrs)
    def __ne__(self, other):
        """Compare two box objects."""
        return not self == other

class RectangularBox(BoxBase):
    """An orthogonal box."""

    def calculate_volume(self):
        """Calculate the volume of the box.

        Returns
        -------
        out : float
            The volume of the box.

        """
        return product(self.length)

    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only.

        For the given positions, this function will apply periodic
        boundary conditions to one dimension only. This can be useful
        for instance in connection with order parameters.

        Parameters
        ----------
        pos : float
            Coordinate to wrap around.
        dim : int
            This selects the dimension to consider.

        """
        if self.periodic[dim]:
            low, length = self.low[dim], self.length[dim]
            ilength = self.ilength[dim]
            relpos = pos - low
            delta = relpos
            if relpos < 0.0 or relpos >= length:
                delta = relpos - np.floor(relpos * ilength) * length
            return delta + low
        return pos

    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position.

        Parameters
        ----------
        pos : nump.array
            Positions to apply periodic boundaries to.

        Returns
        -------
        out : numpy.array, same shape as parameter `pos`
            The periodic-boundary wrapped positions.

        """
        pbcpos = np.zeros(pos.shape)
        for i, periodic in enumerate(self.periodic):
            if periodic:
                low = self.low[i]
                length = self.length[i]
                ilength = self.ilength[i]
                relpos = pos[:, i] - low
                delta = np.where(
                    np.logical_or(relpos < 0.0, relpos >= length),
                    relpos - np.floor(relpos * ilength) * length,
                    relpos
                    )
                pbcpos[:, i] = delta + low
            else:
                pbcpos[:, i] = pos[:, i]
        return pbcpos

    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector.

        Parameters
        ----------
        distance : numpy.array
            The distance vectors.

        Returns
        -------
        out : numpy.array, same shape as the `distance` parameter
            The pbc-wrapped distances.

        Note
        ----
        This will modify the given input matrix inplace. This can be
        changed by setting ``pbcdist = np.copy(distance)``.

        """
        pbcdist = distance
        for i, (periodic, length, ilength) in enumerate(zip(self.periodic,
                                                            self.length,
                                                            self.ilength)):
            if periodic:
                dist = pbcdist[:, i]
                high = 0.5 * length
                k = np.where(np.abs(dist) >= high)[0]
                dist[k] -= np.rint(dist[k] * ilength) * length
        return pbcdist

    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance.

        This will apply periodic boundaries to a distance. Note that the
        distance can be a vector, but not a matrix of distance vectors.

        Parameters
        ----------
        distance : numpy.array with shape `(self.dim,)`
            A distance vector.

        Returns
        -------
        out : numpy.array, same shape as the `distance` parameter
            The periodic-boundary wrapped distance vector.

        """
        pbcdist = np.zeros(distance.shape)
        for i, (periodic, length, ilength) in enumerate(zip(self.periodic,
                                                            self.length,
                                                            self.ilength)):
            if periodic and np.abs(distance[i]) > 0.5*length:
                pbcdist[i] = (distance[i] -
                              np.rint(distance[i] * ilength) * length)
            else:
                pbcdist[i] = distance[i]
        return pbcdist

class TriclinicBox(BoxBase):
    """This class represents a triclinic box."""

    def calculate_volume(self):
        """Calculate and return the volume of the box.

        Returns
        -------
        out : float
            The volume of the box.

        """
        return det(self.box_matrix)

    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only."""
        raise NotImplementedError

    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position."""
        raise NotImplementedError

    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector."""
        raise NotImplementedError

    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance."""
        raise NotImplementedError

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


def list_get(input_list, index):
    """Get an item from a list and handle out-of bounds errors.

    This method is intended to be used when we are picking items from
    a list and possibly we want a number of items which is larger than
    the number of items in the list. Here, we then just return the last
    element.

    Parameters
    ----------
    input_list : list
        The list to pick from.
    index : integer
        The index to pick.

    """
    try:
        return input_list[index]
    except IndexError:
        return input_list[-1]


def array_to_box_matrix(cell):
    """Return a box matrix corresponding to a cell array.

    Parameters
    ----------
    cell : list or numpy.array
        An (1D) array containing 1, 2, 3, 6 or 9 items. These are
        the xx, yy, zz, xy, xz, yx, yz, zx, zy elements. Setting
        x = 0, y = 1 and z = 2 will give the indices in the matrix,
        e.g. yx -> (1, 0) will correspond to the item in row 1 and
        column 0.

    Returns
    -------
    box : numpy.array (2D)
        The box vector on matrix form.

    """
    if len(cell) == 1:
        return 1.0 * np.array([cell[0]])
    if len(cell) == 2:
        return 1.0 * np.array([[cell[0], 0.0],
                               [0.0, cell[1]]])
    if len(cell) == 3:
        return 1.0 * np.array([[cell[0], 0.0, 0.0],
                               [0.0, cell[1], 0.0],
                               [0.0, 0.0, cell[2]]])
    if len(cell) == 6:
        return 1.0 * np.array([[cell[0], cell[3], cell[4]],
                               [0.0, cell[1], cell[5]],
                               [0.0, 0.0, cell[2]]])
    if len(cell) == 9:
        return 1.0 * np.array([[cell[0], cell[3], cell[4]],
                               [cell[5], cell[1], cell[6]],
                               [cell[7], cell[8], cell[2]]])
    logger.error(
        '%d box parameters given, need 1, 2, 3, 6, or 9.', len(cell)
    )
    raise ValueError('Incorrect number of box-parameters!')

def check_consistency(low, high, length):
    """Check that given box bounds are consistent.

    Parameters
    ----------
    low : numpy.array
        The lower bounds for the box.
    high : numpy.array
        The upper bounds for the box.
    length : numpy.array
        The lengths of the box.

    """
    length_re = high - low
    if any(i <= 0 for i in length_re):
        logger.error('Check box settings. Found high <= low.')
        raise ValueError('Incorrect box: high <= low.')
    if not all(np.isclose(length, length_re)):
        logger.error('Check box settings: length != high - low.')
        raise ValueError('Incorrect box: length != high - low.')

def _get_low_high_length(low, high, length, periodic):
    """Determine box cell parameters from input.

    This method will consider the following cases:

    1) We are given low, high and length.
    2) We are given high and length, and determine the low values.
    3) We are given low and length, and determine the high values.
    4) We are given length, assume low to be zero and determine high.
    5) We are given low and high, and determine the length.
    6) We are given just high, assume low to be zero and
       determine the length.
    7) We are just given low, and assume high and length to be infinite.
    8) We are given none of the values, and assume all to infinite.

    Parameters
    ----------
    low : numpy.array or None
        The lower bounds for the box.
    high : numpy.array or None
        The upper bounds for the box.
    length : numpy.array or None
        The lengths of the box.
    periodic : list of boolean or None
        We will assume a periodic box for dimensions where
        this list is True.

    Returns
    -------
    out[0] : numpy.array
        The updated lower bounds for the box.
    out[1] : numpy.array
        The updated upper bounds for the box.
    out[2] : numpy.array
        The updated lengths of the box.
    out[3] : list of boolean
        The updated periodic settings for the box.

    """
    case = (length is not None, low is not None, high is not None)
    if low is not None:
        low = np.array(low)
    if high is not None:
        high = np.array(high)
    if length is not None:
        length = np.array(length)
    if case == (True, True, True):
        # 1) We have given length, low and high.
        pass
    elif case == (True, False, True):
        # 2) Length & high has been given, just determine low:
        low = high - length
    elif case == (True, True, False):
        # 3) Length and low was given, determine high:
        high = low + length
    elif case == (True, False, False):
        # 4) Length is given, set low to 0 and high to low + length:
        low = np.zeros_like(length)
        high = low + length
    elif case == (False, True, True):
        # 5) Low and high is given, determine length:
        length = high - low
    elif case == (False, False, True):
        # 6) High is given, assume low and determine length:
        low = np.zeros_like(high)
        length = high - low
    elif case == (False, True, False):
        # 7) Low given. High and length to be determined.
        # This is not enough info, so we assume an infinite box:
        length = float('inf') * np.ones_like(low)
        high = float('inf') * np.ones_like(low)
    elif case == (False, False, False):
        # Not much info is given. We let the box be similar
        # in shape to the input periodic settings:
        if periodic is None:
            logger.info(
                'Too few settings for the box is given. A 1D box is assumed.'
            )
            periodic = [False]
        low = np.array([-float('inf') for _ in periodic])
        high = float('inf') * np.ones_like(low)
        length = float('inf') * np.ones_like(low)
    return low, high, length, periodic

def create_box(low=None, high=None, cell=None, periodic=None):
    """Set up and create a box.

    Parameters
    ----------
    low : numpy.array, optional
        1D array containing the lower bounds of the cell.
    high : numpy.array, optional
        1D array containing the higher bounds of the cell.
    cell : numpy.array, optional
        1D array, a flattened version of the simulation box matrix.
        This array is expected to contain 1, 2, 3, 6 or 9 items.
        These are the xx, yy, zz, xy, xz, yx, yz, zx, zy elements,
        respectively.
    periodic : list of boolean, optional
        If `periodic[i]` then we should apply periodic boundaries
        to dimension `i`.

    Returns
    -------
    out : object like :py:class:`.BoxBase`
        The object representing the simulation box.

    """
    # If the cell is given, the we should be able to determine
    # the length:
    if cell is not None:
        # make sure the cell does not become an array of objects
        cell = np.array(cell, dtype=float)

        if len(cell) <= 3:
            length = np.array([i for i in cell])
        else:  # Use the xx, yy and zz parameters:
            length = np.array([i for i in cell[:3]])
        # Determine low, high, length and possible periodic:
        low, high, length, periodic = _get_low_high_length(
            low, high, length, periodic
        )
    else:  # Here, cell was not given. Try to obtain it from the length:
        low, high, length, periodic = _get_low_high_length(
            low, high, cell, periodic
        )
        cell = np.array([i for i in length])
    # We can still have periodic not set:
    if periodic is None:
        logger.info(
            'Periodic settings not given. Assumed True for all directions.'
        )
        periodic = [True] * len(length)
    else:
        # If for some reason the periodic settings have wrong length:
        if len(periodic) < len(length):
            logger.info('Setting missing periodic settings to True.')
            for _ in range(len(length) - len(periodic)):
                periodic.append(True)
        elif len(periodic) > len(length):
            logger.error('Too many periodic settings given.')
            raise ValueError('Too many periodic settings given.')
    # Here, everything should be set:
    check_consistency(low, high, length)
    # Create the box:
    obj = TriclinicBox
    if len(cell) <= 3:
        obj = RectangularBox
    return obj(low, high, length, periodic, cell)


def set_up_box(settings, boxs, dim=3):
    """Set up a box from given settings.

    Parameters
    ----------
    settings : dict
        The dict with the simulation settings.
    boxs : dict or None
        If no box settings are given, we can still create a box,
        inferred from the positions of the particles. This dict
        contains the settings to do so.
    dim : integer, optional
        Number of dimensions for the box. This is used only as a last
        resort when no information about the box is given.

    Returns
    -------
    box : object like :py:class:`.BoxBase` or None
        The box if we managed to create it, otherwise None.

    """
    msg = 'Box created {}:\n{}'
    box = None
    if settings.get('box', None) is not None:
        box = create_box(**settings['box'])
        msgtxt = msg.format('from settings', box)
        logger.info(msgtxt)
        debugtxt = 'Settings used:\n{}'.format(settings['box'])
        logger.debug(debugtxt)
    else:
        if boxs is not None:
            box = create_box(**boxs)
            msgtxt = msg.format('from initial positions', box)
            logger.info(msgtxt)
            msgwarn = 'The box was assumed periodic in all directions.'
            logger.warning(msgwarn)
        else:
            if dim > 0:
                box = create_box(periodic=[False]*dim)
                msgtxt = msg.format('without specifications', box)
                logger.info(msgtxt)
                msgwarn = 'The box was assumed nonperiodic in all directions.'
                logger.warning(msgwarn)
    return box

def box_from_restart(restart):
    """Create a box from restart settings.

    Parameters
    ----------
    restart : dict
        A dictionary with restart settings.

    Returns
    -------
    box : object like :py:class:`.BoxBase`
        The box created from the restart settings.

    """
    restart_box = restart.get('box', None)
    if restart_box is None:
        logger.info('No box created from restart settings.')
        return None
    box = create_box(
        low=restart_box.get('low'),
        high=restart_box.get('high'),
        cell=restart_box.get('cell'),
        periodic=restart_box.get('periodic')
    )
    return box

def box_matrix_to_list(matrix, full=False):
    """Return a list representation of the box matrix.

    This method ensures correct ordering of the elements for PyRETIS:
    ``xx, yy, zz, xy, xz, yx, yz, zx, zy``.

    Parameters
    ----------
    matrix : numpy.array
        A matrix (2D) representing the box.
    full : boolean, optional
        Return a full set of parameters (9) if set to True. If False,
        and we need 3 or fewer parameters (i.e. the other 6 are zero)
        we will only return the 3 non-zero ones.

    Returns
    -------
    out : list
        A list with the box-parametres.

    """
    if matrix is None:
        return None
    if np.count_nonzero(matrix) <= 3 and not full:
        return [matrix[0, 0], matrix[1, 1], matrix[2, 2]]
    return [matrix[0, 0], matrix[1, 1], matrix[2, 2],
            matrix[0, 1], matrix[0, 2], matrix[1, 0],
            matrix[1, 2], matrix[2, 0], matrix[2, 1]]
