from infretis.core.common import read_restart_file
from infretis.newc.system import System
from infretis.newc.formats.path import PathExtFile
from infretis.newc.formats.order import OrderPathFile
from infretis.newc.formats.energy import EnergyPathFile
from abc import ABCMeta, abstractmethod
import pickle
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Path:
    def __init__(self, maxlen=None, time_origin=0):
        self.maxlen = maxlen
        self.status = None
        self.generated = None
        self.path_number = None
        self.weights = None
        self.phasepoints = []
        self.min_valid = None
        self.time_origin = time_origin

    @property
    def length(self):
        """Compute the length of the path."""
        return len(self.phasepoints)

    @property
    def ordermin(self):
        """Compute the minimum order parameter of the path."""
        idx = np.argmin([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def ordermax(self):
        """Compute the maximum order parameter of the path."""
        idx = np.argmax([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def adress(self):
        """Compute the maximum order parameter of the path."""
        adresses = set([i.config[0] for i in self.phasepoints])
        return adresses

    def check_interfaces(self, interfaces):
        if self.length < 1:
            logger.warning('Path is empty!')
            return None, None, None, None
        ordermax, ordermin = self.ordermax[0], self.ordermin[0]
        cross = [ordermin < interpos <= ordermax for interpos in interfaces]
        left, right = min(interfaces), max(interfaces)
        # Check end & start point:
        end = self.get_end_point(left, right)
        start = self.get_start_point(left, right)
        middle = 'M' if cross[1] else '*'
        return start, end, middle, cross

    def get_end_point(self, left, right=None):
        """Return the end point of the path as a string.

        The end point is either to the left of the `left` interface or
        to the right of the `right` interface, or somewhere in between.

        Parameters
        ----------
        left : float
            The left interface.
        right : float, optional
            The right interface, equal to left if not specified.

        Returns
        -------
        out : string
            A string representing where the end point is ('L' - left,
            'R' - right or None).

        """
        if right is None:
            right = left
        assert left <= right

        if self.phasepoints[-1].order[0] <= left:
            end = 'L'
        elif self.phasepoints[-1].order[0] >= right:
            end = 'R'
        else:
            end = None
            logger.debug('Undefined end point.')
        return end

    def get_start_point(self, left, right=None):
        """Return the start point of the path as a string.

        The start point is either to the left of the `left` interface or
        to the right of the `right` interface.

        Parameters
        ----------
        left : float
            The left interface.
        right : float, optional
            The right interface, equal to left if not specified.

        Returns
        -------
        out : string
            A string representing where the start point is ('L' - left,
            'R' - right or None).

        """
        if right is None:
            right = left
        assert left <= right
        if self.phasepoints[0].order[0] <= left:
            start = 'L'
        elif self.phasepoints[0].order[0] >= right:
            start = 'R'
        else:
            start = None
            logger.debug('Undefined starting point.')
        return start

    # def get_shooting_point(self):
    #     idx = self.rgen.random_integers(1, self.length - 2)
    #     logger.debug("Selected point with orderp %s",
    #                  self.phasepoints[idx].order[0])
    #     return self.phasepoints[idx], idx

    def append(self, phasepoint):
        """Append a new phase point to the path.

        Parameters
        ----------
        out : object like :py:class:`.System`
            The system information we add to the path.

        """
        if self.maxlen is None or self.length < self.maxlen: ###maxlen adjust
            self.phasepoints.append(phasepoint)
            return True
        logger.debug('Max length exceeded. Could not append to path.')
        return False

    def get_path_data(self, status, interfaces):
        """Return information about the path.

        This information can be stored in a object like
        :py:class:`.PathEnsemble`.

        Parameters
        ----------
        status : string
            This represents the current status of the path.
        interfaces : list
            These are just the interfaces we are currently considering.

        """
        path_info = {
            'generated': self.generated,
            'status': status,
            'length': self.length,
            'ordermax': self.ordermax,
            'ordermin': self.ordermin,
            'weight': self.weight,
        }

        start, end, middle, _ = self.check_interfaces(interfaces)
        path_info['interface'] = (start, middle, end)

        return path_info

    def get_move(self):
        """Return the move used to generate the path."""
        if self.generated is None:
            return None
        return self.generated[0]

    def success(self, target_interface):
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

    def __iadd__(self, other):
        """Add path data to a path from another path, i.e. ``self += other``.

        This will simply append the phase points from `other`.

        Parameters
        ----------
        other : object of type `Path`
            The object to add path data from.

        Returns
        -------
        self : object of type `Path`
            The updated path object.

        """
        for phasepoint in other.phasepoints:
            app = self.append(phasepoint.copy())
            if not app:
                logger.warning(
                    'Truncated path at %d while adding paths', self.length
                )
                return self
        return self

    def copy(self):
        """Return a copy of the path."""
        new_path = self.empty_path()
        for phasepoint in self.phasepoints:
            new_path.append(phasepoint.copy())
        new_path.status = self.status
        new_path.time_origin = self.time_origin
        new_path.generated = self.generated
        new_path.maxlen = self.maxlen
        new_path.path_number = self.path_number
        new_path.weights= self.weights
        return new_path

    def reverse_velocities(self):
        """Reverse the velocities in the system."""
        self.vel_rev = not self.vel_rev

    def reverse(self, order_function=False, rev_v=True):
        """Reverse a path and return the reverse path as a new path.

        This will reverse a path and return the reversed path as
        a new object like :py:class:`.PathBase` object.

        Returns
        -------
        new_path : object like :py:class:`.PathBase`
            The time reversed path.
        order_function : object like :py:class:`.OrderParameter`, optional
            The method to use to re-calculate the order parameter,
            if it is velocity dependent.
        rev_v : boolean, optional
            If True, also the velocities are reversed, if False, the velocities
            for each frame are not altered.

        """
        new_path = self.empty_path()
        new_path.weight = self.weight
        new_path.maxlen = self.maxlen
        for phasepoint in reversed(self.phasepoints):
            new_point = phasepoint.copy()
            if rev_v:
                self.reverse_velocities(new_point)
            app = new_path.append(new_point)
            if not app:  # pragma: no cover
                msg = 'Could not reverse path'
                logger.error(msg)
                return None
        if order_function and order_function.velocity_dependent and rev_v:
            for phasepoint in new_path.phasepoints:
                phasepoint.order = order_function.calculate(phasepoint)
        return new_path

    def restart_info(self):
        """Return a dictionary with restart information."""
        info = {
            'generated': self.generated,
            'time_origin': self.time_origin,
            'status': self.status,
            'weights': self.weights,
            'min_valid': self.min_valid,
            'path_number': self.path_number,
            'phasepoints': self.phasepoints # [i.restart_info() for i in self.phasepoints]
        }
        return info

    def load_restart_info(self, info):
        """Set up the path using restart information."""
        for key, val in info.items():
            # For phasepoints, create new System objects
            # and load the information for these.
            # The snaps still need to forcefield to be re-initiated.
            if key == 'phasepoints':
                for point in val:
                    self.append(point)
            else:
                if hasattr(self, key):
                    setattr(self, key, val)
    
    def write_restart_file(self, loc):
        with open(loc, 'wb') as outfile:
            pickle.dump(self.restart_info(), outfile)

    def empty_path(self, **kwargs):
        """Return an empty path of same class as the current one.

        Returns
        -------
        out : object like :py:class:`.PathBase`
            A new empty path.

        """
        maxlen = kwargs.get('maxlen', None)
        time_origin = kwargs.get('time_origin', 0)
        return self.__class__(maxlen=maxlen,
                              time_origin=time_origin)

    def __eq__(self, other):
        """Check if two paths are equal."""
        if self.__class__ != other.__class__:
            logger.debug('%s and %s.__class__ differ', self, other)
            print('crab 1')
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug('%s and %s.__dict__ differ', self, other)
            print('crab 2')
            return False

        # Compare phasepoints:
        if not len(self.phasepoints) == len(other.phasepoints):
            print('crab 3')
            return False
        for i, j in zip(self.phasepoints, other.phasepoints):
            if not i == j:
                print('toto', i, j)
                print('crab 4')
                return False
        if self.phasepoints:
            # Compare other attributes:
            for i in ('maxlen', 'time_origin', 'status', 'generated',
                      'length', 'ordermax', 'ordermin', 'path_number'):
                attr_self = hasattr(self, i)
                attr_other = hasattr(other, i)
                if attr_self ^ attr_other:  # pragma: no cover
                    logger.warning('Failed comparing path due to missing "%s"',
                                   i)
                    print('crab 5')
                    return False
                if not attr_self and not attr_other:
                    logger.warning(
                        'Skipping comparison of missing path attribute "%s"',
                        i)
                    continue
                if getattr(self, i) != getattr(other, i):
                    print('crab 6')
                    return False
        return True

    def __ne__(self, other):
        """Check if two paths are not equal."""
        return not self == other

    def delete(self, idx):
        """Remove a phase point from the path.

        Parameters
        ----------
        idx : integer
            The index of the frame to remove.

        """
        del self.phasepoints[idx]

    def sorting(self, key, reverse=False):
        """Re-order the phase points according to the given key.

        Parameters
        ----------
        key : string
            The attribute we will sort according to.
        reverse : boolean, optional
            If this is False, the sorting is from big to small.

        Yields
        ------
        out : object like :py:class:`.System`
            The ordered phase points from the path.

        """
        if key in ('ekin', 'vpot'):
            sort_after = [getattr(i.particles, key) for i in self.phasepoints]
        elif key == 'order':
            sort_after = [getattr(i, key)[0] for i in self.phasepoints]
        else:
            sort_after = [getattr(i, key) for i in self.phasepoints]
        idx = np.argsort(sort_after)
        if reverse:
            idx = idx[::-1]
        self.phasepoints = [self.phasepoints[i] for i in idx]

    def update_energies(self, ekin, vpot):
        """Update the energies for the phase points.
    
        This method is useful in cases where the energies are
        read from external engines and returned as a list of
        floats.
    
        Parameters
        ----------
        ekin : list of floats
            The kinetic energies to set.
        vpot : list of floats
            The potential energies to set.
    
        """
        if len(ekin) != len(vpot):
            logger.debug(
                'Kinetic and potential energies have different length.'
            )
        if len(ekin) != len(self.phasepoints):
            logger.debug(
                'Length of kinetic energy and phase points differ %d != %d.',
                len(ekin), len(self.phasepoints)
            )
        if len(vpot) != len(self.phasepoints):
            logger.debug(
                'Length of potential energy and phase points differ %d != %d.',
                len(vpot), len(self.phasepoints)
            )
        for i, phasepoint in enumerate(self.phasepoints):
            try:
                vpoti = vpot[i]
            except IndexError:
                logger.warning(
                    'Ran out of potential energies, setting to None.'
                )
                vpoti = None
            try:
                ekini = ekin[i]
            except IndexError:
                logger.warning(
                    'Ran out of kinetic energies, setting to None.'
                )
                ekini = None
            phasepoint.vpot = vpoti
            phasepoint.ekin = ekini

def paste_paths(path_back, path_forw, overlap=True, maxlen=None):
    """Merge a backward with a forward path into a new path.

    The resulting path is equal to the two paths stacked, in correct
    time. Note that the ordering is important here so that:
    ``paste_paths(path1, path2) != paste_paths(path2, path1)``.

    There are two things we need to take care of here:

    - `path_back` must be iterated in reverse (it is assumed to be a
      backward trajectory).
    - we may have to remove one point in `path2` (if the paths overlap).

    Parameters
    ----------
    path_back : object like :py:class:`.PathBase`
        This is the backward trajectory.
    path_forw : object like :py:class:`.PathBase`
        This is the forward trajectory.
    overlap : boolean, optional
        If True, `path_back` and `path_forw` have a common
        starting-point, that is, the first point in `path_forw` is
        identical to the first point in `path_back`. In time-space, this
        means that the *first* point in `path_forw` is identical to the
        *last* point in `path_back` (the backward and forward path
        started at the same location in space).
    maxlen : float, optional
        This is the maximum length for the new path. If it's not given,
        it will just be set to the largest of the `maxlen` of the two
        given paths.

    Note
    ----
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
            msg = 'Unequal length: Using {} for the new path!'.format(maxlen)
            logger.warning(msg)
    time_origin = path_back.time_origin - path_back.length + 1
    new_path = path_back.empty_path(maxlen=maxlen, time_origin=time_origin)
    for phasepoint in reversed(path_back.phasepoints):
        app = new_path.append(phasepoint)
        if not app:
            msg = 'Truncated while pasting backwards at: {}'
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
            msg = 'Truncated path at: {}'.format(new_path.length)
            logger.warning(msg)
            return new_path
    return new_path


def load_trajtxt(dirname):
    traj_file_name = os.path.join(dirname, 'traj.txt')
    with PathExtFile(traj_file_name, 'r') as trajfile:
        # Just get the first trajectory:
        traj = next(trajfile.load())

        loc = dirname
        # Update trajectory to use full path names:
        for i, snapshot in enumerate(traj['data']):
            config = os.path.join(dirname, snapshot[1])
            traj['data'][i][1] = config
            reverse = (int(snapshot[3]) == -1)
            idx = int(snapshot[2])
            traj['data'][i][2] = idx
            traj['data'][i][3] = reverse
        return traj

def load_ordertxt(traj, dirname, order_function, system, engine):
    order_file_name = os.path.join(dirname, 'order.txt')
    with OrderPathFile(order_file_name, 'r') as orderfile:
        order = next(orderfile.load())
        return order['data'][:, 1:]

def restart_path(restart_file):
    restart_info = read_restart_file(restart_file)
    new_path = Path()
    new_path.load_restart_info(restart_info)
    return new_path

def load_path(pdir):
    trajtxt = os.path.join(pdir, 'traj.txt')
    ordertxt = os.path.join(pdir, 'order.txt')
    assert os.path.isfile(trajtxt)
    assert os.path.isfile(ordertxt)

    # load trajtxt
    with PathExtFile(trajtxt, 'r') as trajfile:
        # Just get the first trajectory:
        traj = next(trajfile.load())

        # Update trajectory to use full path names:
        for i, snapshot in enumerate(traj['data']):
            config = os.path.join(pdir, 'accepted', snapshot[1])
            traj['data'][i][1] = config
            reverse = (int(snapshot[3]) == -1)
            idx = int(snapshot[2])
            traj['data'][i][2] = idx
            traj['data'][i][3] = reverse

        for config in set([frame[1] for frame in traj['data']]):
            assert os.path.isfile(config)

    # load ordertxt
    with OrderPathFile(ordertxt, 'r') as orderfile:
        orderdata = next(orderfile.load())['data'][:, 1:]

    path = Path()
    for snapshot, order in zip(traj['data'], orderdata):
        frame = System()
        frame.order = order
        frame.config = (snapshot[1], snapshot[2])
        frame.vel = snapshot[3]
        path.phasepoints.append(frame)
    _load_energies_for_path(path, pdir)
    # CHECK PATH SOMEWHERE .acc, sta = _check_path(path, path_ensemble)
    return path

def _check_path(path, path_ensemble, warning=True):
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
    status = 'ACC'

    if start is None or start not in path_ensemble.start_condition:
        msg = "Initial path for %s starts at the wrong interface!"
        status = 'SWI'
        accept = False
    if end not in ('R', 'L'):
        msg = "Initial path for %s ends at the wrong interface!"
        status = 'EWI'
        accept = False
    if not cross[1]:
        msg = "Initial path for %s does not cross the middle interface!"
        status = 'NCR'
        accept = False

    if not accept:
        if warning:
            logger.critical(msg, path_ensemble.ensemble_name)
        else:
            logger.debug(msg, path_ensemble.ensemble_name)

    path.status = status
    return accept, status

def _load_energies_for_path(path, dirname):
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
    energy_file_name = os.path.join(dirname, 'energy.txt')
    try:
        with EnergyPathFile(energy_file_name, 'r') as energyfile:
            energy = next(energyfile.load())
            path.update_energies(energy['data']['ekin'],
                            energy['data']['vpot'])
    except FileNotFoundError:
        pass


def load_paths(config):
    load_dir = config['simulation']['load_dir']
    paths = []
    for pn in config['current']['active']:
        restart_file = os.path.join(load_dir, str(pn), 'path.restart') 
        if os.path.isfile(restart_file):
            paths.append(restart_path(restart_file))
        else:
            # load path
            new_path = load_path(os.path.join(load_dir, str(pn)))
            new_path.write_restart_file(restart_file)
            paths.append(load_path(os.path.join(load_dir, str(pn))))
        # assign pn
        paths[-1].path_number = pn
    return paths
