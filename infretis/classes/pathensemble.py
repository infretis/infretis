from infretis.classes.path import Path
from infretis.classes.randomgen import create_random_generator

import collections
import os
import shutil

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class PathEnsemble:
    """Representation of a path ensemble.

    This class represents a collection of paths in a path ensemble.
    In general, paths may be "long and complicated" so here, we really
    just store a simplified abstraction of the path, which is obtained
    by the `Path.get_path_data()` function for a given `Path` object.
    The returned dictionary is stored in the list `PathEnsemble.paths`.
    The only full path we store is the last accepted path. This is
    convenient for the RETIS method where paths may be swapped between
    path ensembles.

    Attributes
    ----------
    ensemble_number : integer
        This integer is used to represent the path ensemble, for RETIS
        simulations it's useful to identify the path ensemble. The path
        ensembles are numbered sequentially 0, 1, 2, etc. This
        corresponds to ``[0^-]``, ``[0^+]``, ``[1^+]``, etc.
    ensemble_name : string
        A string which can be used for printing the ensemble name.
        This is of form ``[0^-]``, ``[0^+]``, ``[1^+]``, etc.
    ensemble_name_simple : string
        A string with a simpler representation of the ensemble name,
        can be used for creating output files etc.
    interfaces : list of floats
        Interfaces, specified with the values for the
        order parameters: `[left, middle, right]`.
    paths : list
        This list contains the stored information for the paths. Here
        we only store the data returned by calling the `get_path_data()`
        function of the `Path` object.
    nstats : dict of ints
        This dict store some statistics for the path ensemble. The keys
        are:

        * npath : The number of paths stored.
        * nshoot : The number of accepted paths generated by shooting.
        * ACC, BWI, ... : Number of paths with given status (from `_STATUS`).
    maxpath : int
        The maximum number of paths to store.
    last_path : object like :py:class:`.PathBase`
        This is the last **accepted** path.

    """

    def __init__(self, ensemble_number, interfaces,
                 rgen=None, maxpath=10000, exe_dir=None):
        """Initialise the PathEnsemble object.

        Parameters
        ----------
        ensemble_number : integer
            An integer used to identify the ensemble.
        interfaces : list of floats
            These are the interfaces specified with the values
            for the order parameters: ``[left, middle, right]``.
        rgen : object like :py:class:`.RandomGenerator`, optional
            The random generator that will be used for the
            paths that required random numbers.
        maxpath : integer, optional
            The maximum number of paths to store information for in memory.
            Note, that this will not influence the analysis as long as
            you are using the output files when running the analysis.
        exe_dir : string, optional
            The base folder where the simulation was executed from.
            This is used to set up output directories for the path
            ensemble.

        """
        if rgen is None:
            rgen = create_random_generator()
        self.rgen = rgen
        self.ensemble_number = ensemble_number
        self.path_number = None
        self.interfaces = tuple(interfaces)  # Should not change interfaces.
        self.last_path = None
        self.nstats = {'npath': 0, 'nshoot': 0, 'ACC': 0}
        self.paths = []
        self.maxpath = maxpath
        if self.ensemble_number == 0:
            self.ensemble_name = '[0^-]'
            self.start_condition = 'R'
        else:
            ensemble_number = self.ensemble_number - 1
            self.ensemble_name = f'[{ensemble_number}^+]'
            self.start_condition = 'L'
        self.ensemble_name_simple = generate_ensemble_name(
            self.ensemble_number
        )
        self.directory = collections.OrderedDict()
        self.directory['path_ensemble'] = None
        self.directory['accepted'] = None
        self.directory['generate'] = None
        self.directory['traj'] = None
        self.directory['exe_dir'] = exe_dir
        if exe_dir is not None:
            path_dir = os.path.join(exe_dir, self.ensemble_name_simple)
            self.update_directories(path_dir)
            self.directory['home_dir'] = exe_dir


    def __eq__(self, other):
        """Check if two path_ensemble are equal."""
        equal = True
        if self.__class__ != other.__class__:
            logger.debug('%s and %s.__class__ differ', self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug('%s and %s.__dict__ differ', self, other)
            equal = False

        for i in ['directory', 'interfaces', 'nstats', 'paths']:
            if hasattr(self, i):
                for j, k in zip(getattr(self, i), getattr(other, i)):
                    if j != k:
                        logger.debug('%s for %s and %s attributes are %s and '
                                     '%s', i, self, other, j, k)
                        equal = False

        for i in ['ensemble_name',
                  'ensemble_name_simple', 'ensemble_number',
                  'maxpath', 'start_condition']:
            if hasattr(self, i):
                if getattr(self, i) != getattr(other, i):
                    logger.debug('%s for %s and %s, attributes are %s and %s',
                                 i, self, other,
                                 getattr(self, i), getattr(other, i))
                    equal = False

        if hasattr(self, 'last_path'):
            if self.last_path != other.last_path:
                logger.debug('last paths differs')
                equal = False

        if hasattr(self, 'rgen'):
            if self.rgen.__class__ != other.rgen.__class__:
                logger.debug('self.rgen.__class__ differs')
                return False
            if self.rgen.__dict__['seed'] != other.rgen.__dict__['seed']:
                logger.debug('rgen seed differs')
                equal = False
            if not big_fat_comparer(self.rgen.__dict__['rgen'].get_state(),
                                    other.rgen.__dict__['rgen'].get_state()):
                logger.debug('rgen differs')
                equal = False

        return equal

    def __ne__(self, other):
        """Check if two paths are not equal."""
        return not self == other

    def directories(self):
        """Yield the directories PyRETIS should make."""
        for key in self.directory:
            yield self.directory[key]

    def update_directories(self, path):
        """Update directory names.

        This method will not create new directories, but it will
        update the directory names.

        Parameters
        ----------
        path : string
            The base path to set.

        """
        for key, val in self.directory.items():
            if key == 'path_ensemble':
                self.directory[key] = path
            else:
                self.directory[key] = os.path.join(path, key)
            if val is None:
                logger.debug('Setting directory "%s" to %s', key,
                             self.directory[key])
            else:
                logger.debug('Updating directory "%s": %s -> %s',
                             key, val, self.directory[key])

    def reset_data(self):
        """Erase the stored data in the path ensemble.

        It can be used in combination with flushing the data to a
        file in order to periodically write and empty the amount of data
        stored in memory.

        Notes
        -----
        We do not reset `self.last_path` as this might be used in the
        RETIS function.

        """
        self.paths = []
        for key in self.nstats:
            self.nstats[key] = 0

    def store_path(self, path):
        """Store a path by explicitly moving it.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to store.

        """
        self._move_path(path, self.directory['accepted'])
        self.last_path = path
        for entry in self.list_superfluous():
            try:
                os.remove(entry)
            except OSError:  # pragma: no cover
                pass

    def add_path_data(self, path, status, cycle=0):
        """Append data from the given path to `self.path_data`.

        This will add the data from a given` path` to the list path data
        for this ensemble. If will also update `self.last_path` if the
        given `path` is accepted.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the object to store data from.
        status : string
            This is the status of the path. Note that the path object
            also has a status property. However, this one might not be
            set, for instance when the path is just None. We therefore
            use `status` here as a parameter.
        cycle : int, optional
            The current cycle number.

        """
        if len(self.paths) >= self.maxpath:
            # This is just to limit the data we keep in memory in
            # case of really long simulations.
            logger.debug(('Path-data memory storage reset for ensemble %s.\n'
                          'This is just to limit the amount of data we store '
                          'in memory.\nThis will *NOT* influence the '
                          'simulation'), self.ensemble_name)
            self.paths = []
        # Update statistics:
        if path is None:
            # Here we add a dummy path with minimal info. This is because we
            # could not generate a path for some reason which should be
            # specified by the status.
            path_data = {'status': status, 'generated': ('', 0, 0, 0),
                         'weight': 1.}
        else:
            path_data = path.get_path_data(status, self.interfaces)
            if 'EXP' in status:
                path_data['status'] = 'EXP'
            if path_data['status'] in {'ACC', 'EXP'}:  # Store the path:
                self.store_path(path)
                if path_data['generated'][0] in {'sh', 'ss', 'wt', 'wf'}:
                    self.nstats['nshoot'] += 1
        path_data['cycle'] = cycle  # Also store cycle number.
        self.paths.append(path_data)  # Store the new data.
        # Update some statistics:
        # This is to count also for the first occurrence of the status:
        self.nstats[status] = self.nstats.get(status, 0) + 1
        self.nstats['npath'] += 1

    def get_accepted(self):
        """Yield accepted paths from the path ensemble.

        This function will return an iterator useful for iterating over
        accepted paths only. In the path ensemble we store both accepted
        and rejected paths. This function will loop over all paths
        stored and yield the accepted paths the correct number of times.
        """
        last_path = None
        for path in self.paths:
            if path['status'] == 'ACC':
                last_path = path
            yield last_path

    def get_acceptance_rate(self):
        """Return acceptance rate for the path ensemble.

        The acceptance rate is obtained as the fraction of accepted
        paths to the total number of paths in the path ensemble. This
        will only consider the paths that are currently stored in
        `self.paths`.

        Returns
        -------
        out : float
            The acceptance rate.

        """
        acc = 0
        npath = 0
        for path in self.paths:
            if path['status'] == 'ACC':
                acc += 1
            npath += 1
        return float(acc) / float(npath)

    def get_paths(self):
        """Yield the different paths stored in the path ensemble.

        It is included here in order to have a simple compatibility
        between the :py:class:`.PathEnsemble` object and the
        py:class:`.PathEnsembleFile` object. This is useful for the
        analysis.

        Yields
        ------
        out : dict
            This is the dictionary representing the path data.

        """
        for path in self.paths:
            yield path

    def move_path_to_generate(self, path, prefix=None):
        """Move a path for temporary storing."""
        self._move_path(path, self.directory['generate'], prefix=prefix)

    def __str__(self):
        """Return a string with some info about the path ensemble."""
        msg = [f'Path ensemble: {self.ensemble_name}']
        msg += [f'\tInterfaces: {self.interfaces}']
        if self.nstats['npath'] > 0:
            npath = self.nstats['npath']
            nacc = self.nstats.get('ACC', 0)
            msg += [f'\tNumber of paths stored: {npath}']
            msg += [f'\tNumber of accepted paths: {nacc}']
            ratio = float(nacc) / float(npath)
            msg += [f'\tRatio accepted/total paths: {ratio}']
        return '\n'.join(msg)

    def restart_info(self):
        """Return a dictionary with restart information."""
        restart = {
            'nstats': self.nstats,
            'interfaces': self.interfaces,
            'ensemble_number': self.ensemble_number,
        }
        if hasattr(self, 'rgen'):
            restart['rgen'] = self.rgen.get_state()

        if self.last_path:
            restart['last_path'] = self.last_path.restart_info()

        return restart

    def load_restart_info(self, info, cycle=0):
        """Load restart information.

        Parameters
        ----------
        info : dict
            A dictionary with the restart information.
        cycle : integer, optional
            The current simulation cycle.

        """
        self.nstats = info['nstats']
        # for attr in ('interfaces', 'ensemble_number'):
        #     if info[attr] != getattr(self, attr):
        #         logger.warning(
        #             'Inconsistent path ensemble restart info for %s', attr)
        for key in info:
            if key == 'rgen':
                self.rgen = create_random_generator(info[key])
            elif key == 'last_path':
                rgen = create_random_generator(info[key]['rgen'])
                path = Path(rgen=rgen)
                path.load_restart_info(info['last_path'])
                path_data = path.get_path_data('ACC', self.interfaces)
                path_data['cycle'] = cycle
                self.last_path = path
                self.paths.append(path_data)
            elif hasattr(self, key):
                setattr(self, key, info[key])

        # Update file names:
        directory = self.directory['accepted']
        self._copy_path(self.last_path, self.directory['accepted'])
        for entry in self.list_superfluous():
            try:
                os.remove(entry)
            except OSError:  # pragma: no cover
                pass

        for phasepoint in self.last_path.phasepoints:
            filename = os.path.basename(phasepoint.particles.get_pos()[0])
            new_file_name = os.path.join(directory, filename)
            if not os.path.isfile(new_file_name):
                logger.critical('The restart path "%s" does not exist',
                                new_file_name)
            phasepoint.particles.set_pos((new_file_name,
                                          phasepoint.particles.get_pos()[1]))

    def list_superfluous(self):
        """List files in accepted directory that we do not need."""
        last = set()
        if self.last_path:
            for phasepoint in self.last_path.phasepoints:
                pos_file, _ = phasepoint.particles.get_pos()
                last.add(os.path.split(pos_file)[-1])
        for entry in os.scandir(self.directory['accepted']):
            if entry.is_file() and os.path.split(entry.path)[-1] not in last:
                yield entry.path

    @staticmethod
    def _move_path(path, target_dir, prefix=None):
        """Move a path to a given target directory.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to move.
        target_dir : string
            The location where we are moving the path to.
        prefix : string, optional
            To give a prefix to the name of moved files.

        """
        logger.debug('Moving path to %s', target_dir)
        new_pos, source = _generate_file_names(path, target_dir,
                                               prefix=prefix)
        for pos, phasepoint in zip(new_pos, path.phasepoints):
            phasepoint.particles.set_pos(pos)
        for src, dest in source.items():
            if src == dest:
                logger.debug('Skipping move %s -> %s', src, dest)
            else:
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        logger.debug('Removing %s as it exists', dest)
                        os.remove(dest)
                logger.debug('Moving %s -> %s', src, dest)
                os.rename(src, dest)

    @staticmethod
    def _copy_path(path, target_dir, prefix=None):
        """Copy a path to a given target directory.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to copy.
        target_dir : string
            The location where we are copying the path to.
        prefix : string, optional
            To give a prefix to the name of copied files.

        Returns
        -------
        out : object like py:class:`.PathBase`
            A copy of the input path.

        """
        path_copy = path.copy()
        new_pos, source = _generate_file_names(path_copy, target_dir,
                                               prefix=prefix)
        # Update positions:
        for pos, phasepoint in zip(new_pos, path_copy.phasepoints):
            phasepoint.particles.set_pos(pos)
        for src, dest in source.items():
            if src != dest:
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        logger.debug('Removing %s as it exists', dest)
                        os.remove(dest)
                logger.debug('Copy %s -> %s', src, dest)
                shutil.copy(src, dest)
        return path_copy

def generate_ensemble_name(ensemble_number, zero_pad=3):
    """Generate a simple name for an ensemble.

    The simple name will have a format like 01, 001, 0001 etc. and it
    is used to name the path ensemble and the output directory.

    Parameters
    ----------
    ensemble_number : int
        The number representing the ensemble.
    zero_pad : int, optional
        The number of zeros to use for padding the name.

    Returns
    -------
    out : string
        The ensemble name.

    """
    if zero_pad < 3:
        logger.warning('zero_pad must be >= 3. Setting it to 3.')
        zero_pad = 3
    fmt = f'{{:0{zero_pad}d}}'
    return fmt.format(ensemble_number)

def _generate_file_names(path, target_dir, prefix=None):
    """Generate new file names for moving copying paths.

    Parameters
    ----------
    path : object like :py:class:`.PathBase`
        This is the path object we are going to store.
    target_dir : string
        The location where we are moving the path to.
    prefix : string, optional
        The prefix can be used to prefix the name of the files.

    Returns
    -------
    out[0] : list
        A list with new file names.
    out[1] : dict
        A dict which defines the unique "source -> destination" for
        copy/move operations.

    """
    source = {}
    new_pos = []
    for phasepoint in path.phasepoints:
        pos_file, idx = phasepoint.particles.get_pos()
        if pos_file not in source:
            localfile = os.path.basename(pos_file)
            if prefix is not None:
                localfile = f'{prefix}{localfile}'
            dest = os.path.join(target_dir, localfile)
            source[pos_file] = dest
        dest = source[pos_file]
        new_pos.append((dest, idx))
    return new_pos, source