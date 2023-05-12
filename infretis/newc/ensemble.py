from infretis.classes.randomgen import create_random_generator
import collections
import os
import shutil
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class PathEnsemble:
    def __init__(self, ens_num, interfaces,
                 rgen=None, engine='turtle', mc_move=None):
        if rgen is None:
            rgen = create_random_generator()
        self.rgen = rgen
        self.ens_num = ens_num
        self.interfaces = tuple(interfaces)  # Should not change interfaces.
        self.last_path = None
        self.engine = engine
        self.worker = None
        self.mc_move = mc_move
        self.start_cond = None
        self.tis_set = None

        if self.ens_num == 0:
            self.ensemble_name = '[0^-]'
            self.start_cond= 'R'
        else:
            ens_num = self.ens_num - 1
            self.ensemble_name = f'[{ens_num}^+]'
            self.start_cond= 'L'
        self.ensemble_name_simple = generate_ensemble_name(
            self.ens_num
        )
        self.directory = collections.OrderedDict()

    def directories(self):
        """Yield the directories PyRETIS should make."""
        for key in self.directory:
            yield self.directory[key]

    def get_shooting_point(self, path):
        idx = self.rgen.random_integers(1, path.length - 2)
        logger.debug("Selected point with orderp %s",
                     path.phasepoints[idx].order[0])
        return path.phasepoints[idx], idx

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

def create_ensembles(config):
    intfs = config['simulation']['interfaces']
    ens_intfs = []

    # set intfs for [0-] and [0+]
    ens_intfs.append([float('-inf'), intfs[0], intfs[0]])
    ens_intfs.append([intfs[0], intfs[0], intfs[-1]])

    # set interfaces and set detect for [1+], [2+], ...
    reactant, product = intfs[0], intfs[-1]
    for i, i_ens in enumerate(range(2, len(intfs))):
        middle = intfs[i + 1]
        ens_intfs.append([reactant, middle, product])

    # create all path ensembles
    pensembles = {}
    for i, ens_intf in enumerate(ens_intfs):
        rgen_ens = create_random_generator()   ##############RESTART SEED FROM RESTART...
        engine = config['engine']['engine']    ##############GROMACS
        move = config['simulation']['shooting_moves'][i]
        mc_move = {'move': move}
        if move in ('wf', 'wt', 'ss'):
            mc_move['high_accept'] = config['simulation']['tis_set'].get('high_accept', True)
            mc_move['n_jumps'] = config['simulation']['tis_set'].get('n_jumps', True)
        if 'interface_cap' in config['simulation']['tis_set']:
            mc_move['interface_cap'] = config['simulation']['tis_set']['interface_cap']

        pensembles[i] = PathEnsemble(i, ens_intf, rgen_ens, engine, mc_move)
        pensembles[i].tis_set = config['simulation']['tis_set']

    return pensembles

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
