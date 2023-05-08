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

        if self.ens_num == 0:
            self.ensemble_name = '[0^-]'
            self.start_condition = 'R'
        else:
            ens_num = self.ens_num - 1
            self.ensemble_name = f'[{ens_num}^+]'
            self.start_condition = 'L'
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
        pensembles[i] = PathEnsemble(i, ens_intf, rgen_ens, engine, move)

    return pensembles
