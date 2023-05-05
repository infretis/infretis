from infretis.classes.system import System
from infretis.classes.particles import Particles
from infretis.classes.pathensemble import PathEnsemble
from infretis.classes.simulation import Simulation
from infretis.classes.orderparameter import create_orderparameter
from infretis.core.gromacs import gromacs_settings
from infretis.core.gromacs import read_gromacs_file, read_gromos96_file
from infretis.core.common import (prepare_engine,
                                  prepare_system,
                                  print_to_screen,
                                  write_ensemble_restart,
                                  initiate_path_simulation,
                                  create_random_generator,
                                  task_from_settings,
                                  make_dirs)

from datetime import datetime
import os
import struct
import ast
from copy import copy as copy0
from copy import deepcopy
from abc import ABCMeta, abstractmethod
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Define formats for the trajectory output:
ALLOW_MULTIPLE = {
    'collective-variable',
    'engine',
    'ensemble',
    'initial-path',
    'orderparameter',
    'potential'}
SPECIAL_MULTIPLE = {
    'collective-variable',
    'ensemble',
    'potential'}
URL = 'http://www.pyretis.org'
GIT_URL = 'https://gitlab.com/pyretis/pyretis'
SECTIONS = { }
HEADING = '{}\n{}\nFor more info, please see: {}\nHave Fun!'
SECTIONS['heading'] = {
    'text': 'frog' }
HERE = os.path.abspath('.')
SECTIONS['simulation'] = {
    'endcycle': None,
    'exe_path': HERE,
    'flux': None,
    'interfaces': None,
    'restart': None,
    'rgen': 'rgen',
    'seed': None,
    'startcycle': None,
    'steps': None,
    'task': 'md',
    'zero_ensemble': None,
    'zero_left': None,
    'permeability': None,
    'swap_attributes': None,
    'priority_shooting': False,
    'overlap': None,
    'mincycle': None }
SECTIONS['engine'] = {
    'class': None,
    'exe_path': HERE,
    'module': None,
    'rgen': 'rgen' }
SECTIONS['forcefield'] = {
    'description': None }
SECTIONS['potential'] = {
    'class': None,
    'parameter': None }
SECTIONS['orderparameter'] = {
    'class': None,
    'module': None,
    'name': 'Order Parameter' }
SECTIONS['collective-variable'] = {
    'class': None,
    'module': None,
    'name': None }
SECTIONS['box'] = {
    'cell': None,
    'high': None,
    'low': None,
    'periodic': None }
SECTIONS['particles'] = {
    'mass': None,
    'name': None,
    'npart': None,
    'position': None,
    'ptype': None,
    'type': 'internal',
    'velocity': None }
SECTIONS['system'] = {
    'dimensions': 3,
    'input_file': None,
    'temperature': 1,
    'units': 'lj' }
SECTIONS['unit-system'] = {
    'charge': None,
    'energy': None,
    'length': None,
    'mass': None,
    'name': None }
SECTIONS['tis'] = {
    'allowmaxlength': False,
    'aimless': True,
    'ensemble_number': None,
    'detect': None,
    'freq': None,
    'maxlength': None,
    'nullmoves': None,
    'n_jumps': None,
    'high_accept': False,
    'interface_sour': None,
    'interface_cap': None,
    'relative_shoots': None,
    'rescale_energy': False,
    'rgen': 'rgen',
    'seed': None,
    'shooting_move': 'sh',
    'shooting_moves': [],
    'sigma_v': -1,
    'zero_momentum': False }
SECTIONS['output'] = {
    'backup': 'append',
    'cross-file': 1,
    'energy-file': 1,
    'pathensemble-file': 1,
    'prefix': None,
    'order-file': 1,
    'restart-file': 1,
    'screen': 10,
    'trajectory-file': 100 }
SECTIONS['initial-path'] = {
    'method': None }
SECTIONS['retis'] = {
    'nullmoves': None,
    'relative_shoots': None,
    'rgen': None,
    'seed': None,
    'swapfreq': None,
    'swapsimul': None }
SECTIONS['ensemble'] = {
    'interface': None }
SPECIAL_KEY = {
    'parameter'}


def read_xyz_file(filename):
    """Read files in XYZ format.

    This method will read a XYZ file and yield the different snapshots
    found in the file.

    Parameters
    ----------
    filename : string
        The file to open.

    Yields
    ------
    out : dict
        This dict contains the snapshot.

    Examples
    ----
    The positions will **NOT** be converted to a specified set of units.

    """
    xyz_keys = ('atomname', 'x', 'y', 'z', 'vx', 'vy', 'vz')
    for snapshot in read_txt_snapshots(filename, data_keys=xyz_keys):
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

READFILE = {'xyz': {'reader': read_xyz_file,
                    'units': {'length': 'A', 'velocity': 'A/fs'}},
            'txt': {'reader': read_txt_snapshots,
                    'units': None},
            'gro': {'reader': read_gromacs_file,
                    'units': {'length': 'nm', 'velocity': 'nm/ps'}},
            'g96': {'reader': read_gromos96_file,
                    'units': {'length': 'nm', 'velocity': 'nm/ps'}}}

def create_ensembles(settings):
    """Create a list of dictionary from ensembles simulation settings.

    Parameters
    ----------
    settings : dict
        This dict contains the settings needed to create the path
        ensemble.

    Returns
    -------
    ensembles : list of dicts
        List of ensembles.

    """
    # Example:
    # ensembles, assuming len(interfaces) = 3
    # (RE)TIS:   flux=False, zero_ensemble=False  [1+]
    # (RE)TIS:   flux=False, zero_ensemble=True   [0+], [1+]
    # (RE)TIS:   flux=True,  zero_ensemble=True   [0-], [0+], [1+]
    # /      :   flux=True,  zero_ensemble=False  doesn't make sense
    # so number of ensembles can be 1, 2, or 3
    ensembles = []
    intf = settings['simulation']['interfaces']

    j_ens = 0
    # add [0-] if it exists
    if settings['simulation'].get('flux', False) or \
            settings['simulation'].get('zero_left', False):
        reactant, middle, product = float('-inf'), intf[0], intf[0]
        if settings['simulation'].get('zero_left', False):
            reactant = settings['simulation']['zero_left']
            if settings['simulation'].get('permeability', False):
                middle = (settings['simulation']['zero_left'] + intf[0]) / 2
        settings['ensemble'][j_ens]['simulation']['interfaces'] =\
            [reactant, middle, product]
        j_ens += 1

    # add [0+] if it exists
    if settings['simulation'].get('zero_ensemble', True):
        reactant, middle, product = intf[0], intf[0], intf[-1]

        settings['ensemble'][j_ens]['simulation']['interfaces'] = \
            [reactant, middle, product]
        j_ens += 1

    # j_ens is the number of ensembles that is skipped

    # set interfaces and set detect for [1+], [2+], ...
    reactant, product = intf[0], intf[-1]
    for i, i_ens in enumerate(range(j_ens, len(settings['ensemble']))):
        middle = intf[i + 1]    # the lambda_i interface
        settings['ensemble'][i_ens]['simulation']['interfaces'] = \
            [reactant, middle, product]
        settings['ensemble'][i_ens]['tis']['detect'] = intf[i + 2]  # next intf
    # create all ensembles
    for i in range(len(settings['ensemble'])):
        ensembles.append(create_ensemble(settings['ensemble'][i]))

    return ensembles

def create_ensemble(settings):
    """Create the path ensemble from (ensemble) simulation settings.

    Parameters
    ----------
    settings : dict
        This dict contains the settings needed to create the path
        ensemble.

    Returns
    -------
    ensemble : dict of
        objects that contains all the information needed in the ensemble.

    """
    i_ens = settings['tis']['ensemble_number']

    logtxt = f'\nCREATING  ENSEMBLE  {i_ens}\n====================='
    print_to_screen(logtxt, level='message')
    logger.info(logtxt)

    rgen_ens = create_random_generator(settings['tis'])
    rgen_path = create_random_generator(settings['system'])

    system = prepare_system(settings)
    engine = prepare_engine(settings)
    interfaces = settings['simulation']['interfaces']
    exe_dir = settings['simulation'].get('exe_path', os.path.abspath('.'))
    path_ensemble = PathEnsemble(i_ens, interfaces, rgen=rgen_path, exe_dir=exe_dir)

    order_function = create_orderparameter(settings)
    if i_ens == 0 and settings['simulation'].get("permeability", False):
        path_ensemble.start_condition = ['R', 'L']
        if hasattr(order_function, 'mirror_pos'):
            offset = getattr(order_function, 'offset', 0)
            moved_interfaces = [i - offset for i in interfaces]
            left, right = moved_interfaces[0], moved_interfaces[2]
            correct_mirror = (left+right)/2.
            if abs(order_function.mirror_pos-correct_mirror) > 1E-5:
                msg = "Order function should have a mirror at "
                msg += f"{correct_mirror}, found one at "
                msg += f"{order_function.mirror_pos} instead."
                raise ValueError(msg)
    # Add check to see if mirror makes sense
    engine.can_use_order_function(order_function)
    ensemble = {'engine': engine,
                'system': system,
                'order_function': order_function,
                'interfaces': interfaces,
                'exe_path': exe_dir,
                'path_ensemble': path_ensemble,
                'rgen': rgen_ens}
    return ensemble

def reopen_file(filename, fileh, inode, bytes_read):
    """Reopen a file if the inode has changed.

    Parameters
    ----------
    filename : string
        The name of the file we are working with.
    fileh : file object
        The current open file object.
    inode : integer
        The current inode we are using.
    bytes_read : integer
        The position we should start reading at.

    Returns
    -------
    out[0] : file object or None
        The new file object.
    out[1] : integer or None
        The new inode.

    """
    if os.stat(filename).st_ino != inode:
        new_fileh = open(filename, 'rb')
        fileh.close()
        new_inode = os.fstat(new_fileh.fileno()).st_ino
        new_fileh.seek(bytes_read)
        return new_fileh, new_inode
    return None, None

def create_simulation(settings):
    """Create simulation(s) from given settings.

    This function will set up some common simulation types.
    It is meant as a helper function to automate some very common set-up
    task. It will here check what kind of simulation we are to perform
    and then call the appropriate function for setting that type of
    simulation up.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    simulation : object like :py:class:`.Simulation`
        This object will correspond to the selected simulation type.

    """

    # Improve setting quality
    add_default_settings(settings)
    check_ensemble(settings)
    ensembles = create_ensembles(settings)

    controls = {'rgen': create_random_generator(settings['simulation']),
                'steps': settings['simulation']['steps'],
                'startcycle': settings['simulation'].get('startcycle', 0)}
    simulation = SimulationRETIS(ensembles, settings, controls)
    msgtxt = '{}'.format(simulation)
    logger.info('Created simulation:\n%s', msgtxt)

    return simulation

class PathSimulation(Simulation):
    """A base class for TIS/RETIS simulations.

    Attributes
    ----------
    ensembles : list of dictionaries of objects
        Each contains:

        * `path_ensemble`: objects like :py:class:`.PathEnsemble`
          This is used for storing results for the different path ensembles.
        * `engine`: object like :py:class:`.EngineBase`
          This is the integrator that is used to propagate the system in time.
        * `rgen`: object like :py:class:`.RandomGenerator`
          This is a random generator used for the generation of paths.
        * `system`: object like :py:class:`.System`
          This is the system the simulation will act on.

    settings : dict
        A dictionary with TIS and RETIS settings. We expect that
        we can find ``settings['tis']`` and possibly
        ``settings['retis']``. For ``settings['tis']`` we further
        expect to find the keys:

        * `aimless`: Determines if we should do aimless shooting
          (True) or not (False).
        * `sigma_v`: Scale used for non-aimless shooting.
        * `seed`: A integer seed for the random generator used for
          the path ensemble we are simulating here.

        Note that the
        :py:func:`pyretis.core.tis.make_tis_step_ensemble` method
        will make use of additional keys from ``settings['tis']``.
        Please see this method for further details. For the
        ``settings['retis']`` we expect to find the following keys:

        * `swapfreq`: The frequency for swapping moves.
        * `relative_shoots`: If we should do relative shooting for
          the path ensembles.
        * `nullmoves`: Should we perform null moves.
        * `swapsimul`: Should we just swap a single pair or several pairs.

      required_settings : tuple of strings
        This is just a list of the settings that the simulation
        requires. Here it is used as a check to see that we have
        all we need to set up the simulation.

    """

    required_settings = ('tis', 'retis')
    name = 'Generic path simulation'
    simulation_type = 'generic-path'
    simulation_output = [
        {
            'type': 'pathensemble',
            'name': 'path_ensemble',
            'result': ('pathensemble-{}',),
        },
        {
            'type': 'path-order',
            'name': 'path_ensemble-order',
            'result': ('path-{}', 'status-{}'),
        },
        {
            'type': 'path-traj-{}',
            'name': 'path_ensemble-traj',
            'result': ('path-{}', 'status-{}', 'pathensemble-{}'),
        },
        {
            'type': 'path-energy',
            'name': 'path_ensemble-energy',
            'result': ('path-{}', 'status-{}'),
        },
    ]

    def __init__(self, ensembles, settings, controls):
        """Initialise the path simulation object.

        Parameters
        ----------
        ensembles : list of dicts
            Each contains:

            * `path_ensemble`: object like :py:class:`.PathEnsemble`
              This is used for storing results for the simulation. It
              is also used for defining the interfaces for this simulation.
            * `system`: object like :py:class:`.System`
              This is the system we are investigating.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `engine`: object like :py:class:`.EngineBase`
              This is the integrator that is used to propagate the
              system in time.
            * `rgen`: object like :py:class:`.RandomGenerator`
              This is the random generator to use in the ensemble.

        settings : dict
            This dictionary contains the settings for the simulation.
        controls: dict of parameters to set up and control the simulations
            It contains:

            * `steps`: int, optional
              The number of simulation steps to perform.
            * `startcycle`: int, optional
              The cycle we start the simulation on, useful for restarts.
            * `rgen`: object like :py:class:`.RandomGenerator`
              This object is the random generator to use in the simulation.

        """
        super().__init__(settings, controls)
        self.ensembles = ensembles
        self.settings = settings
        self.rgen = controls.get('rgen', create_random_generator())

        for key in self.required_settings:
            if key not in self.settings:
                logtxt = 'Missing required setting "{}" for simulation "{}"'
                logtxt = logtxt.format(key, self.name)
                logger.error(logtxt)
                raise ValueError(logtxt)
            self.settings[key] = settings[key]

        # Additional setup for shooting:
        for i, ensemble in enumerate(ensembles):
            # ensemble['system'].potential_and_force()

            if self.settings['ensemble'][i]['tis']['sigma_v'] < 0.0:
                self.settings['ensemble'][i]['tis']['aimless'] = True
                logger.debug('%s: aimless is True', self.name)
            else:
                logger.debug('Path simulation: Creating sigma_v.')
                sigv = (self.settings['ensemble'][i]['tis']['sigma_v'] *
                        np.sqrt(ensemble['system'].particles.imass))
                logger.debug('Path simulation: sigma_v created and set.')
                self.settings['ensemble'][i]['tis']['sigma_v'] = sigv
                self.settings['ensemble'][i]['tis']['aimless'] = False
                logger.debug('Path simulation: aimless is False')

    def restart_info(self):
        """Return restart info.

        The restart info for the path simulation includes the state of
        the random number generator(s).

        Returns
        -------
        info : dict,
            Contains all the updated simulation settings and counters.

        """
        info = super().restart_info()
        info['simulation']['rgen'] = self.rgen.get_state()

        # Here we store only the necessary info to initialize the
        # ensemble objects constructed in `pyretis.setup.createsimulation`
        # Note: these info are going to be stored in restart.pyretis
        if hasattr(self, 'ensembles'):
            info['ensemble'] = []
            for ens in self.ensembles:
                info['ensemble'].append(
                    {'restart': os.path.join(
                        ens['path_ensemble'].ensemble_name_simple,
                        'ensemble.restart')})

        return info

    def load_restart_info(self, info):
        """Load restart information.

        Note: This method load the info for the main simulation, the actual
              ensemble restart is done in initiate_restart.

        Parameters
        ----------
        info : dict
            The dictionary with the restart information, should be
            similar to the dict produced by :py:func:`.restart_info`.

        """
        super().load_restart_info(info)
        self.rgen = create_random_generator(info['simulation']['rgen'])

    def create_output_tasks(self, settings, progress=False):
        """Create output tasks for the simulation.

        This method will generate output tasks based on the tasks
        listed in :py:attr:`.simulation_output`.

        Parameters
        ----------
        settings : dict
            These are the simulation settings.
        progress : boolean
            For some simulations, the user may select to display a
            progress bar, we then need to disable the screen output.

        """
        logging.debug('Clearing output tasks & adding pre-defined ones')
        self.output_tasks = []
        for ensemble in self.ensembles:
            path_ensemble = ensemble['path_ensemble']
            directory = path_ensemble.directory['path_ensemble']
            idx = path_ensemble.ensemble_number
            logger.info('Creating output directories for path_ensemble %s',
                        path_ensemble.ensemble_name)
            for dir_name in path_ensemble.directories():
                msg_dir = make_dirs(dir_name)
                logger.info('%s', msg_dir)
            for task_dict in self.simulation_output:
                task_dict_ens = task_dict.copy()
                if 'result' in task_dict_ens:
                    task_dict_ens['result'] = \
                        [key.format(idx) for key in task_dict_ens['result']]
                task = task_from_settings(task_dict_ens, settings, directory,
                                          ensemble['engine'], progress)
                if task is not None:
                    logger.debug('Created output task:\n%s', task)
                    self.output_tasks.append(task)

    def write_restart(self, now=False):
        """Create a restart file.

        Parameters
        ----------
        now : boolean, optional
            If True, the output file will be written irrespective of the
            step number.

        """
        super().write_restart(now=now)
        if now or (self.restart_freq is not None and
                   self.cycle['stepno'] % self.restart_freq == 0):
            for ens, e_set in zip(self.ensembles, self.settings['ensemble']):
                write_ensemble_restart(ens, e_set)

    def initiate(self, settings):
        """Initialise the path simulation.

        Parameters
        ----------
        settings : dictionary
            The simulation settings.

        """
        init = initiate_path_simulation(self, settings)
        path_numbers = iter(settings['current']['active'])
        print_to_screen('')
        for i_ens, (accept, path, status, path_ensemble) in enumerate(init):
            print_to_screen(
                f'Found initial path for {path_ensemble.ensemble_name}:',
                level='success' if accept else 'warning',
            )
            for line in str(path).split('\n'):
                print_to_screen(f'- {line}')
            logger.info('Found initial path for %s',
                        path_ensemble.ensemble_name)
            logger.info('%s', path)
            print_to_screen('')

            path_number = next(path_numbers)
            path.path_number = path_number

            idx = path_ensemble.ensemble_number
            path_ensemble_result = {
                f'pathensemble-{idx}': path_ensemble,
                f'path-{idx}': path,
                f'status-{idx}': status,
                'cycle': self.cycle,
                'system': self.system,
            }
            # If we are doing a restart, we do not print out at the
            # "restart" step as we assume that this is already
            # outputted in the "previous" simulation (the one
            # we restart from):
            # if settings['initial-path']['method'] != 'restart':
            #     for task in self.output_tasks:
            #         task.output(path_ensemble_result)
            #     write_ensemble_restart({'path_ensemble': path_ensemble},
            #                            settings['ensemble'][i_ens])
            if self.soft_exit():
                return False
        return True

class SimulationRETIS(PathSimulation):
    """A RETIS simulation.

    This class is used to define a RETIS simulation where the goal is
    to calculate crossing probabilities for several path ensembles.

    The attributes are documented in the parent class, please see:
    :py:class:`.PathSimulation`.
    """

    required_settings = ('retis',)
    name = 'RETIS simulation'
    simulation_type = 'retis'
    simulation_output = PathSimulation.simulation_output + [
        {
            'type': 'pathensemble-retis-screen',
            'name': 'path_ensemble-retis-screen',
            'result': ('pathensemble-{}',),
        },
    ]

    def __str__(self):
        """Just a small function to return some info about the simulation."""
        msg = ['RETIS simulation']
        msg += ['Ensembles:']
        for ensemble in self.ensembles:
            path_ensemble = ensemble['path_ensemble']
            msgtxt = (f'{path_ensemble.ensemble_name}: '
                      f'Interfaces: {path_ensemble.interfaces}')
            msg += [msgtxt]
        nstep = self.cycle['endcycle'] - self.cycle['startcycle']
        msg += [f'Number of steps to do: {nstep}']
        return '\n'.join(msg)

def add_default_settings(settings):
    """Add default settings.

    Parameters
    ----------
    settings : dict
        The current input settings.

    Returns
    -------
    None, but this method might add data to the input settings.

    """
    for sec, sec_val in SECTIONS.items():
        if sec not in settings:
            settings[sec] = {}
        for key, val in sec_val.items():
            if val is not None and key not in settings[sec]:
                settings[sec][key] = val
    to_remove = [key for key in settings if len(settings[key]) == 0]
    for key in to_remove:
        settings.pop(key, None)

    task = settings['simulation'].get('task')
    if task not in settings:
        settings[task] = {}

    eng_name = settings['engine'].get('class', 'NoneEngine')
    if eng_name[:7].lower() in {'gromacs', 'cp2k', 'lammps'}:
        settings['particles']['type'] = 'external'
        settings['engine']['type'] = 'external'
        input_path = os.path.join(settings['engine'].get('exe_path', '.'),
                                  settings['engine'].get('input_path', '.'))
        engine_checker = {'gromacs': gromacs_settings}
        # Checks engine specific settings
        if engine_checker.get(eng_name[:7].lower()):
            engine_checker[eng_name[:7].lower()](settings, input_path)

def check_ensemble(settings):
    """Check that the ensemble input parameters are complete.

    Parameters
    ----------
    settings : dict
        The settings needed to set up the simulation.

    """
    if 'ensemble' in settings:
        savelambda = []
        for ens in settings['ensemble']:
            if 'interface' in ens:
                savelambda.append(ens['interface'])

                if not ens['interface'] \
                        in settings['simulation']['interfaces']:
                    msg = f"The ensemble interface {ens['interface']} "\
                        "is not present in the simulation interface list"
                    break

            else:
                msg = "An ensemble is present without reference interface"
                break

        is_sorted = all(aaa <= bbb for aaa, bbb in zip(savelambda[:-1],
                                                       savelambda[1:]))
        if not is_sorted:
            msg = "Interface positions in the ensemble simulation "\
                "are NOT properly sorted (ascending order)"

    else:
        msg = "No ensemble in settings"

    if 'msg' in locals():
        raise ValueError(msg)

    return True

def look_for_input_files(input_path, required_files,
                         extra_files=None):
    """Check that required files for external engines are present.

    It will first search for the default files.
    If not present, it will search for the files with the
    same extension. In this search,
    if there are no files or multiple files for a required
    extension, the function will raise an Error.
    There might also be optional files which are not required, but
    might be passed in here. If these are not present we will
    not fail, but delete the reference to this file.

    Parameters
    ----------
    input_path : string
        The path to the folder where the input files are stored.
    required_files : dict of strings
        These are the file names types of the required files.
    extra_files : list of strings, optional
        These are the file names of the extra files.

    Returns
    -------
    out : dict
        The paths to the required and extra files we found.

    """
    if not os.path.isdir(input_path):
        msg = f'Input path folder {input_path} not existing'
        raise ValueError(msg)

    # Get the list of files in the input_path folder
    files_in_input_path = \
        [i.name for i in os.scandir(input_path) if i.is_file()]

    input_files = {}
    # Check if the required files are present
    for file_type, file_to_check in required_files.items():
        req_ext = os.path.splitext(file_to_check)[1][1:].lower()
        if file_to_check in files_in_input_path:
            input_files[file_type] = os.path.join(input_path, file_to_check)
            logger.debug('%s input: %s', file_type, input_files[file_type])
        else:
            # If not present, let's try to explore the folder by extension
            file_counter = 0
            for file_input in files_in_input_path:
                file_ext = os.path.splitext(file_input)[1][1:].lower()
                if req_ext == file_ext:
                    file_counter += 1
                    selected_file = file_input

            # Since we are guessing the correct files, give an error if
            # multiple entries are possible.
            if file_counter == 1:
                input_files[file_type] = os.path.join(input_path,
                                                      selected_file)
                logger.warning(f'using {input_files[file_type]} '
                               + f'as "{file_type}" file')
            else:
                msg = f'Missing input file "{file_to_check}" '
                if file_counter > 1:
                    msg += f'and multiple files have extension ".{req_ext}"'
                raise ValueError(msg)

    # Check if the extra files are present
    if extra_files:
        input_files['extra_files'] = []
        for file_to_check in extra_files:
            if file_to_check in files_in_input_path:
                input_files['extra_files'].append(file_to_check)
            else:
                msg = f'Extra file {file_to_check} not present in {input_path}'
                logger.info(msg)

    return input_files
