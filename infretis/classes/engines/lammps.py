# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""This module defines the class for interfacing LAMMPS.

Important classes defined here
------------------------------

LAMMPSEngine (:py:class:`.LAMMPSEngine`)
    The class responsible for interfacing with LAMMPS.

    TO DO:
    * .dcd format for fast navigation and selection of snapshots
    * read timestep from .toml instead of lammps data
    * make order parameter calculation internal (or provide a .py file)
    * (Done) modify_velocities    
        * Should zero_momentum option (in .toml) be available in velocity generation?
    * run_lammps
        * create_lammps_md_input
            * system_to_lammps
            * add_to_lammps_input
"""
import logging
import os
import shlex
import subprocess
from time import sleep

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    PERIODIC_TABLE,
    box_matrix_to_list,
    box_vector_angles,
    convert_snapshot,
    look_for_input_files,
    read_xyz_file,
    write_xyz_trajectory,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


# Let us define the LAMMPS commands PyRETIS will make use of:

# 1) The commands related to the order parameter.
# This first command is to ensure that the order parameter is
# calculated also for cases where we do not run any MD steps. This
# can for instance be when we are generating velocities:
ORDER_HACK = """run 0 every 1 "print '{}' file {} screen no" # :-)"""
# The parameter is here the variables to print.
# The next command is outputting the order parameter:
ORDER_FIX = 'fix order_output all print {} "{}" append {} screen no'
# The parameters are: frequency, variables to print and the file name.

# 2) The commands related to thermodynamic output:
# Define a thermo style for LAMMPS:
THERMO_STYLE = 'thermo_style custom step temp press pe ke etotal'

# 3) The commands related to trajectory output and intput:
TRAJ_DUMP = 'dump {} all custom {} {} id {}'
# The three parameters are here: dump-id, frequency, output file name
# and the format for what to print. What to print depends on the number
# dimensions we consider and this is defined below:
TRAJ_OUT = {
    1: 'x vx ix',
    2: 'x y vx vy ix iy',
    3: 'x y z vx vy vz ix iy iz',
}
# Modify the format for the output by increasing the number of digits
# used for positions and velocities:
TRAJ_FMT = 'dump_modify {} format {} %23.16g'
# The parameter is here the dump-id and the column to apply it to.
# Add a command to undump the trajectory
TRAJ_UNDUMP = 'undump traj_pyretis'
# We also need to read in trajectories and/or snapshots:
READ_DUMP = 'read_dump {} {} {} box yes'
# The parameters are: the filename and the index in that file.
# The next command will read in LAMMPS restart files:
READ_RESTART = 'read_restart {}'
# The parameter is here the file name.

# 4) The commands for generating velocities:
GENERATE_VEL = 'velocity all create ${SET_TEMP} ${VEL_SEED} dist gaussian'
# The parameters are here the temperature and the random seed, which
# are actually set elsewhere in the LAMMPS input script.
# The next command sets the seed to use:
GENERATE_VEL_SEED = 'variable VEL_SEED equal {}'
# The parameter is here the seed to use.

# 5) The commands for stopping a simulation based on the order parameter
# and the given interface positions:
INTERFACE_RIGHT_FIX = 'fix op_stop_right all halt {} v_op_1 > {}'
# The parameters are here the frequency and the right interface.
INTERFACE_LEFT_FIX = 'fix op_stop_left all halt {} v_op_1 < {}'
# The parameters are here the frequency and the left interface.

# 6) The commands for rescaling energies if this is requested:
RESCALE_ENERGY = [
    'variable ke_rescale equal "v_SET_ENERGY - pe"',
    'variable alpha equal "sqrt(v_ke_rescale / ke)"',# 1) The commands related to the order parameter.
# This first command is to ensure that the order parameter is
# calculated also for cases where we do not run any MD steps. This
# can for instance be when we are generating velocities:
ORDER_HACK = """run 0 every 1 "print '{}' file {} screen no" # :-)"""
# The parameter is here the variables to print.
# The next command is outputting the order parameter:
ORDER_FIX = 'fix order_output all print {} "{}" append {} screen no'
# Here we need different commands, depending on the number
# of dimensions we are simulating. These are hard-coded below:
RESCALE_DIM = {
    3: [
        'variable vx atom "v_alpha * vx"',
        'variable vy atom "v_alpha * vy"',
        'variable vz atom "v_alpha * vz"',
        'velocity all set v_vx v_vy v_vz',
        ],
    2: [
        'variable vx atom "v_alpha * vx"',
        'variable vy atom "v_alpha * vy"',
        'velocity all set v_vx v_vy 0.0',
        ],
    1: [
        'variable vx atom "v_alpha * vx"',
        'velocity all set v_vx 0.0 0.0',
        ],
}

# 7) Commands for reversing velocities:
# First the command for reversing:
VEL_REV = 'variable        v{0} atom -v{0}'
# The argument is here to dimension to apply this to, e.g. "x".
# Then, the command to actually set the velocities.
VEL_SET = 'velocity        all set {} {} {}'
# The arguments are the velocities to set, either v_vx, v_vy, v_vz
# or "NULL".


def read_lammps_log(filename):
    """Read some info from a LAMMPS log file.

    In particular, this method is used to read the thermodynamic
    output from a simulation (e.g. potential and kinetic energies).

    Parameters
    ----------
    filename : string
        The path to the LAMMPS log file.

    Returns
    -------
    out : dict
        A dict containing the data we found in the file.

    """
    energy_keys = []
    energy_data = {}
    read_energy = False
    with open(filename, 'r', encoding='utf-8') as logfile:
        for lines in logfile:
            if lines.startswith('Step'):
                # Assume that this is the start of the thermo output.
                energy_keys = [i.strip() for i in lines.strip().split()]
                for key in energy_keys:
                    # Note: This will discard the previously read
                    # thermodynamic data. This is because we only want
                    # to read the final section of thermodynamic data,
                    # which, by construction of the LAMMPS input file,
                    # correspond to the output of the MD run.
                    energy_data[key] = []
                read_energy = True
                continue
            if lines.startswith('Loop time'):
                # Assume this marks the end of the thermo output.
                read_energy = False
                energy_keys = []
                continue
            if read_energy and energy_keys:
                # Assume that we are reading energies.
                try:
                    data = [float(i) for i in lines.strip().split()]
                except ValueError:
                    # Some text has snuck into the thermo output,
                    # ignore it:
                    continue
                for key, value in zip(energy_keys, data):
                    if key == 'Step':
                        energy_data[key].append(int(value))
                    else:
                        energy_data[key].append(value)
    for key, val in energy_data.items():
        energy_data[key] = np.array(val)
    out = {'energy': energy_data}
    return out


def read_lammps_input(filename):
    """Read a LAMMPS input file.

    This will read in a LAMMPS input file and can be used to read
    the value of particular settings, for instance the time step.

    Parameters
    ----------
    filename : string
        The path to the LAMMPS input file.

    Returns
    -------
    out : list of tuples
        The settings found in the LAMMPS input file. The tuples are
        on the form `(keyword, setting)`.

    """
    settings = []
    with open(filename, 'r', encoding='utf-8') as infile:
        for lines in infile:
            current_line, _, _ = lines.strip().partition('#')
            split = current_line.split()
            if split:
                keyword, setting = split[0], split[1:]
                settings.append((keyword, ' '.join(setting)))
    return settings


def add_to_lammps_input(infile, outfile, to_add):
    """Add PyRETIS specific settings to an input file for LAMMPS.

    This method will append settings needed by PyRETIS to an input
    file for LAMMPS.

    Parameters
    ----------
    infile : string
        Path to a file which contains the LAMMPS settings.
    outfile : string
        Path to the file we should create.
    to_add : list of strings
        The settings to add for PyRETIS.

    """
    if os.path.isfile(outfile):
        logger.debug(
            '"%s" exists and will be overwritten by PyRETIS.', outfile
        )
    with open(infile, 'r', encoding='utf-8') as indata:
        data = indata.read()
        with open(outfile, 'w', encoding='utf-8') as output:
            output.write(data)
            output.write('\n'.join(to_add))


def _add_order_fix(settings):
    """Add the fix for printing the order parameter for LAMMPS."""
    lines = []
    if 'order-fix' in settings:
        logger.debug('Adding fix for order parameter to LAMMPS input.')
        lines.append('# Order parameter fix:')
        for line in settings['order-fix']:
            lines.append(line)
    return lines


def _add_traj_dump(settings):
    """Add the dump commands for storing the LAMMPS trajectory."""
    lines = [
        '# PyRETIS trajectory settings:',
        '# PyRETIS requested the following trajectory output:',
    ]
    lines.append(
        TRAJ_DUMP.format(
            'traj_pyretis',
            settings['subcycles'],
            settings['traj'],
            TRAJ_OUT[settings['dimension']],
        )
    )
    # Format for positions:
    idx = 0
    for i in range(int(settings['dimension'])):
        idx = i + 2
        lines.append(TRAJ_FMT.format('traj_pyretis', idx))
    # Format for velocities:
    for i in range(int(settings['dimension'])):
        lines.append(TRAJ_FMT.format('traj_pyretis', idx + 1 + i))
    return lines


def _add_stopping_condition(settings):
    """Add the interface stopping condition to the LAMMPS input."""
    lines = []
    if 'interfaces' in settings:
        lines.append('# Stopping condition fix:')
        lines.append(
            INTERFACE_RIGHT_FIX.format(
                settings['subcycles'],
                settings['interfaces'][-1],
            )
        )
        if abs(settings['interfaces'][0]) != float('inf'):
            lines.append(
                INTERFACE_LEFT_FIX.format(
                    settings['subcycles'],
                    settings['interfaces'][0],
                )
            )
    return lines


def _add_generate_vel(settings):
    """Add generation of velocities to the LAMMPS input."""
    lines = []
    if 'generate_vel' in settings:
        seed = settings['generate_vel'].get('seed', 1)
        lines.append('# PyRETIS requested random velocities')
        lines.append(GENERATE_VEL_SEED.format(seed))
        lines.append(GENERATE_VEL)
        lines.append('run 0')
        if settings['generate_vel'].get('rescale', None):
            for i in RESCALE_ENERGY:
                lines.append(i)
            for i in RESCALE_DIM[settings['dimension']]:
                lines.append(i)
    return lines


def create_lammps_md_input(system, infile, outfile, settings):
    """Create MD input file for LAMMPS.

    We will here write to a new file, by appending text to the
    given template.

    Parameters
    ----------
    system : object like :py:class:`.System`
        The system contains the current particle state, and
        this determines the initial configuration and if we
        are to modify velocities in some way (e.g. reversing
        them before running or drawing new ones).
    infile : string
        Path to a file which contains the LAMMPS settings.
    settings : dict
        The settings we are going to use for creating the
        LAMMPS file.

    Returns
    -------
    out : list of strings
        The LAMMPS commands written to the output file.

    """
    new_lines = ['', '# Added by PyRETIS:', '']
    # Add reading of initial configuration:
    config = system_to_lammps(
        system,
        settings['reverse_velocities'],
        settings['dimension']
    )
    if config:
        new_lines += config
    # Update the thermo output from LAMMPS:
    new_lines.append('# PyRETIS requested the following thermo output:')
    new_lines.append(THERMO_STYLE)
    new_lines.append(f'thermo {settings["subcycles"]}')
    # Add randomization of velocities, if requested:
    new_lines += _add_generate_vel(settings)
    # Add storing of the LAMMPS trajectory:
    new_lines += _add_traj_dump(settings)
    # Add order parameter calculation:
    new_lines += _add_order_fix(settings)
    # Add stopping condition(s):
    new_lines += _add_stopping_condition(settings)
    # Add the number of steps to run:
    new_lines.append('# Requested steps by PyRETIS:')
    new_lines.append(f'run {settings["steps_subcycles"]}')
    # Add "clean-up" to the LAMMPS file:
    new_lines.append('# PyRETIS clean up:')
    new_lines.append(TRAJ_UNDUMP)
    add_to_lammps_input(infile, outfile, new_lines)
    return new_lines


def system_to_lammps(system, reverse_velocities, dimension):
    """Convert a LAMMPS system into LAMMPS commands for loading it.

    This method will convert a system created by LAMMPS into
    LAMMPS commands, so that LAMMPS can read the given snapshot
    and use it.

    Parameters
    ----------
    system : object like :py:class:`.System`
        The system we are converting.
    reverse_velocities : boolean
        True if the velocities in the system are to be reversed.
    dimension : integer
        The number of dimensions used in the LAMMPS simulation.

    Returns
    -------
    out : list of strings
        The commands needed by LAMMPS to read the configuration.

    """
    config = system.particles.get_pos()
    lammps = []
    if config[0].endswith('.restart') and config[1] == 0:
        # Assume that this is a LAMMPS restart file.
        lammps.append('# PyRETIS requested a LAMMPS restart file:')
        lammps.append(READ_RESTART.format(config[0]))
    elif config[0].endswith('.data') and config[1] == 0:
        # Assume that this is a LAMMPS data file AND that there
        # is already a line in the LAMMPS input file reading this
        # file. The motivation behind this is that we define this
        # file type to be the initial configuration and that data files
        # will never be used for something else.
        lammps.append(f'# PyRETIS requested the LAMMPS data file: {config[0]}')
    else:
        # Assume that this is a LAMMPS trajectory.
        lammps.append('# PyRETIS requested the following snapshot:')
        lammps.append(
            READ_DUMP.format(config[0], config[1], TRAJ_OUT[dimension])
        )
        lammps.append('# Reset time step:')
        lammps.append('reset_timestep 0')
    if reverse_velocities:
        # We add commands for reversing the velocities:
        lammps.append('# PyRETIS requested reversing of velocities:')
        vel = ['NULL', 'NULL', 'NULL']
        for i, dim in enumerate(['x', 'y', 'z'][:dimension]):
            lammps.append(VEL_REV.format(dim))
            vel[i] = f'v_v{dim}'
        lammps.append(VEL_SET.format(*vel))
        lammps.append('run 0')
    return lammps


class LAMMPSEngine(ExternalMDEngine):
    """
    A class for interfacing LAMMPS.

    Attributes
    ----------
    lmp : string
        The command for executing LAMMPS
    input_path : string
        The directory where the input files are stored.
    input_files : dict of strings
        The names of the input files.

    """

    needs_order = False

    def __init__(self, 
                lmp,
                input_path,
                timestep,
                subcycles,
                extra_files=None,
                exe_path=os.path.abspath("."),
                seed=0,
                sleep=0.1):
        """Set up the LAMMPS engine.

        Parameters
        ----------
        lmp : string
            The LAMMPS executable.
        input_path : string
            The absolute path to where the input files are stored.
        timestep : float
            The time step used in the LAMMPS simulation.
        subcycles : integer
            The frequency of output of data by LAMMPS.
        extra_files : list of strings, optional
            Additional files needed to run the LAMMPS simulation.
        seed : integer, optional
            A seed for the random number generator.
        extra_files : list
            List of extra files which may be required to run LAMMPS.
        exe_path: string, optional
            The path on which the engine is executed
    
        """
        # Note: We set time step to zero for now. This will be updated
        # to the correct value when the timestep has been read from
        # the input script:
        super().__init__('LAMMPS Engine', timestep, subcycles)
        self.lmp = lmp
        self.input_path = os.path.abspath(input_path)
        # Define the files we require:
        input_files = {
            'conf': 'system.data',
            'template': 'lammps.in',  # The template file for LAMMPS.
            'order': 'order.in',  # Definition of the order parameter.
        }
        # The user may choose to structure the LAMMPS input into
        # several files. This is something we will not try to figure
        # out, and we assume that the user knows what he/she is doing.
        # We assume that these extra files will also be present and
        # stored in the given input path. Now, we look for such files:
        self.input_files = look_for_input_files(
            self.input_path,
            input_files,
        )
        self.run_files = {
            'conf': self.input_files['conf'],
            'order': self.input_files['order'],
        }
        if extra_files is not None:
            extra = look_for_input_files(
                self.input_path,
                {f'file-{i}': val for i, val in enumerate(extra_files)}
            )
            for key, val in extra.items():
                self.run_files[key] = val
        # Read the LAMMPS input file:
        self.settings = {
            'dimension': 3,
            'timestep': 0.0
        }
        for (key, val) in read_lammps_input(self.input_files['template']):
            self.settings[key] = val
        # Also set the timestep explicitly in case it is needed:
        self.timestep = self.settings['timestep']

    def _make_order_fix(self, ordername):
        """Create a LAMMPS fix for the order parameter.

        Note that we here make LAMMPS also output for step zero.

        """
        order_settings = read_lammps_input(
            os.path.join(self.input_path, self.input_files['order'])
        )
        ops = ['$(step)']
        for key, val in order_settings:
            if key == 'variable':
                ops.append(f'${{{val.strip().split()[0]}}}')
        txt = [
            ORDER_HACK.format(' '.join(ops), ordername),
            ORDER_FIX.format(self.subcycles, ' '.join(ops), ordername),
        ]
        return txt

    def add_input_files(self, dirname):
        """Add required input files to a given directory.

        Parameters
        ----------
        dirname : string
            The path to the directory where we want to add the files.

        """
        for _, filepathi in self.run_files.items():
            basename = os.path.basename(filepathi)
            dest = os.path.join(dirname, basename)
            if not os.path.isfile(dest):
                logger.debug('Adding input file "%s" to "%s"',
                             basename, dirname)
                self._copyfile(filepathi, dest)

    def read_order_parameters(self, ordername):
        """Read order parameters as calculated by LAMMPS.

        We assume here that these can be found in the current execute
        directory in a file named `ordername`, and further that they
        contain the step number in the first column, followed by the
        order parameters in following columns.

        """
        orderfile = os.path.join(self.exe_dir, ordername)
        order = np.loadtxt(orderfile, ndmin=2)
        # Removing the first column, as this is not an order parameter,
        # but the step number in the simulation. The usage of deleting
        # along axis=1 is the motivation for forcing ndim=2 above.
        order = np.delete(order, 0, 1)
        return order

    def read_energies(self, name):
        """Read energies obtained in a LAMMPS run.

        Here, we assume that the energies can be found in a log file
        in the current execute directory.

        """
        logfile = os.path.join(self.exe_dir, f'{name}.log')
        data = read_lammps_log(logfile)
        return data['energy']

    def _propagate_from(self, name, path, system, ens_set, msg_file,
                        reverse=False):  # pragma: no cover
        """
        Propagate with LAMMPS from the current system configuration.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble: dict
            It contains:

            * `system`: object like :py:class:`.System`
              The system object gives the initial state.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces`: list of floats
              These interfaces define the stopping criterion.

        msg_file : object like :py:class:`.FileIO`
            An object we use for writing out messages that are useful
            for inspecting the status of the current propagation.
        reverse : boolean, optional
            If True, the system will be propagated backward in time.

        Returns
        -------
        success : boolean
            This is True if we generated an acceptable path.
        status : string
            A text description of the current status of the propagation.

        """
        return

    def step(self, system, name):  # pragma: no cover
        """Perform a single step with LAMMPS.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.
        name : string
            To name the output files from the LAMMPS step.

        Returns
        -------
        out : string
            The name of the output configuration, obtained after
            completing the step.

        """
        return

    def _extract_frame(self, traj_file, idx, out_file):  # pragma: no cover
        return

    @staticmethod
    def _read_configuration(filename):  # pragma: no cover  
        return

    def set_mdrun(self, config, md_items):
        """Copy  from cp2k.py"""
        self.exe_dir = md_items["w_folder"]
        self.rgen = md_items["picked"][md_items["ens_nums"][0]]["ens"]["rgen"]

    def _reverse_velocities(self, filename, outfile):  # pragma: no cover
        """Reverse the velocities for a given configuration."""
        return

    def modify_velocities(self, system, vel_settings=None):
        #ensemble -> system
        """Modify velocities."""
        dek = None
  
        kin_old = system.ekin
        rgen = self.rgen
        kin_new = None
        if vel_settings.get('aimless', False):
            logger.debug('Modifying velocities with LAMMPS.')
            name = 'generate_vel'
            if rgen:
                seed = rgen.integers(1, 2147483647 - 1)
                # Note: `integers` is inclusive for the lower bound
                # and exclusive for the upper bound. 
                # We don't like to have > 2**31 - 1 as a
                # possibility. For some reason, LAMMPS might also
                # die/hang for a seed equal to 2**31 - 1, so this is
                # why 1 is subtracted here, resulting in the range:
                # [1, 2147483646]
            else:
                seed = 1
            settings = {
                'steps_subcycles': 0,
                'reverse_velocities': False,
                'generate_vel': {
                    'seed': seed,
                    'rescale': vel_settings.get('rescale_energy',
                                                vel_settings.get('rescale'))
                },
            }
            self.run_lammps(system, settings, name) # here
            kin_new = system.ekin
            if kin_old is None:
                dek = float('inf')
            else:
                dek = kin_new - kin_old
            return dek, kin_new
        raise ValueError(
            'LAMMPS only support the aimless velocity modification.'
        )

    def run_lammps(self, system, settings, name):
        """Execute LAMMPS.

        This method will handle input files, run LAMMPS and
        return some data after the run.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system defines the initial state we are using.
        settings : dict
            This dict contains settings for creating the LAMMPS
            input file.
        name : string
            This string is used as a base name for some of the
            LAMMPS input scripts and output files.

        Returns
        -------
        out[0] : numpy.array
            The order parameters obtained during the run.
        out[1] : dict of numpy.arrays
            The energies, each dictionary key corresponds to a
            specific energy term.
        out[2] : string
            The name of the trajectory created by LAMMPS in this run.

        """
        logger.debug('Adding input files for LAMMPS.')
        self.add_input_files(self.exe_dir)
        # Name the trajectory output:
        settings['traj'] = os.path.join(self.exe_dir, f'{name}.lammpstrj')
        settings['order'] = f'order_{name}.txt'
        # Add the subcycles & timestep settings:
        settings['subcycles'] = self.subcycles
        settings['timestep'] = self.timestep
        settings['dimension'] = int(self.settings['dimension'])
        # Add the order-fix:
        settings['order-fix'] = self._make_order_fix(settings['order'])
        # Name the input script for LAMMPS:
        input_file = f'{name}.in'
        script_file = os.path.join(self.exe_dir, input_file)
        # Create the input file we will use for running LAMMPS:
        create_lammps_md_input(
            system,
            self.input_files['template'],
            script_file,
            settings,
        )
        cmd = shlex.split(self.lmp)
        cmd_arguments = [
            '-in', input_file,
            '-l', f'{name}.log',
            '-screen', f'{name}.screen'
        ]
        cmd.extend(cmd_arguments)
        logger.debug('Executing LAMMPS "%s".', name)
        self.execute_command(cmd, cwd=self.exe_dir)
        logger.debug('Reading LAMMPS order parameters & energies after run.')
        order = self.read_order_parameters(settings['order'])
        energy = self.read_energies(name)
        # Set the state of the system to the last point:
        system.order = order[-1]
        system.ekin = energy['KinEng'][-1]
        system.vpot = energy['PotEng'][-1]
        system.set_pos(
            (settings['traj'], settings['steps_subcycles'])
        )
        return order, energy, settings['traj']

    def integrate(self, system, steps):  # order_function=None, thermo='full'):
        """Propagate several integration steps with LAMMPS."""
        logger.debug('Integrating with LAMMPS.')
        # Add settings for this run:
        settings = {
            'steps_subcycles': steps * self.subcycles,
            'reverse_velocities': system.particles.get_vel(),
        }
        # Execute LAMMPS:
        self.run_lammps(system, settings, 'pyretis_md')

    def propagate(self, path, ensemble, reverse=False):
        """
        Propagate the equations of motion with the external code.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
            We are here not returning a new path - this since we want
            to delegate the creation of the path to the method
            that is running `propagate`.
        ensemble: dict
            It contains:

            * `system`: object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces`: list of floats
              These interfaces define the stopping criterion.

        reverse : boolean, optional
            If True, the system will be propagated backward in time.

        Returns
        -------
        success : boolean
            This is True if we generated an acceptable path.
        status : string
            A text description of the current status of the propagation.

        """
        initial_state = ensemble['system']
        interfaces = ensemble['interfaces']
        logger.debug('Running propagate with: "%s"', self.description)
        prefix = str(counter())
        if reverse:
            logger.debug('Running backward in time.')
            name = prefix + '_trajB'
        else:
            logger.debug('Running forward in time.')
            name = prefix + '_trajF'
        logger.debug('Trajectory name: "%s"', name)
        # Also create a message file for inspecting progress:
        msg_file_name = os.path.join(self.exe_dir, f'msg-{name}.txt')
        logger.debug('Writing propagation progress to: %s', msg_file_name)
        msg_file = FileIO(msg_file_name, 'w', None, backup=False)
        msg_file.open()
        msg_file.write(f'# Preparing propagation with {self.description}')
        msg_file.write(f'# Trajectory label: {name}')

        system = initial_state.copy()
        logger.debug('Initial state: %s', system)
        # For storing LAMMPS settings:
        settings = {
            'steps_subcycles': path.maxlen * self.subcycles,
            'interfaces': interfaces,
            'reverse_velocities': reverse != system.particles.vel_rev,
        }

        if settings['reverse_velocities']:
            logger.debug('Reversing velocities in initial config.')
        system.particles.set_vel(reverse)

        # Propagate from the current point:
        msg_file.write(f'# Interfaces: {interfaces}')
        msg_file.write(f'# Running LAMMPS in folder: {self.exe_dir}')
        order, energy, traj = self.run_lammps(system, settings, name)
        msg_file.write('# Done running LAMMPS, adding points to path.')
        for i, orderi in enumerate(order):
            phase_point = system.copy()
            phase_point.order = orderi
            phase_point.particles.ekin = energy['KinEng'][i]
            phase_point.particles.vpot = energy['PotEng'][i]
            phase_point.particles.set_pos((traj, i * self.subcycles))
            phase_point.particles.set_vel(reverse)
            status, success, stop, _ = self.add_to_path(
                path,
                phase_point,
                interfaces[0],
                interfaces[-1]
            )
            if stop:
                break
        msg_file.write('# Propagation with LAMMPS all done.')
        msg_file.close()
        return success, status

    def dump_phasepoint(self, phasepoint, deffnm='conf'):
        """Dump a phase point to a new file.

        Note
        ----
        We do not reverse the velocities here.

        """
        settings = {'steps_subcycles': 0, 'reverse_velocities': False}
        self.run_lammps(phasepoint, settings, deffnm)

    def calculate_order(self, ensemble):
        """Return the last seen order parameter."""
        return ensemble['system'].order
