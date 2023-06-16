# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""A CP2K external MD integrator interface.

This module defines a class for using CP2K as an external engine.

Important classes defined here
------------------------------

CP2KEngine (:py:class:`.CP2KEngine`)
    A class responsible for interfacing CP2K.
"""
import logging
import code
import os
import re
import shlex
import numpy as np
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.rgen import create_random_generator
from pyretis.inout.settings import look_for_input_files
from pyretis.core.units import CONVERT, CONSTANTS
from pyretis.setup.createsystem import PERIODIC_TABLE


from pyretis.inout.formats.xyz import (
    read_xyz_file,
    write_xyz_trajectory,
    convert_snapshot
)
from pyretis.inout.formats.cp2k import (
    update_cp2k_input,
    read_cp2k_input,
    set_parents,
    read_cp2k_restart,
    read_cp2k_box,
    read_cp2k_energy,
)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


OUTPUT_FILES = {
    'energy': '{}-1.ener',
    'restart': '{}-1.restart',
    'pos': '{}-pos-1.xyz',
    'vel': '{}-vel-1.xyz',
    'wfn': '{}-RESTART.wfn',
    'wfn-bak': '{}-RESTART.wfn.bak-'
}


REGEXP_BACKUP = re.compile(r'\.bak-\d$')

def guess_particle_mass(particle_no, particle_type, unit):
    """Guess a particle mass from it's type.

    Parameters
    ----------
    particle_no : integer
        Just used to identify the particle number.
    particle_type : string
        Used to identify the particle.
    unit : string
        The system of units. This is used in case we try to get the
        mass from the periodic table where the units are in `g/mol`.

    """
    logger.info(('Mass not specified for particle no. %i\n'
                 'Will guess from particle type "%s"'), particle_no,
                particle_type)
    mass = PERIODIC_TABLE.get(particle_type, None)
    if mass is None:
        particle_mass = 1.0
        logger.info(('-> Could not find mass. '
                     'Assuming %f (internal units)'), particle_mass)
    else:
        particle_mass = CONVERT['mass']['g/mol', unit] * mass
        logger.info(('-> Using a mass of %f g/mol '
                     '(%f in internal units)'), mass, particle_mass)
    return particle_mass


def kinetic_energy(vel, mass):
    """Obtain the kinetic energy for given velocities and masses.

    Parameters
    ----------
    vel : numpy.array
        The velocities
    mass : numpy.array
        The masses. This is assumed to be a column vector.

    Returns
    -------
    out[0] : float
        The kinetic energy
    out[1] : numpy.array
        The kinetic energy tensor.

    """
    mom = vel * mass
    if len(mass) == 1:
        kin = 0.5 * np.outer(mom, vel)
    else:
        kin = 0.5 * np.einsum('ij,ik->jk', mom, vel)
    return kin.trace(), kin

def reset_momentum(vel, mass):
    """Set the linear momentum of all particles to zero.
       Note that velocities are modified in place, but also
       returned.

    Parameters
    ----------
    vel : numpy.array
        The velocities of the particles in system. 
    mass : numpy.array
        The masses of the particles in the system. 

    Returns
    -------
    out : numpy.array
        Returns the modified velocities of the particles.

    """
    # avoid creating an extra dimension by indexing array with None

    mom = np.sum(vel * mass,  axis=0)
    print("mom unscaled",np.sum(vel))
    vel -= (mom / mass.sum())
    mom = np.sum(vel * mass, axis=0)
    print("mom scaled",np.sum(vel))
    return vel

def write_for_step_vel(infile, outfile, timestep, subcycles, posfile, vel,
                       name='md_step', print_freq=None):
    """Create input file for a single step.

    Note, the single step actually consists of a number of subcycles.
    But from PyRETIS' point of view, this is a single step.
    Further, we here assume that we start from a given xyz file and
    we also explicitly give the velocities here.

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    timestep : float
        The time-step to use for the simulation.
    subcycles : integer
        The number of sub-cycles to perform.
    posfile : string
        The (base)name for the input file to read positions from.
    vel : numpy.array
        The velocities to set in the input.
    name : string, optional
        A name for the CP2K project.
    print_freq : integer, optional
        How often we should print to the trajectory file.

    """
    if print_freq is None:
        print_freq = subcycles
    to_update = {
        'GLOBAL': {
            'data': [f'PROJECT {name}',
                     'RUN_TYPE MD',
                     'PRINT_LEVEL LOW'],
            'replace': True,
        },
        'MOTION->MD':  {
            'data': {'STEPS': subcycles,
                     'TIMESTEP': timestep}
        },
        'MOTION->PRINT->RESTART': {
            'data': ['BACKUP_COPIES 0'],
            'replace': True,
        },
        'MOTION->PRINT->RESTART->EACH': {
            'data': {'MD': print_freq}
        },
        'MOTION->PRINT->VELOCITIES->EACH': {
            'data': {'MD': print_freq}
        },
        'MOTION->PRINT->TRAJECTORY->EACH': {
            'data': {'MD': print_freq}
        },
        'FORCE_EVAL->SUBSYS->TOPOLOGY': {
            'data': {'COORD_FILE_NAME': posfile,
                     'COORD_FILE_FORMAT': 'xyz'}
        },
        'FORCE_EVAL->SUBSYS->VELOCITY': {
            'data': [],
            'replace': True,
        },
        #'FORCE_EVAL->DFT->SCF->PRINT->RESTART': {
        #    'data': ['BACKUP_COPIES 0'],
        #    'replace': True,
        #},
    }
    for veli in vel:
        to_update['FORCE_EVAL->SUBSYS->VELOCITY']['data'].append(
            f'{veli[0]} {veli[1]} {veli[2]}'
        )
    remove = [
        'EXT_RESTART',
        'FORCE_EVAL->SUBSYS->COORD'
    ]
    update_cp2k_input(infile, outfile, update=to_update, remove=remove)


def write_for_integrate(infile, outfile, timestep, subcycles, posfile,
                        name='md_step', print_freq=None):
    """Create input file for a single step for the integrate method.

    Here, we do minimal changes and just set the time step and subcycles
    and the starting configuration.

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    timestep : float
        The time-step to use for the simulation.
    subcycles : integer
        The number of sub-cycles to perform.
    posfile : string
        The (base)name for the input file to read positions from.
    name : string, optional
        A name for the CP2K project.
    print_freq : integer, optional
        How often we should print to the trajectory file.

    """
    if print_freq is None:
        print_freq = subcycles
    to_update = {
        'GLOBAL': {
            'data': [f'PROJECT {name}',
                     'RUN_TYPE MD',
                     'PRINT_LEVEL LOW'],
            'replace': True,
        },
        'MOTION->MD':  {
            'data': {'STEPS': subcycles,
                     'TIMESTEP': timestep}
        },
        'MOTION->PRINT->RESTART': {
            'data': ['BACKUP_COPIES 0'],
            'replace': True,
        },
        'MOTION->PRINT->RESTART->EACH': {
            'data': {'MD': print_freq}
        },
        'MOTION->PRINT->VELOCITIES->EACH': {
            'data': {'MD': print_freq}
        },
        'MOTION->PRINT->TRAJECTORY->EACH': {
            'data': {'MD': print_freq}
        },
        'FORCE_EVAL->SUBSYS->TOPOLOGY': {
            'data': {'COORD_FILE_NAME': posfile,
                     'COORD_FILE_FORMAT': 'xyz'}
        },
        #'FORCE_EVAL->DFT->SCF->PRINT->RESTART': {
        #    'data': ['BACKUP_COPIES 0'],
        #    'replace': True,
       # },
    }
    remove = [
        'EXT_RESTART',
        'FORCE_EVAL->SUBSYS->COORD'
    ]
    update_cp2k_input(infile, outfile, update=to_update, remove=remove)


def write_for_continue(infile, outfile, timestep, subcycles,
                       name='md_continue'):
    """
    Create input file for a single step.

    Note, the single step actually consists of a number of subcycles.
    But from PyRETIS' point of view, this is a single step.
    Here, we make use of restart files named ``previous.restart``
    and ``previous.wfn`` to continue a run.

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    timestep : float
        The time-step to use for the simulation.
    subcycles : integer
        The number of sub-cycles to perform.
    name : string, optional
        A name for the CP2K project.

    """
    to_update = {
        'GLOBAL': {
            'data': [f'PROJECT {name}',
                     'RUN_TYPE MD',
                     'PRINT_LEVEL LOW'],
            'replace': True,
        },
        'MOTION->MD':  {
            'data': {'STEPS': subcycles,
                     'TIMESTEP': timestep}
        },
        'MOTION->PRINT->RESTART': {
            'data': ['BACKUP_COPIES 0'],
            'replace': True,
        },
        'MOTION->PRINT->RESTART->EACH': {
            'data': {'MD': subcycles}
        },
        'MOTION->PRINT->VELOCITIES->EACH': {
            'data': {'MD': subcycles}
        },
        'MOTION->PRINT->TRAJECTORY->EACH': {
            'data': {'MD': subcycles}
        },
        'EXT_RESTART': {
            'data': ['RESTART_VEL',
                     'RESTART_POS',
                     'RESTART_FILE_NAME previous.restart'],
            'replace': True
        },
        #'FORCE_EVAL->DFT': {
        #    'data': {'WFN_RESTART_FILE_NAME': 'previous.wfn'},
        #},
        #'FORCE_EVAL->DFT->SCF->PRINT->RESTART': {
        #    'data': ['BACKUP_COPIES 0'],
        #    'replace': True,
        #},
    }
    remove = [
        #'FORCE_EVAL->SUBSYS->TOPOLOGY',
        'FORCE_EVAL->SUBSYS->VELOCITY',
        'FORCE_EVAL->SUBSYS->COORD'
        'FORCE_EVAL->DFT->RESTART_FILE_NAME',
    ]
    update_cp2k_input(infile, outfile, update=to_update, remove=remove)


def write_for_genvel(infile, outfile, posfile, seed,
                     name='genvel'):  # pragma: no cover
    """Create input file for velocity generation.

    2022.05.25: Note: This function is no longer in use
    as we now generate velocities internally. However we keep
    this function in case of future need.

    Parameters
    ----------
    infile : string
        The input template to use.
    outfile : string
        The file to create.
    posfile : string
        The (base)name for the input file to read positions from.
    seed : integer
        A seed for generating velocities.
    name : string, optional
        A name for the CP2K project.

    """
    to_update = {
        'GLOBAL': {
            'data': [f'PROJECT {name}',
                     f'SEED {seed}',
                     'RUN_TYPE MD',
                     'PRINT_LEVEL LOW'],
            'replace': True,
        },
       # 'FORCE_EVAL->DFT->SCF': {
       #     'data': {'SCF_GUESS': 'ATOMIC'}
       # },
        'MOTION->MD':  {
            'data': {'STEPS': 1,
                     'TIMESTEP': 0}
        },
        'MOTION->PRINT->RESTART': {
            'data': ['BACKUP_COPIES 0'],
            'replace': True,
        },
        'MOTION->PRINT->RESTART->EACH': {
            'data': {'MD': 1}
        },
        'MOTION->PRINT->VELOCITIES->EACH': {
            'data': {'MD': 1}
        },
        'MOTION->PRINT->TRAJECTORY->EACH': {
            'data': {'MD': 1}
        },
        'FORCE_EVAL->SUBSYS->TOPOLOGY': {
            'data': {'COORD_FILE_NAME': posfile,
                     'COORD_FILE_FORMAT': 'xyz'}
        },
       # 'FORCE_EVAL->DFT->SCF->PRINT->RESTART': {
       #     'data': ['BACKUP_COPIES 0'],
       #     'replace': True,
       # },
    }
    remove = [
        'EXT_RESTART',
        'FORCE_EVAL->SUBSYS->VELOCITY',
        'FORCE_EVAL->DFT->RESTART_FILE_NAME',
    ]
    update_cp2k_input(infile, outfile, update=to_update, remove=remove)


class CP2KEngine(EngineBase):
    """
    A class for interfacing CP2K.

    This class defines the interface to CP2K.

    Attributes
    ----------
    cp2k : string
        The command for executing CP2K.
    input_path : string
        The directory where the input files are stored.
    timestep : float
        The time step used in the CP2K MD simulation.
    subcycles : integer
        The number of steps each CP2K run is composed of.
    rgen : object like :py:class:`.RandomGenerator`
        An object we use to set seeds for velocity generation.
    extra_files : list
        List of extra files which may be required to run CP2K.

    """

    def __init__(self, cp2k, input_path, timestep, subcycles,
                 extra_files=None, exe_path=os.path.abspath('.'),  seed=0):
        """Set up the CP2K engine.

        Parameters
        ----------
        cp2k : string
            The CP2K executable.
        input_path : string
            The path to where the input files are stored.
        timestep : float
            The time step used in the CP2K simulation.
        subcycles : integer
            The number of steps each CP2K run is composed of.
        extra_files : list
            List of extra files which may be required to run CP2K.
        seed : integer, optional
            A seed for the random number generator.
        extra_files : list
            List of extra files which may be required to run CP2K.
        exe_path: string, optional
            The path on which the engine is executed

        """
        super().__init__('CP2K external engine', timestep,
                         subcycles)
        #self.rgen = create_random_generator({'seed': seed})
        self.ext = 'xyz'
        self.cp2k = shlex.split(cp2k)
        logger.info('Command for execution of CP2K: %s', ' '.join(self.cp2k))
        # Store input path:
        self.input_path = os.path.join(exe_path, input_path)
        # Set the defaults input files:
        default_files = {
            'conf': f'initial.{self.ext}',
            'template': 'cp2k.inp',
        }
        # Check the presence of the defaults input files or, if absent,
        # try to find then by extension.
        self.input_files = look_for_input_files(self.input_path, default_files)
        
        # add mass, temperature and unit information to engine 
        # which is needed for velocity modification 
        pos, vel, box, atoms = self._read_configuration(self.input_files['conf'])
        mass = [guess_particle_mass(i, name, 'g/mol') for i,name in enumerate(atoms)]
        self.mass = np.reshape(mass,(len(mass),1))*1822.8884858012982 # conversion g/mol -> cp2k
        
        # read temperature from cp2k input, defaults to 300
        self.temperature=None
        section = 'MOTION->MD'
        nodes = read_cp2k_input(self.input_files['template'])
        node_ref = set_parents(nodes)
        md_settings = node_ref[section]
        for data in md_settings.data:
            if 'temperature' in data.lower():
                self.temperature = float(data.split()[-1])
        if self.temperature==None:
            logger.info(f'No temperature specified in cp2k input. Using 300 K.')
            self.temperature=300.0

        self.beta = 1/(self.temperature * CONSTANTS['kB']['cp2k'])

        # todo, these info can be processed by look_for_input_files using
        # the extra_files option.
        self.extra_files = []
        if extra_files is not None:
            for key in extra_files:
                fname = os.path.join(self.input_path, key)
                if not os.path.isfile(fname):
                    logger.critical('Extra CP2K input file "%s" not found!',
                                    fname)
                else:
                    self.extra_files.append(fname)


    def run_cp2k(self, input_file, proj_name):
        """
        Run the CP2K executable.

        Returns
        -------
        out : dict
            The files created by the run.

        """
        cmd = self.cp2k + ['-i', input_file]
        logger.debug('Executing CP2K %s: %s', proj_name, input_file)
        self.execute_command(cmd, cwd=self.exe_dir, inputs=None)
        out = {}
        for key, name in OUTPUT_FILES.items():
            out[key] = os.path.join(self.exe_dir, name.format(proj_name))
        return out

    def _extract_frame(self, traj_file, idx, out_file):
        """
        Extract a frame from a trajectory file.

        This method is used by `self.dump_config` when we are
        dumping from a trajectory file. It is not used if we are
        dumping from a single config file.

        Parameters
        ----------
        traj_file : string
            The trajectory file to dump from.
        idx : integer
            The frame number we look for.
        out_file : string
            The file to dump to.

        """
        for i, snapshot in enumerate(read_xyz_file(traj_file)):
            if i == idx:
                box, xyz, vel, names = convert_snapshot(snapshot)
                if os.path.isfile(out_file):
                    logger.debug('CP2K will overwrite %s', out_file)
                write_xyz_trajectory(out_file, xyz, vel, names, box,
                                     append=False)
                return
        logger.error('CP2K could not extract index %i from %s!',
                     idx, traj_file)

    def _propagate_from(self, name, path, system, ens_set, msg_file, reverse=False):
        print(ens_set)
        """
        Propagate with CP2K from the current system configuration.

        Here, we assume that this method is called after the propagate()
        has been called in the parent. The parent is then responsible
        for reversing the velocities and also for setting the initial
        state of the system.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble : dict
            It contains the simulations info:

            * `system` : object like :py:class:`.System`
              The system to act on.
            * `engine` : object like :py:class:`.EngineBase`
              This is the integrator that is used to propagate the system
              in time.
            * `order_function` : object like :py:class:`.OrderParameter`
              The class used for calculating the order parameters.
            * `interfaces` : list of floats
              These defines the interfaces for which we will check the
              crossing(s).

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
            A text description of the current status of the
            propagation.

        """
        status = f'propagating with CP2K (reverse = {reverse})'
        interfaces = ens_set['interfaces']
        logger.debug(status)
        success = False
        left, _, right = interfaces
        logger.debug('Adding input files for CP2K')
        # First, copy the required input files:
        self.add_input_files(self.exe_dir)
        # Get positions and velocities from the input file.
        initial_conf = system.config[0]
        box, xyz, vel, atoms = self._read_configuration(initial_conf)
        if box is None:
            box, _ = read_cp2k_box(self.input_files['template'])
        # Add CP2K input for a single step:
        step_input = os.path.join(self.exe_dir, 'step.inp')
        write_for_step_vel(self.input_files['template'], step_input,
                           self.timestep, self.subcycles,
                           os.path.basename(initial_conf),
                           vel, name=name)
        # And create the input file for continuing:
        continue_input = os.path.join(self.exe_dir, 'continue.inp')
        write_for_continue(self.input_files['template'], continue_input,
                           self.timestep, self.subcycles, name=name)
        # Get the order parameter before the run:
        order = self.calculate_order(system, xyz=xyz, vel=vel, box=box)
        traj_file = os.path.join(self.exe_dir, f'{name}.{self.ext}')
        # Create a message file with some info about this run:
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        msg_file.write(f'# Trajectory file is: {traj_file}')
        # Run the first step:
        msg_file.write('# Running first CP2k step.')
        out_files = self.run_cp2k('step.inp', name)
        restart_file = os.path.join(self.exe_dir, out_files['restart'])
        prestart_file = os.path.join(self.exe_dir, 'previous.restart')
        wave_file = os.path.join(self.exe_dir, out_files['wfn'])
        pwave_file = os.path.join(self.exe_dir, 'previous.wfn')

        # Note: Order is calculated at the END of each iteration!
        i = 0
        # Write the config so we have a non-empty file:
        write_xyz_trajectory(traj_file, xyz, vel, atoms, box, step=i,
                             append=False)
        msg_file.write('# Running main CP2k propagation loop.')
        msg_file.write('# Step order parameter cv1 cv2 ...')
        for i in range(path.maxlen):
            msg_file.write(f'{i} {" ".join([str(j) for j in order])}')
            snapshot = {'order': order, 'config': (traj_file, i),
                        'vel_rev': reverse}
            phase_point = self.snapshot_to_system(system, snapshot)
            status, success, stop, add = self.add_to_path(path, phase_point,
                                                          left, right)
            if add and i > 0:
                # Write the previous configuration:
                write_xyz_trajectory(traj_file, xyz, vel, atoms, box,
                                     step=i)
            if stop:
                logger.debug('CP2K propagation ended at %i. Reason: %s',
                             i, status)
                break
            if i == 0:
                pass
            elif i > 0:
                self._movefile(restart_file, prestart_file)
                #self._movefile(wave_file, pwave_file)
                if i < path.maxlen - 1:
                    out_files = self.run_cp2k('continue.inp', name)
            self._remove_files(self.exe_dir,
                               self._find_backup_files(self.exe_dir))
            # Read config after the step
            if i < path.maxlen - 1:
                atoms, xyz, vel, box, _ = read_cp2k_restart(restart_file)
                order = self.calculate_order(system,
                                             xyz=xyz, vel=vel, box=box)
        msg_file.write('# Propagation done.')
        energy_file = out_files['energy']
        msg_file.write(f'# Reading energies from: {energy_file}')
        energy = read_cp2k_energy(energy_file)
        end = (i + 1) * self.subcycles
        ekin = energy.get('ekin', [])
        vpot = energy.get('vpot', [])
        path.update_energies(ekin[:end:self.subcycles],
                             vpot[:end:self.subcycles])
        for _, files in out_files.items():
            self._removefile(files)
        self._removefile(prestart_file)
        self._removefile(pwave_file)
        self._removefile(continue_input)
        self._removefile(step_input)
        return success, status

    def step(self, system, name):
        raise NotImplementedError("Surprise, step not implemented!")

    def add_input_files(self, dirname):
        """Add required input files to a given directory.

        Parameters
        ----------
        dirname : string
            The full path to where we want to add the files.

        """
        for files in self.extra_files:
            basename = os.path.basename(files)
            dest = os.path.join(dirname, basename)
            if not os.path.isfile(dest):
                logger.debug('Adding input file "%s" to "%s"',
                             basename, dirname)
                self._copyfile(files, dest)

    @staticmethod
    def _find_backup_files(dirname):
        """Return backup-files in the given directory."""
        out = []
        for entry in os.scandir(dirname):
            if entry.is_file():
                match = REGEXP_BACKUP.search(entry.name)
                if match is not None:
                    out.append(entry.name)
        return out

    @staticmethod
    def _read_configuration(filename):
        """
        Read CP2K output configuration.

        This method is used when we calculate the order parameter.

        Parameters
        ----------
        filename : string
            The file to read the configuration from.

        Returns
        -------
        box : numpy.array
            The box dimensions if we manage to read it.
        xyz : numpy.array
            The positions.
        vel : numpy.array
            The velocities.
        names : list of strings
            The atom names found in the file.

        """
        xyz, vel, box, names = None, None, None, None
        for snapshot in read_xyz_file(filename):
            box, xyz, vel, names = convert_snapshot(snapshot)
            break  # Stop after the first snapshot.
        return box, xyz, vel, names

    def set_mdrun(self, config, md_items):
        """Remove or rename?"""
        self.exe_dir = md_items['w_folder']
        #self.rgen = md_items['picked']['tis_set']['rgen']
        self.rgen = md_items['picked'][md_items['ens_nums'][0]]['ens']['rgen']

    def _reverse_velocities(self, filename, outfile):
        """Reverse velocity in a given snapshot.

        Parameters
        ----------
        filename : string
            The configuration to reverse velocities in.
        outfile : string
            The output file for storing the configuration with
            reversed velocities.

        """
        box, xyz, vel, names = self._read_configuration(filename)
        print("read",vel[0])
        write_xyz_trajectory(outfile, xyz, -1.0*vel, names, box, append=False)
        box, xyz, vel, names = self._read_configuration(outfile)
        print("write",vel[0])



    def modify_velocities(self, system, vel_settings=None):
        """
        Modfy the velocities of all particles. Note that cp2k by default
        removes the center of mass motion, thus, we need to rescale the 
        momentum to zero by default.

        """
        rgen = self.rgen
        mass = self.mass
        beta  = self.beta
        rescale = vel_settings.get('rescale_energy',
    	                               vel_settings.get('rescale'))
        pos = self.dump_frame(system)
        box, xyz, vel, atoms = self._read_configuration(pos)
        #system.pos = xyz
        print("start",system.vel)
        if box is None:
            box, _ = read_cp2k_box(self.input_files['template'])
    	# to-do: retrieve system.vpot from previous energy file.
        if None not in ((rescale, system.vpot)) and rescale is not False:
            print("Rescale")
            if rescale > 0:
                kin_old = rescale - system.vpot
                do_rescale = True
            else:
                print("Warning")
                logger.warning('Ignored re-scale 6.2%f < 0.0.', rescale)
                return 0.0, kinetic_energy(vel, mass)[0]
        else:
            kin_old = kinetic_energy(vel, mass)[0]
            do_rescale = False
        if vel_settings.get('aimless', False):
            vel, _ = rgen.draw_maxwellian_velocities(vel, mass, beta)
            print("VEL",vel[0])
            print("Aimless")
        else:
            dvel, _ = rgen.draw_maxwellian_velocities(vel, mass, beta, sigma_v=vel_settings['sigma_v'])
            print("Aimless false")
            vel += dvel
        # make reset momentum the default
        if vel_settings.get('momentum', True):
            print("CHECK VEL SETTINGS MOMENTUM")
            vel = reset_momentum(vel, mass)
        if do_rescale:
            #system.rescale_velocities(rescale, external=True)
            raise NotImplementedError("Option 'rescale_energy' is not implemented for CP2K yet.")
        conf_out = os.path.join(self.exe_dir,
                '{}.{}'.format('genvel', self.ext))
        write_xyz_trajectory(conf_out, xyz, vel,
                atoms, box, append=False)
        kin_new = kinetic_energy(vel, mass)[0]
        system.config=(conf_out, None)
        system.ekin = kin_new
        if kin_old == 0.0:
            dek = float('inf')
            logger.debug(('Kinetic energy not found for previous point.'
                '\n(This happens when the initial configuration '
                'does not contain energies.)'))
        else:
           dek = kin_new - kin_old
        print("end",system.vel)
        return dek, kin_new
