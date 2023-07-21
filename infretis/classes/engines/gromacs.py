"""Gromacs engine."""

from time import sleep
import signal
import subprocess
import os
import shlex
import logging
import struct
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    look_for_input_files,
    box_matrix_to_list,
)
import numpy as np
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_GROMACS_MAGIC = 1993
_GRO_FMT = '{0:5d}{1:5s}{2:5s}{3:5d}{4:8.3f}{5:8.3f}{6:8.3f}'
_GRO_VEL_FMT = _GRO_FMT + '{7:8.4f}{8:8.4f}{9:8.4f}'
_GRO_BOX_FMT = '{:15.9f}'
_G96_FMT = '{0:}{1:15.9f}{2:15.9f}{3:15.9f}\n'
_G96_FMT_FULL = '{0:5d} {1:5s} {2:5s}{3:7d}{4:15.9f}{5:15.9f}{6:15.9f}\n'
_G96_BOX_FMT = '{:15.9f}' * 9 + '\n'
_G96_BOX_FMT_3 = '{:15.9f}' * 3 + '\n'
_GROMACS_MAGIC = 1993
_DIM = 3
_TRR_VERSION = 'GMX_trn_file'
_TRR_VERSION_B = b'GMX_trn_file'
_SIZE_FLOAT = struct.calcsize('f')
_SIZE_DOUBLE = struct.calcsize('d')
_HEAD_FMT = '{}13i'
_HEAD_ITEMS = ('ir_size', 'e_size', 'box_size', 'vir_size', 'pres_size',
               'top_size', 'sym_size', 'x_size', 'v_size', 'f_size',
               'natoms', 'step', 'nre', 'time', 'lambda')
TRR_HEAD_SIZE = 1000
TRR_DATA_ITEMS = ('box_size', 'vir_size', 'pres_size',
                  'x_size', 'v_size', 'f_size')


class GromacsEngine(EngineBase):
    """
    A class for interfacing GROMACS.

    This class defines the interface to GROMACS.

    Attributes
    ----------
    gmx : string
        The command for executing GROMACS. Note that we are assuming
        that we are using version 5 (or later) of GROMACS.
    mdrun : string
        The command for executing GROMACS mdrun. In some cases, this
        executable can be different from ``gmx mdrun``.
    mdrun_c : string
        The command for executing GROMACS mdrun when continuing a
        simulation. This is derived from the ``mdrun`` command.
    input_path : string
        The directory where the input files are stored.
    subcycles : int,
        The number of simulation steps of the external engine for each
        PyRETIS step (e.g. interaction between the softwares frequency)
    exe_path : string, optional
        The absolute path at which the main PyRETIS simulation will be run.
    maxwarn : integer
        Setting for the GROMACS ``grompp -maxwarn`` option.
    gmx_format : string
        This string selects the output format for GROMACS.
    write_vel : boolean, optional
        True if we want to output the velocities.
    write_force : boolean, optional
        True if we want to output the forces.

    """

    def __init__(self, gmx, mdrun, input_path, timestep, subcycles,
                 exe_path=os.path.abspath('.'),
                 maxwarn=0, gmx_format='gro',
                 write_vel=True,
                 write_force=False):
        """Set up the GROMACS engine.

        Parameters
        ----------
        gmx : string
            The GROMACS executable.
        mdrun : string
            The GROMACS mdrun executable.
        input_path : string
            The absolute path to where the input files are stored.
        timestep : float
            The time step used in the GROMACS MD simulation.
        subcycles : integer
            The number of steps each GROMACS MD run is composed of.
        exe_path : string, optional
            The absolute path at which the main PyRETIS simulation will be run.
        maxwarn : integer, optional
            Setting for the GROMACS ``grompp -maxwarn`` option.
        gmx_format : string, optional
            The format used for GROMACS configurations.
        write_vel : boolean, optional
            Determines if GROMACS should write velocities or not.
        write_force : boolean, optional
            Determines if GROMACS should write forces or not.

        """
        super().__init__('GROMACS engine zamn', timestep, subcycles)
        self.ext = gmx_format
        if self.ext not in ('g96', 'gro'):
            msg = 'Unknown GROMACS format: "%s"'
            logger.error(msg, self.ext)
            raise ValueError(msg % self.ext)
        # Define the GROMACS GMX command:
        self.gmx = gmx
        # Define GROMACS GMX MDRUN commands:
        self.mdrun = mdrun + ' -s {} -deffnm {} -c {}'
        # This is for continuation of a GROMACS simulation:
        self.mdrun_c = mdrun + ' -s {} -cpi {} -append -deffnm {} -c {}'
        self.ext_time = self.timestep * self.subcycles
        self.maxwarn = maxwarn
        # Define the energy terms, these are hard-coded, but
        # here we open up for changing that:
        self.energy_terms = self.select_energy_terms('path')
        self.input_path = os.path.join(exe_path, input_path)
        # Set the defaults input files:
        default_files = {
            'conf': f'conf.{self.ext}',
            'input_o': 'grompp.mdp',  # "o" = original input file.
            'topology': 'topol.top'}
        extra_files = {
            'index': 'index.ndx',
        }

        # An user doesn't need to have problems with g96 and mdtraj.
        file_g = os.path.join(self.input_path, 'conf.')
        if self.ext == 'gro':
            self.top, _, _, _ = read_gromacs_gro_file(file_g+self.ext)
        elif self.ext == 'g96':
            if not os.path.isfile(file_g+'gro'):
                cmd = [self.gmx, 'editconf',
                       '-f', file_g+self.ext,
                       '-o', file_g+'gro']
                self.execute_command(cmd, cwd=None)

            self.top, _, _, _ = read_gromos96_file(file_g+self.ext)
            self.top['VELOCITY'] = self.top['POSITION'].copy()

        # Check the presence of the defaults input files or, if absent,
        # try to find them by their expected extension.
        self.input_files = look_for_input_files(self.input_path,
                                                default_files,
                                                extra_files)
        # Check the input file and create a PyRETIS version with
        # consistent settings:
        settings = {
            'dt': self.timestep,
            'nstxout-compressed': 0,
            'gen_vel': 'no'
        }
        for key in ('nsteps', 'nstxout', 'nstvout', 'nstfout', 'nstlog',
                    'nstcalcenergy', 'nstenergy'):
            settings[key] = self.subcycles
        if not write_vel:
            settings['nstvout'] = 0
        if not write_force:
            settings['nstfout'] = 0

        # PyRETIS construct its own mdp file
        self.input_files['input'] = os.path.join(self.input_path,
                                                 'pyretis.mdp')
        self._modify_input(self.input_files['input_o'],
                           self.input_files['input'], settings, delim='=')
        logger.info(('Created GROMACS mdp input from %s. You might '
                     'want to check the input file: %s'),
                    self.input_files['input_o'], self.input_files['input'])

        # Generate a tpr file using the input files:
        logger.info('Creating ".tpr" for GROMACS in %s', self.input_path)
        self.exe_dir = self.input_path

        out_files = self._execute_grompp(self.input_files['input'],
                                         self.input_files['conf'], 'topol')

        # This will generate some noise, let's remove files we don't need:
        mdout = os.path.join(self.input_path, out_files['mdout'])
        self._removefile(mdout)
        # We also remove GROMACS backup files after creating the tpr:
        self._remove_gromacs_backup_files(self.input_path)
        # Keep the tpr file.
        self.input_files['tpr'] = os.path.join(self.input_path,
                                               out_files['tpr'])
        logger.info('GROMACS ".tpr" created: %s', self.input_files['tpr'])

    @staticmethod
    def select_energy_terms(terms):
        """Select energy terms to extract from GROMACS.

        Parameters
        ----------
        terms : string
            This string will name the terms to extract. Currently
            we only allow for two types of output, but this can be
            customized in the future.

        """
        allowed_terms = {
            'full': ('\n'.join(('Potential', 'Kinetic-En.', 'Total-Energy',
                                'Temperature', 'Pressure'))).encode(),
            'path': b'Potential\nKinetic-En.',
        }
        if terms not in allowed_terms:
            return allowed_terms['path']
        return allowed_terms[terms]

    @staticmethod
    def rename_energies(gmx_energy):
        """Rename GROMACS energy terms to PyRETIS convention."""
        energy_map = {'potential': 'vpot',
                      'kinetic en.': 'ekin',
                      'temperature': 'temp',
                      'total energy': 'etot',
                      'pressure': 'press'}
        energy = {}
        for key, val in gmx_energy.items():
            name = energy_map.get(key, key)
            energy[name] = val[0]
        return energy

    def _execute_grompp(self, mdp_file, config, deffnm):
        """Execute the GROMACS preprocessor.

        Parameters
        ----------
        mdp_file : string
            The path to the mdp file.
        config : string
            The path to the GROMACS config file to use as input.
        deffnm : string
            A string used to name the GROMACS files.

        Returns
        -------
        out_files : dict
            This dict contains files that were created by the GROMACS
            preprocessor.

        """
        topol = self.input_files['topology']
        tpr = f'{deffnm}.tpr'
        cmd = [self.gmx, 'grompp', '-f', mdp_file, '-c', config,
               '-p', topol, '-o', tpr]
        cmd = shlex.split(' '.join(cmd))
        if 'index' in self.input_files:
            cmd.extend(['-n', self.input_files['index']])
        if self.maxwarn > 0:
            cmd.extend(['-maxwarn', str(self.maxwarn)])
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {'tpr': tpr, 'mdout': 'mdout.mdp'}
        return out_files

    def _execute_mdrun(self, tprfile, deffnm):
        """
        Execute GROMACS mdrun.

        This method is intended as the initial ``gmx mdrun`` executed.
        That is, we here assume that we do not continue a simulation.

        Parameters
        ----------
        tprfile : string
            The .tpr file to use for executing GROMACS.
        deffnm : string
            To give the GROMACS simulation a name.

        Returns
        -------
        out_files : dict
            This dict contains the output files created by ``mdrun``.
            Note that we here hard code the file names.

        """
        confout = f'{deffnm}.{self.ext}'
        cmd = shlex.split(self.mdrun.format(tprfile, deffnm, confout))
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {'conf': confout,
                     'cpt_prev': f'{deffnm}_prev.cpt'}
        for key in ('cpt', 'edr', 'log', 'trr'):
            out_files[key] = f'{deffnm}.{key}'
        self._remove_gromacs_backup_files(self.exe_dir)
        return out_files

    def _execute_grompp_and_mdrun(self, config, deffnm):
        """
        Execute GROMACS ``grompp`` and ``mdrun``.

        Here we use the input file given in the input directory.

        Parameters
        ----------
        config : string
            The path to the GROMACS config file to use as input.
        deffnm : string
            A string used to name the GROMACS files.

        Returns
        -------
        out_files : dict of strings
            The files created by this command.

        """
        out_files = {}
        out_grompp = self._execute_grompp(self.input_files['input'],
                                          config, deffnm)
        tpr_file = out_grompp['tpr']
        for key, value in out_grompp.items():
            out_files[key] = value
        out_mdrun = self._execute_mdrun(tpr_file,
                                        deffnm)
        for key, value in out_mdrun.items():
            out_files[key] = value
        return out_files

    def _execute_mdrun_continue(self, tprfile, cptfile, deffnm):
        """
        Continue the execution of GROMACS.

        Here, we assume that we have already executed ``gmx mdrun`` and
        that we are to append and continue a simulation.

        Parameters
        ----------
        tprfile : string
            The .tpr file which defines the simulation.
        cptfile : string
            The last checkpoint file (.cpt) from the previous run.
        deffnm : string
            To give the GROMACS simulation a name.

        Returns
        -------
        out_files : dict
            The output files created/appended by GROMACS when we
            continue the simulation.

        """
        confout = f'{deffnm}.{self.ext}'.format(deffnm, self.ext)
        self._removefile(confout)
        cmd = shlex.split(self.mdrun_c.format(tprfile, cptfile,
                                              deffnm, confout))
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {'conf': confout}
        for key in ('cpt', 'edr', 'log', 'trr'):
            out_files[key] = f'{deffnm}.{key}'
        self._remove_gromacs_backup_files(self.exe_dir)
        return out_files

    def _extend_gromacs(self, tprfile, time):
        """Extend a GROMACS simulation.

        Parameters
        ----------
        tprfile : string
            The file to read for extending.
        time : float
            The time (in ps) to extend the simulation by.

        Returns
        -------
        out_files : dict
            The files created by GROMACS when we extend.

        """
        tpxout = f'ext_{tprfile}'
        self._removefile(tpxout)
        cmd = [self.gmx, 'convert-tpr', '-s', tprfile,
               '-extend', str(time), '-o', tpxout]
        self.execute_command(cmd, cwd=self.exe_dir)
        out_files = {'tpr': tpxout}
        return out_files

    def _extend_and_execute_mdrun(self, tpr_file, cpt_file, deffnm):
        """Extend GROMACS and execute mdrun.

        Parameters
        ----------
        tpr_file : string
            The location of the "current" .tpr file.
        cpt_file : string
            The last checkpoint file (.cpt) from the previous run.
        deffnm : string
            To give the GROMACS simulation a name.

        Returns
        -------
        out_files : dict
            The files created by GROMACS when we extend.

        """
        out_files = {}
        out_grompp = self._extend_gromacs(tpr_file, self.ext_time)
        ext_tpr_file = out_grompp['tpr']
        for key, value in out_grompp.items():
            out_files[key] = value
        out_mdrun = self._execute_mdrun_continue(ext_tpr_file, cpt_file,
                                                 deffnm)
        for key, value in out_mdrun.items():
            out_files[key] = value
        # Move extended tpr so that we can continue extending:
        source = os.path.join(self.exe_dir, ext_tpr_file)
        dest = os.path.join(self.exe_dir, tpr_file)
        self._movefile(source, dest)
        out_files['tpr'] = tpr_file
        return out_files

    def _remove_gromacs_backup_files(self, dirname):
        """Remove files GROMACS has backed up.

        These are files starting with a '#'

        Parameters
        ----------
        dirname : string
            The directory where we are to remove files.

        """
        for entry in os.scandir(dirname):
            if entry.name.startswith('#') and entry.is_file():
                filename = os.path.join(dirname, entry.name)
                self._removefile(filename)

    def _extract_frame(self, traj_file, idx, out_file):
        """Extract a frame from a .trr, .xtc or .trj file.

        If the extension is different from .trr, .xtc or .trj, we will
        basically just copy the given input file.

        Parameters
        ----------
        traj_file : string
            The GROMACS file to open.
        idx : integer
            The frame number we look for.
        out_file : string
            The file to dump to.

        Note
        ----
        This will only properly work if the frames in the input
        trajectory are uniformly spaced in time.

        """
        trajexts = ['.trr', '.xtc', '.trj']

        logger.debug('Extracting frame, idx = %i', idx)
        logger.debug('Source file: %s, out file: %s', traj_file, out_file)
        if traj_file[-4:] in trajexts:
            _, data = read_trr_frame(traj_file, idx)
            xyz = data['x']
            vel = data.get('v')
            box = box_matrix_to_list(data['box'], full=True)
            if out_file[-4:] == '.gro':
                write_gromacs_gro_file(out_file, self.top, xyz, vel, box)
            elif out_file[-4:] == '.g96':
                write_gromos96_file(out_file, self.top, xyz, vel, box)

        else:
            cmd = [self.gmx, 'trjconv',
                   '-f', traj_file,
                   '-s', self.input_files['tpr'],
                   '-o', out_file]

            self.execute_command(cmd, inputs=b'0', cwd=None)

    def get_energies(self, energy_file, begin=None, end=None):
        """Return energies from a GROMACS run.

        Parameters
        ----------
        energy_file : string
            The file to read energies from.
        begin : float, optional
            Select the time for the first frame to read.
        end : float, optional
            Select the time for the last frame to read.

        Returns
        -------
        energy : dict fo numpy.arrays
            The energies read from the produced GROMACS xvg file.

        """
        cmd = [self.gmx, 'energy', '-f', energy_file]
        if begin is not None:
            begin = max(begin, 0)
            cmd.extend(['-b', str(begin)])
        if end is not None:
            cmd.extend(['-e', str(end)])
        self.execute_command(cmd, inputs=self.energy_terms,
                             cwd=self.exe_dir)
        xvg_file = os.path.join(self.exe_dir, 'energy.xvg')
        energy = read_xvg_file(xvg_file)
        self._removefile(xvg_file)
        return energy

    def _propagate_from(self, name, path, system, ens_set, msg_file, reverse=False):
        """
        Propagate with GROMACS from the current system configuration.

        Here, we assume that this method is called after the propagate()
        has been called in the parent. The parent is then responsible
        for reversing the velocities and also for setting the initial
        state of the system.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`pyretis.core.Path.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble: dict
            it contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces` : list of floats
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
        status = f'propagating with GROMACS (reverse = {reverse})'
        # system = ensemble['system']
        interfaces = ens_set['interfaces']
        order_function = self.order_function
        logger.debug(status)
        success = False
        left, _, right = interfaces
        # Dumping of the initial config were done by the parent, here
        # we will just use it:
        initial_conf = system.config[0]
        # Get the current order parameter:
        # order = self.calculate_order(ensemble)
        order = self.calculate_order(system)
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        # So, here we will just blast off GROMACS and check the .trr
        # output when we can.
        # 1) Create mdp_file with updated number of steps:
        settings = {'gen_vel': 'no',
                    'nsteps': path.maxlen * self.subcycles,
                    'continuation': 'no'}
        mdp_file = os.path.join(self.exe_dir, f'{name}.mdp')
        self._modify_input(self.input_files['input'], mdp_file, settings,
                           delim='=')
        # 2) Run GROMACS preprocessor:
        out_files = self._execute_grompp(mdp_file, initial_conf, name)
        # Generate some names that will be created by mdrun:
        confout = f'{name}.{self.ext}'
        out_files['conf'] = confout
        out_files['cpt_prev'] = f'{name}_prev.cpt'
        for key in ('cpt', 'edr', 'log', 'trr'):
            out_files[key] = f'{name}.{key}'
        # Remove some of these files if present (e.g. left over from a
        # crashed simulation). This is so that GromacsRunner will not
        # start reading a .trr left from a previous simulation.

        # if right == -0.26:
        #     print('pipipipi')

        remove = [val for key, val in out_files.items() if key != 'tpr']
        self._remove_files(self.exe_dir, remove)
        tpr_file = out_files['tpr']
        trr_file = os.path.join(self.exe_dir, out_files['trr'])
        edr_file = os.path.join(self.exe_dir, out_files['edr'])
        cmd = shlex.split(self.mdrun.format(tpr_file, name, confout))
        # 3) Fire off GROMACS mdrun:
        logger.debug('Executing GROMACS.')
        msg_file.write(f'# Trajectory file is: {trr_file}')
        msg_file.write('# Starting GROMACS.')
        msg_file.write('# Step order parameter cv1 cv2 ...')
        with GromacsRunner(cmd, trr_file, edr_file, self.exe_dir) as gro:
            for i, data in enumerate(gro.get_gromacs_frames()):
                # Update the configuration file:
                system.set_pos((trr_file, i))
                # Also provide the loaded positions since they are
                # available:
                system.pos = data['x']
                system.vel = data.get('v', None)
                if system.vel is not None and reverse:
                    system.vel *= -1
                # ##### length = box_matrix_to_list(data['box'])
                # ##### system.update_box(length)
                order = order_function.calculate(system)
                msg_file.write(f'{i} {" ".join([str(j) for j in order])}')
                snapshot = {'order': order,
                            'config': (trr_file, i),
                            'vel_rev': reverse}
                phase_point = self.snapshot_to_system(system, snapshot)
                status, success, stop, _ = self.add_to_path(path, phase_point,
                                                            left, right)
                if stop:
                    logger.debug('Ending propagate at %i. Reason: %s',
                                 i, status)
                    break
        logger.debug('GROMACS propagation done, obtaining energies!')
        msg_file.write('# Propagation done.')
        msg_file.write(f'# Reading energies from: {out_files["edr"]}')
        energy = self.get_energies(out_files['edr'])
        path.update_energies(energy['kinetic en.'], energy['potential'])
        logger.debug('Removing GROMACS output after propagate.')
        remove = [val for key, val in out_files.items()
                  if key not in ('trr', 'gro', 'g96')]
        self._remove_files(self.exe_dir, remove)
        self._remove_gromacs_backup_files(self.exe_dir)
        msg_file.flush()
        return success, status

    def step(self, system, name):
        """Perform a single step with GROMACS.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.
        name : string
            To name the output files from the GROMACS step.

        Returns
        -------
        out : string
            The name of the output configuration, obtained after
            completing the step.

        """
        initial_conf = self.dump_frame(system)
        # Save as a single snapshot file:
        system.particles.set_pos((initial_conf, None))
        system.particles.set_vel(False)
        out_grompp = self._execute_grompp(self.input_files['input'],
                                          initial_conf,
                                          name)
        out_mdrun = self._execute_mdrun(out_grompp['tpr'], name)
        conf_abs = os.path.join(self.exe_dir, out_mdrun['conf'])
        logger.debug('Obtaining GROMACS energies after single step.')
        energy = self.get_energies(out_mdrun['edr'])
        system.particles.set_pos((conf_abs, None))
        system.particles.set_vel(False)
        system.particles.vpot = energy['potential'][-1]
        system.particles.ekin = energy['kinetic en.'][-1]
        logger.debug('Removing GROMACS output after single step.')
        remove = [val for _, val in out_grompp.items()]
        remove += [val for key, val in out_mdrun.items() if key != 'conf']
        self._remove_files(self.exe_dir, remove)
        return out_mdrun['conf']

    def _prepare_shooting_point(self, input_file):
        """
        Create the initial configuration for a shooting move.

        This creates a new initial configuration with random velocities.
        Here, the random velocities are obtained by running a zero-step
        GROMACS simulation.

        Parameters
        ----------
        input_file : string
            The input configuration to generate velocities for.

        Returns
        -------
        output_file : string
            The name of the file created.
        energy : dict
            The energy terms read from the GROMACS .edr file.

        """
        # gen_mdp = os.path.join(self.exe_dir, 'genvel.mdp')
        gen_mdp = os.path.join(self.exe_dir, 'genvel.mdp')
        if os.path.isfile(gen_mdp):
            logger.debug('%s found. Re-using it!', gen_mdp)
        else:
            # Create output file to generate velocities:
            settings = {'gen_vel': 'yes', 'gen_seed': -1, 'nsteps': 0,
                        'continuation': 'no'}
            self._modify_input(self.input_files['input'], gen_mdp, settings,
                               delim='=')
        # Run GROMACS grompp for this input file:
        # out_grompp = self._execute_grompp(os.path.basename(gen_mdp), os.path.basename(input_file), 'genvel')
        out_grompp = self._execute_grompp(gen_mdp, input_file, 'genvel')
        remove = [val for _, val in out_grompp.items()]
        # Run GROMACS mdrun for this tpr file:
        out_mdrun = self._execute_mdrun(out_grompp['tpr'], 'genvel')
        remove += [val for key, val in out_mdrun.items() if key != 'conf']
        confout = os.path.join(self.exe_dir, out_mdrun['conf'])
        energy = self.get_energies(out_mdrun['edr'])
        # Remove run-files:
        logger.debug('Removing GROMACS output after velocity generation.')
        self._remove_files(self.exe_dir, remove)
        return confout, energy

    def set_mdrun(self, config, md_items):
        """Sets the worker terminal command to be run"""
        base = config['dask']['wmdrun'][md_items['pin']]
        self.mdrun = base + ' -s {} -deffnm {} -c {}'
        self.mdrun_c = base + ' -s {} -cpi {} -append -deffnm {} -c {}'
        self.exe_dir = md_items['w_folder']

    def _read_configuration(self, filename):
        """Read output from GROMACS .g96/gro files.

        Parameters
        ----------
        filename : string
            The file to read the configuration from.

        Returns
        -------
        box : numpy.array
            The box dimensions.
        xyz : numpy.array
            The positions.
        vel : numpy.array
            The velocities.

        """
        box = None
        if self.ext == 'g96':
            _, xyz, vel, box = read_gromos96_file(filename)
        elif self.ext == 'gro':
            _, xyz, vel, box = read_gromacs_gro_file(filename)
        else:
            msg = 'GROMACS engine does not support reading "%s"'
            logger.error(msg, self.ext)
            raise ValueError(msg % self.ext)
        return box, xyz, vel

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
        if self.ext == 'g96':
            txt, xyz, vel, _ = read_gromos96_file(filename)
            write_gromos96_file(outfile, txt, xyz, -1 * vel)
        elif self.ext == 'gro':
            txt, xyz, vel, _ = read_gromacs_gro_file(filename)
            write_gromacs_gro_file(outfile, txt, xyz, -1 * vel)
        else:
            msg = 'GROMACS engine does not support writing "%s"'
            logger.error(msg, self.ext)
            raise ValueError(msg % self.ext)

    def modify_velocities(self, system, vel_settings):
        """Modify the velocities of the current state.

        This method will modify the velocities of a time slice.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system`: object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating.

        vel_settings: dict
            It contains:

            * `sigma_v`: numpy.array, optional
              These values can be used to set a standard deviation (one
              for each particle) for the generated velocities.
            * `aimless`: boolean, optional
              Determines if we should do aimless shooting or not.
            * `momentum`: boolean, optional
              If True, we reset the linear momentum to zero after
              generating.
            * `rescale or rescale_energy`: float, optional
              In some NVE simulations, we may wish to re-scale the
              energy to a fixed value. If `rescale` is a float > 0,
              we will re-scale the energy (after modification of
              the velocities) to match the given float.

        Returns
        -------
        dek : float
            The change in the kinetic energy.
        kin_new : float
            The new kinetic energy.

        """
        dek = None
        kin_old = None
        kin_new = None
        rescale = vel_settings.get('rescale_energy',
                                   vel_settings.get('rescale'))
        if rescale is not None and rescale is not False and rescale > 0:
            msgtxt = 'GROMACS engine does not support energy re-scale.'
            logger.error(msgtxt)
            raise NotImplementedError(msgtxt)
        kin_old = system.ekin
        if vel_settings.get('aimless', False):
            pos = self.dump_frame(system)
            posvel, energy = self._prepare_shooting_point(pos)
            kin_new = energy['kinetic en.'][-1]
            system.set_pos((posvel, None))
            system.set_vel(False)
            system.ekin = kin_new
            system.vpot = energy['potential'][-1]
        else:  # Soft velocity change, from a Gaussian distribution:
            msgtxt = 'GROMACS engine only support aimless shooting!'
            logger.error(msgtxt)
            raise NotImplementedError(msgtxt)
        if vel_settings.get('momentum', False):
            pass
        if kin_old is None or kin_new is None:
            dek = float('inf')
            logger.debug(('Kinetic energy not found for previous point.'
                          '\n(This happens when the initial configuration '
                          'does not contain energies.)'))
        else:
            dek = kin_new - kin_old
        return dek, kin_new


class GromacsRunner:
    """A helper class for running GROMACS.

    This class handles the reading of the TRR on the fly and
    it is used to decide when to end the GROMACS execution.

    Attributes
    ----------
    cmd : string
        The command for executing GROMACS.
    trr_file : string
        The GROMACS TRR file we are going to read.
    edr_file : string
        A .edr file we are going to read.
    exe_dir : string
        Path to where we are currently running GROMACS.
    fileh : file object
        The current open file object.
    running : None or object like :py:class:`subprocess.Popen`
        The process running GROMACS.
    bytes_read : integer
        The number of bytes read so far from the TRR file.
    ino : integer
        The current inode we are using for the file.
    stop_read : boolean
        If this is set to True, we will stop the reading.
    SLEEP : float
        How long we wait after an unsuccessful read before
        reading again.
    data_size : integer
        The size of the data (x, v, f, box, etc.) in the TRR file.
    header_size : integer
        The size of the header in the TRR file.

    """

    SLEEP = 0.1

    def __init__(self, cmd, trr_file, edr_file, exe_dir):
        """Set the GROMACS command and the files we need.

        Parameters
        ----------
        cmd : string
            The command for executing GROMACS.
        trr_file : string
            The GROMACS TRR file we are going to read.
        edr_file : string
            A .edr file we are going to read.
        exe_dir : string
            Path to where we are currently running GROMACS.

        """
        self.cmd = cmd
        self.trr_file = trr_file
        self.edr_file = edr_file
        self.exe_dir = exe_dir
        self.fileh = None
        self.running = None
        self.bytes_read = 0
        self.ino = 0
        self.stop_read = True
        self.data_size = 0
        self.header_size = 0
        self.stdout_name = None
        self.stderr_name = None
        self.stdout = None
        self.stderr = None

    def start(self):
        """Start execution of GROMACS and wait for output file creation."""
        logger.debug('Starting GROMACS execution in %s', self.exe_dir)

        self.stdout_name = os.path.join(self.exe_dir, 'stdout.txt')
        self.stderr_name = os.path.join(self.exe_dir, 'stderr.txt')
        self.stdout = open(self.stdout_name, 'wb')
        self.stderr = open(self.stderr_name, 'wb')

        self.running = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=self.stdout,
            stderr=self.stderr,
            shell=False,
            cwd=self.exe_dir,
            preexec_fn=os.setsid,
        )
        present = []
        # Wait for the TRR/EDR files to appear:
        for fname in (self.trr_file, self.edr_file):
            while not os.path.isfile(fname):
                logger.debug('Waiting for GROMACS file "%s"', fname)
                sleep(self.SLEEP)
                poll = self.check_poll()
                if poll is not None:
                    logger.debug('GROMACS execution stopped')
                    break
            if os.path.isfile(fname):
                present.append(fname)
        # Prepare and open the TRR file:
        self.bytes_read = 0
        # Ok, so GROMACS might have crashed in between writing the
        # files. Check that both files are indeed here:
        if self.trr_file in present and self.edr_file in present:
            self.fileh = open(self.trr_file, 'rb')
            self.ino = os.fstat(self.fileh.fileno()).st_ino
            self.stop_read = False
        else:
            self.stop_read = True

    def __enter__(self):
        """Start running GROMACS, for a context manager."""
        self.start()
        return self

    def get_gromacs_frames(self):
        """Read the GROMACS TRR file on-the-fly."""
        first_header = True
        header = None
        while not self.stop_read:
            poll = self.check_poll()
            if poll is not None:
                # GROMACS is done, read remaining data.
                self.stop_read = True
                if os.path.getsize(self.trr_file) - self.bytes_read > 0:
                    for _, data, _ in read_remaining_trr(self.trr_file,
                                                         self.fileh,
                                                         self.bytes_read):
                        yield data

            else:
                # First we try to get the header from the file:
                size = os.path.getsize(self.trr_file)
                if self.header_size == 0:
                    header_size = TRR_HEAD_SIZE
                else:
                    header_size = self.header_size
                if size >= self.bytes_read + header_size:
                    # Try to read next frame:
                    try:
                        header, new_bytes = read_trr_header(self.fileh)
                    except EOFError:
                        new_fileh, new_ino = reopen_file(self.trr_file,
                                                         self.fileh,
                                                         self.ino,
                                                         self.bytes_read)
                        if new_fileh is not None:
                            self.fileh = new_fileh
                            self.ino = new_ino
                    if header is not None:
                        self.bytes_read += new_bytes
                        self.header_size = new_bytes
                        if first_header:
                            logger.debug('TRR header was: %i', new_bytes)
                            first_header = False
                        # Calculate the size of the data:
                        self.data_size = sum([header[key] for key in
                                              TRR_DATA_ITEMS])
                        data = None
                        while data is None:
                            size = os.path.getsize(self.trr_file)
                            if size >= self.bytes_read + self.data_size:
                                try:
                                    data, new_bytes = get_data(self.fileh,
                                                               header)
                                except EOFError:
                                    new_fileh, new_ino = reopen_file(
                                        self.trr_file,
                                        self.fileh,
                                        self.ino,
                                        self.bytes_read)
                                    if new_fileh is not None:
                                        self.fileh = new_fileh
                                        self.ino = new_ino
                                if data is None:
                                    # Data is not ready, just wait:
                                    sleep(self.SLEEP)
                                else:
                                    self.bytes_read += new_bytes
                                    yield data
                            else:
                                # Data is not ready, just wait:
                                sleep(self.SLEEP)
                else:
                    # Header was not ready, just wait before trying again.
                    sleep(self.SLEEP)

    def close(self):
        """Close the file, in case that is explicitly needed."""
        if self.fileh is not None and not self.fileh.closed:
            logger.debug('Closing GROMACS file: "%s"', self.trr_file)
            self.fileh.close()
        for handle in (self.stdout, self.stderr):
            if handle is not None and not handle.closed:
                handle.close()

    def stop(self):
        """Stop the current GROMACS execution."""
        if self.running:
            for handle in (self.running.stdin, self.running.stdout,
                           self.running.stderr):
                if handle:
                    try:
                        handle.close()
                    except AttributeError:
                        pass
            if self.running.returncode is None:
                logger.debug('Terminating GROMACS execution')
                os.killpg(os.getpgid(self.running.pid), signal.SIGTERM)

            self.running.wait(timeout=360)
        self.stop_read = True
        self.close()  # Close the TRR file.

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Just stop execution and close file for a context manager."""
        self.stop()

    def __del__(self):
        """Just stop execution and close file."""
        self.stop()

    def check_poll(self):
        """Check the current status of the running subprocess."""
        if self.running:
            poll = self.running.poll()
            if poll is not None:
                logger.debug('Execution of GROMACS stopped')
                logger.debug('Return code was: %i', poll)
                if poll != 0:
                    logger.error('STDOUT, see file: %s', self.stdout_name)
                    logger.error('STDERR, see file: %s', self.stderr_name)
                    raise RuntimeError('Error in GROMACS execution.')
            return poll
        raise RuntimeError('GROMACS is not running.')

def read_trr_frame(filename, index):
    """Return a given frame from a TRR file."""
    idx = 0
    with open(filename, 'rb') as infile:
        while True:
            try:
                header, _ = read_trr_header(infile)
                if idx == index:
                    data = read_trr_data(infile, header)
                    return header, data
                skip_trr_data(infile, header)
                idx += 1
                if idx > index:
                    logger.error('Frame %i not found in %s', index, filename)
                    return None, None
            except EOFError:
                return None, None


def read_trr_header(fileh):
    """Read a header from a TRR file.

    Parameters
    ----------
    fileh : file object
        The file handle for the file we are reading.

    Returns
    -------
    header : dict
        The header read from the file.

    """
    start = fileh.tell()
    endian = '>'
    magic = read_struct_buff(fileh, f'{endian}1i')[0]
    if magic == _GROMACS_MAGIC:
        pass
    else:
        magic = swap_integer(magic)
        if not magic == _GROMACS_MAGIC:
            logger.critical(
                'TRR file might be inconsistent! Could find _GROMACS_MAGIC'
            )
        endian = swap_endian(endian)
    slen = read_struct_buff(fileh, f'{endian}2i')
    raw = read_struct_buff(fileh, f'{endian}{slen[0] - 1}s')
    version = raw[0].split(b'\0', 1)[0].decode('utf-8')
    if not version == _TRR_VERSION:
        raise ValueError('Unknown format')

    head_fmt = _HEAD_FMT.format(endian)
    head_s = read_struct_buff(fileh, head_fmt)
    header = {}
    for i, val in enumerate(head_s):
        key = _HEAD_ITEMS[i]
        header[key] = val
    # The next are either floats or double
    double = is_double(header)
    if double:
        fmt = f'{endian}2d'
    else:
        fmt = f'{endian}2f'
    header_r = read_struct_buff(fileh, fmt)
    header['time'] = header_r[0]
    header['lambda'] = header_r[1]
    header['endian'] = endian
    header['double'] = double
    return header, fileh.tell() - start


def gromacs_settings(settings, input_path):
    """Read and processes GROMACS settings.

    Parameters
    ----------
    settings : dict
        The current input settings..
    input_path : string
        The GROMACS input path

    """
    ext = settings['engine'].get('gmx_format', 'gro')
    default_files = {'conf': f'conf.{ext}',
                     'input_o': 'grompp.mdp',
                     'topology': 'topol.top',
                     'index': 'index.ndx'}
    settings['engine']['input_files'] = {}
    for key in ('conf', 'input_o', 'topology', 'index'):
        # Add input path and the input files if input is no given:
        settings['engine']['input_files'][key] = \
            settings['engine'].get(key,
                                   os.path.join(input_path,
                                                default_files[key]))


def read_gromacs_generic(filename):
    """Read GROMACS files.

    This method will read a GROMACS file and yield the different
    snapshots found in the file. This file is intended to be used
    to just count the n of snapshots stored in a file.

    Parameters
    ----------
    filename : string
        The file to check.

    Yields
    ------
    out : None.

    """
    if filename[-4:] == '.gro':
        for i in read_gromacs_file(filename):
            yield None
    if filename[-4:] == '.g96':
        yield None
    if filename[-4:] == '.trr':
        for _ in read_trr_file(filename):
            yield None


def read_gromacs_file(filename):
    """Read GROMACS GRO files.

    This method will read a GROMACS file and yield the different
    snapshots found in the file. This file is intended to be used
    if we want to read all snapshots present in a file.

    Parameters
    ----------
    filename : string
        The file to open.

    Yields
    ------
    out : dict
        This dict contains the snapshot.

    """
    with open(filename, 'r', encoding='utf-8') as fileh:
        for snapshot in read_gromacs_lines(fileh):
            yield snapshot


def read_gromacs_gro_file(filename):
    """Read a single configuration GROMACS GRO file.

    This method will read the first configuration from the GROMACS
    GRO file and return the data as give by
    :py:func:`.read_gromacs_lines`. It will also explicitly
    return the matrices with positions, velocities and box size.

    Parameters
    ----------
    filename : string
        The file to read.

    Returns
    -------
    frame : dict
        This dict contains all the data read from the file.
    xyz : numpy.array
        The positions. The array is (N, 3) where N is the
        number of particles.
    vel : numpy.array
        The velocities. The array is (N, 3) where N is the
        number of particles.
    box : numpy.array
        The box dimensions.

    """
    snapshot = None
    xyz = None
    vel = None
    box = None
    with open(filename, 'r', encoding='utf8') as fileh:
        snapshot = next(read_gromacs_lines(fileh))
        box = snapshot.get('box', None)
        xyz = snapshot.get('xyz', None)
        vel = snapshot.get('vel', None)
    return snapshot, xyz, vel, box


def write_gromacs_gro_file(outfile, txt, xyz, vel=None, box=None):
    """Write configuration in GROMACS GRO format.

    Parameters
    ----------
    outfile : string
        The name of the file to create.
    txt : dict of lists of strings
        This dict contains the information on residue-numbers, names,
        etc. required to write the GRO file.
    xyz : numpy.array
        The positions to write.
    vel : numpy.array, optional
        The velocities to write.
    box: numpy.array, optional
        The box matrix.

    """
    resnum = txt['residunr']
    resname = txt['residuname']
    atomname = txt['atomname']
    atomnr = txt['atomnr']
    npart = len(xyz)
    with open(outfile, 'w', encoding='utf-8') as output:
        output.write(f'{txt["header"]}\n')
        output.write(f'{npart}\n')
        for i in range(npart):
            if vel is None:
                buff = _GRO_FMT.format(
                    resnum[i],
                    resname[i],
                    atomname[i],
                    atomnr[i],
                    xyz[i, 0],
                    xyz[i, 1],
                    xyz[i, 2])
            else:
                buff = _GRO_VEL_FMT.format(
                    resnum[i],
                    resname[i],
                    atomname[i],
                    atomnr[i],
                    xyz[i, 0],
                    xyz[i, 1],
                    xyz[i, 2],
                    vel[i, 0],
                    vel[i, 1],
                    vel[i, 2])
            output.write(f'{buff}\n')
        if box is None:
            box = ' '.join([_GRO_BOX_FMT.format(i) for i in txt['box']])
        else:
            box = ' '.join([_GRO_BOX_FMT.format(i) for i in box])
        output.write(f'{box}\n')


def read_gromos96_file(filename):
    """Read a single configuration GROMACS .g96 file.

    Parameters
    ----------
    filename : string
        The file to read.

    Returns
    -------
    rawdata : dict of list of strings
        This is the raw data read from the file grouped into sections.
        Note that this does not include the actual positions and
        velocities as these are returned separately.
    xyz : numpy.array
        The positions.
    vel : numpy.array
        The velocities.
    box : numpy.array
        The simulation box.

    """
    _len = 15
    _pos = 24
    rawdata = {'TITLE': [], 'POSITION': [], 'VELOCITY': [], 'BOX': [],
               'POSITIONRED': [], 'VELOCITYRED': []}
    section = None
    with open(filename, 'r', encoding='utf-8', errors='replace') as gromosfile:
        for lines in gromosfile:
            new_section = False
            stripline = lines.strip()
            if stripline == 'END':
                continue
            for key in rawdata:
                if stripline == key:
                    new_section = True
                    section = key
                    break
            if new_section:
                continue
            rawdata[section].append(lines.rstrip())
    txtdata = {}
    xyzdata = {}
    for key in ('POSITION', 'VELOCITY'):
        txtdata[key] = []
        xyzdata[key] = []
        for line in rawdata[key]:
            txt = line[:_pos]
            txtdata[key].append(txt)
            pos = [float(line[i:i+_len]) for i in range(_pos, 4*_len, _len)]
            xyzdata[key].append(pos)
        for line in rawdata[key+'RED']:
            txt = line[:_pos]
            txtdata[key].append(txt)
            pos = [float(line[i:i+_len]) for i in range(0, 3*_len, _len)]
            xyzdata[key].append(pos)
        xyzdata[key] = np.array(xyzdata[key])
    rawdata['POSITION'] = txtdata['POSITION']
    rawdata['VELOCITY'] = txtdata['VELOCITY']
    if not rawdata['VELOCITY']:
        # No velocities were found in the input file.
        xyzdata['VELOCITY'] = np.zeros_like(xyzdata['POSITION'])
        logger.info('Input g96 did not contain velocities')
    if rawdata['BOX']:
        box = np.array([float(i) for i in rawdata['BOX'][0].split()])
    else:
        box = None
        logger.info('Input g96 did not contain box vectors.')
    return rawdata, xyzdata['POSITION'], xyzdata['VELOCITY'], box


def write_gromos96_file(filename, raw, xyz, vel, box=None):
    """Write configuration in GROMACS .g96 format.

    Parameters
    ----------
    filename : string
        The name of the file to create.
    raw : dict of lists of strings
        This contains the raw data read from a .g96 file.
    xyz : numpy.array
        The positions to write.
    vel : numpy.array
        The velocities to write.
    box: numpy.array, optional
        The box matrix.

    """
    _keys = ('TITLE', 'POSITION', 'VELOCITY', 'BOX')
    with open(filename, 'w', encoding='utf-8') as outfile:
        for key in _keys:
            if key not in raw:
                continue
            outfile.write(f'{key}\n')
            for i, line in enumerate(raw[key]):
                if key == 'POSITION':
                    outfile.write(_G96_FMT.format(line, *xyz[i]))
                elif key == 'VELOCITY':
                    if vel is not None:
                        outfile.write(_G96_FMT.format(line, *vel[i]))
                elif box is not None and key == 'BOX':
                    if len(box) == 3:
                        outfile.write(_G96_BOX_FMT_3.format(*box))
                    else:
                        outfile.write(_G96_BOX_FMT.format(*box))
                else:
                    outfile.write(f'{line}\n')
            outfile.write('END\n')


def read_struct_buff(fileh, fmt):
    """Unpack from a file handle with a given format.

    Parameters
    ----------
    fileh : file object
        The file handle to unpack from.
    fmt : string
        The format to use for unpacking.

    Returns
    -------
    out : tuple
        The unpacked elements according to the given format.

    Raises
    ------
    EOFError
        We will raise an EOFError if `fileh.read()` attempts to read
        past the end of the file.

    """
    buff = fileh.read(struct.calcsize(fmt))
    if not buff:
        raise EOFError
    return struct.unpack(fmt, buff)


def is_double(header):
    """Determine if we should use double precision.

    This method determined the precision to use when reading
    the TRR file. This is based on the header read for a given
    frame which defines the sizes of certain "fields" like the box
    or the positions. From this size, the precision can be obtained.

    Parameters
    ----------
    header : dict
        The header read from the TRR file.

    Returns
    -------
    out : boolean
        True if we should use double precision.

    """
    key_order = ('box_size', 'x_size', 'v_size', 'f_size')
    size = 0
    for key in key_order:
        if header[key] != 0:
            if key == 'box_size':
                size = int(header[key] / _DIM**2)
                break
            size = int(header[key] / (header['natoms'] * _DIM))
            break
    if size not in (_SIZE_FLOAT, _SIZE_DOUBLE):
        raise ValueError('Could not determine size!')
    return size == _SIZE_DOUBLE


def skip_trr_data(fileh, header):
    """Skip coordinates/box data etc.

    This method is used when we want to skip a data section in
    the TRR file. Rather than reading the data, it will use the
    size read in the header to skip ahead to the next frame.

    Parameters
    ----------
    fileh : file object
        The file handle for the file we are reading.
    header : dict
        The header read from the TRR file.

    """
    offset = sum([header[key] for key in TRR_DATA_ITEMS])
    fileh.seek(offset, 1)


def read_trr_data(fileh, header):
    """Read box, coordinates etc. from a TRR file.

    Parameters
    ----------
    fileh : file object
        The file handle for the file we are reading.
    header : dict
        The header read from the file.

    Returns
    -------
    data : dict
        The data we read from the file. It may contain the following
        keys if the data was found in the frame:

        - ``box`` : the box matrix,
        - ``vir`` : the virial matrix,
        - ``pres`` : the pressure matrix,
        - ``x`` : the coordinates,
        - ``v`` : the velocities, and
        - ``f`` : the forces

    """
    data = {}
    endian = header['endian']
    double = header['double']
    for key in ('box', 'vir', 'pres'):
        header_key = f'{key}_size'
        if header[header_key] != 0:
            data[key] = read_matrix(fileh, endian, double)
    for key in ('x', 'v', 'f'):
        header_key = f'{key}_size'
        if header[header_key] != 0:
            data[key] = read_coord(fileh, endian, double,
                                   header['natoms'])
    return data


def read_trr_file(filename, read_data=True):
    """Yield frames from a TRR file."""
    with open(filename, 'rb') as infile:
        while True:
            try:
                header, _ = read_trr_header(infile)
                if read_data:
                    data = read_trr_data(infile, header)
                else:
                    skip_trr_data(infile, header)
                    data = None
                yield header, data
            except EOFError:
                return None, None
            except struct.error:
                logger.warning(
                    'Could not read a frame from the TRR file. Aborting!'
                )
                return None, None


def read_matrix(fileh, endian, double):
    """Read a matrix from the TRR file.

    Here, we assume that the matrix will be of
    dimensions (_DIM, _DIM).

    Parameters
    ----------
    fileh : file object
        The file handle to read from.
    endian : string
        Determines the byte order.
    double : boolean
        If true, we will assume that the numbers
        were stored in double precision.

    Returns
    -------
    mat : numpy.array
        The matrix as an array.

    """
    if double:
        fmt = f'{endian}{_DIM**2}d'
    else:
        fmt = f'{endian}{_DIM**2}f'
    read = read_struct_buff(fileh, fmt)
    mat = np.zeros((_DIM, _DIM))
    for i in range(_DIM):
        for j in range(_DIM):
            mat[i, j] = read[i * _DIM + j]
    return mat


def read_coord(fileh, endian, double, natoms):
    """Read a coordinate section from the TRR file.

    This method will read the full coordinate section from a TRR
    file. The coordinate section may be positions, velocities or
    forces.

    Parameters
    ----------
    fileh : file object
        The file handle to read from.
    endian : string
        Determines the byte order.
    double : boolean
        If true, we will assume that the numbers
        were stored in double precision.
    natoms : int
        The number of atoms we have stored coordinates for.

    Returns
    -------
    mat : numpy.array
        The coordinates as a numpy array. It will have
        ``natoms`` rows and ``_DIM`` columns.

    """
    if double:
        fmt = f'{endian}{natoms * _DIM}d'
    else:
        fmt = f'{endian}{natoms * _DIM}f'
    read = read_struct_buff(fileh, fmt)
    mat = np.array(read)
    mat.shape = (natoms, _DIM)
    return mat


def read_xvg_file(filename):
    """Return data in xvg file as numpy array."""
    data = []
    legends = []
    with open(filename, 'r', encoding='utf-8') as fileh:
        for lines in fileh:
            if lines.startswith('@ s') and lines.find('legend') != -1:
                legend = lines.split('legend')[-1].strip()
                legend = legend.replace('"', '')
                legends.append(legend.lower())
            else:
                if lines.startswith('#') or lines.startswith('@'):
                    pass
                else:
                    data.append([float(i) for i in lines.split()])
    data = np.array(data)
    data_dict = {'step': np.arange(tuple(data.shape)[0])}
    for i, key in enumerate(legends):
        data_dict[key] = data[:, i+1]
    return data_dict


def get_data(fileh, header):
    """Read data from the TRR file.

    Parameters
    ----------
    fileh : file object
        The file we are reading.
    header : dict
        The previously read header. Contains sizes and what to read.

    Returns
    -------
    data : dict
        The data read from the file.
    data_size : integer
        The size of the data read.

    """
    data_size = sum([header[key] for key in TRR_DATA_ITEMS])
    data = read_trr_data(fileh, header)
    return data, data_size


def read_remaining_trr(filename, fileh, start):
    """Read remaining frames from the TRR file.

    Parameters
    ----------
    filename : string
        The file we are reading from.
    fileh : file object
        The file object we are reading from.
    start : integer
        The current position we are at.

    Yields
    ------
    out[0] : string
        The header read from the file
    out[1] : dict
        The data read from the file.
    out[2] : integer
        The size of the data read.

    """
    stop = False
    bytes_read = start
    bytes_total = os.path.getsize(filename)
    logger.debug('Reading remaing data from: %s', filename)
    while not stop:
        if bytes_read >= bytes_total:
            stop = True
            continue
        header = None
        new_bytes = bytes_read
        try:
            header, new_bytes = read_trr_header(fileh)
        except EOFError:  # pragma: no cover
            # Just assume that we have reached the end of the
            # file and we just stop here. It should not be reached,
            # kept for safety
            stop = True
            continue
        if header is not None:
            bytes_read += new_bytes
            try:
                data, new_bytes = get_data(fileh, header)
                if data is not None:
                    bytes_read += new_bytes
                    yield header, data, bytes_read
            except EOFError:  # pragma: no cover
                # Hopefully, this code should not be reached.
                # kept for safety
                stop = True
                continue


def read_gromacs_lines(lines):
    """Read and parse GROMACS GRO data.

    This method will read a GROMACS file and yield the different
    snapshots found in the file.

    Parameters
    ----------
    lines : iterable
        Some lines of text data representing a GROMACS GRO file.

    Yields
    ------
    out : dict
        This dict contains the snapshot.

    """
    lines_to_read = 0
    snapshot = {}
    read_natoms = False
    gro = (5, 5, 5, 5, 8, 8, 8, 8, 8, 8)
    gro_keys = ('residunr', 'residuname', 'atomname', 'atomnr',
                'x', 'y', 'z', 'vx', 'vy', 'vz')
    gro_type = (int, str, str, int, float, float, float, float, float, float)
    for line in lines:
        if read_natoms:
            read_natoms = False
            lines_to_read = int(line.strip()) + 1
            continue  # just skip to next line
        if lines_to_read == 0:  # new snapshot
            if snapshot:
                _add_matrices_to_snapshot(snapshot)
                yield snapshot
            snapshot = {'header': line.strip()}
            read_natoms = True
        elif lines_to_read == 1:  # read box
            snapshot['box'] = np.array(
                [float(i) for i in line.strip().split()]
            )
            lines_to_read -= 1
        else:  # read atoms
            lines_to_read -= 1
            current = 0
            for i, key, gtype in zip(gro, gro_keys, gro_type):
                val = line[current:current+i].strip()
                if not val:
                    # This typically happens if we try to read velocities
                    # and they are not present in the file.
                    break
                value = gtype(val)
                current += i
                try:
                    snapshot[key].append(value)
                except KeyError:
                    snapshot[key] = [value]
    if snapshot:
        _add_matrices_to_snapshot(snapshot)
        yield snapshot


def _add_matrices_to_snapshot(snapshot):
    """Extract positions and velocities as matrices from GROMACS.

    The extracted positions and velocities will be added to the given
    snapshot.

    Parameters
    ----------
    snapshot : dict
        This dict contains the data read from the GROMACS file.

    Returns
    -------
    xyz : numpy.array
        The positions as an array, (N, 3).
    vel : numpy.array
        The velocities as an array, (N, 3).

    """
    xyz = np.zeros((len(snapshot['atomnr']), 3))
    for i, key in enumerate(('x', 'y', 'z')):
        if key in snapshot:
            xyz[:, i] = snapshot[key]
    vel = np.zeros_like(xyz)
    for i, key in enumerate(('vx', 'vy', 'vz')):
        if key in snapshot:
            vel[:, i] = snapshot[key]
    snapshot['xyz'] = xyz
    snapshot['vel'] = vel
    return xyz, vel


