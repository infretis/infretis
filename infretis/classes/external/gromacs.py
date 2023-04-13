from infretis.classes.engine import EngineBase
from infretis.classes.box import box_matrix_to_list
from infretis.core.gromacs import (read_gromacs_gro_file,
                                   write_gromacs_gro_file,
                                   read_gromos96_file,
                                   write_gromos96_file,
                                   read_struct_buff,
                                   read_trr_frame,
                                   read_trr_header,
                                   read_xvg_file,
                                   get_data,
                                   read_remaining_trr,)

from time import sleep
import signal
import subprocess
import os
import shlex
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
        super().__init__('GROMACS engine', timestep, subcycles)
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

    def _propagate_from(self, name, path, ensemble, msg_file, reverse=False):
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
        system = ensemble['system']
        interfaces = ensemble['interfaces']
        order_function = ensemble['order_function']
        logger.debug(status)
        success = False
        left, _, right = interfaces
        # Dumping of the initial config were done by the parent, here
        # we will just use it:
        initial_conf = system.particles.get_pos()[0]
        # Get the current order parameter:
        order = self.calculate_order(ensemble)
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
                system.particles.set_pos((trr_file, i))
                # Also provide the loaded positions since they are
                # available:
                system.particles.pos = data['x']
                system.particles.vel = data.get('v', None)
                if system.particles.vel is not None and reverse:
                    system.particles.vel *= -1
                length = box_matrix_to_list(data['box'])
                system.update_box(length)
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

    def modify_velocities(self, ensemble, vel_settings):
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
        system = ensemble['system']
        rescale = vel_settings.get('rescale_energy',
                                   vel_settings.get('rescale'))
        if rescale is not None and rescale is not False and rescale > 0:
            msgtxt = 'GROMACS engine does not support energy re-scale.'
            logger.error(msgtxt)
            raise NotImplementedError(msgtxt)
        kin_old = system.particles.ekin
        if vel_settings.get('aimless', False):
            pos = self.dump_frame(system)
            posvel, energy = self._prepare_shooting_point(pos)
            kin_new = energy['kinetic en.'][-1]
            system.particles.set_pos((posvel, None))
            system.particles.set_vel(False)
            system.particles.ekin = kin_new
            system.particles.vpot = energy['potential'][-1]
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

    def integrate(self, ensemble, steps, thermo='full'):
        """
        Perform several integration steps.

        This method will perform several integration steps using
        GROMACS. It will also calculate order parameter(s) and energy
        terms if requested.

        Parameters
        ----------
        ensemble: dict
            it contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.

        steps : integer
            The number of steps we are going to perform. Note that we
            do not integrate on the first step (e.g. step 0) but we do
            obtain the other properties. This is to output the starting
            configuration.
        thermo : string, optional
            Select the thermodynamic properties we are to obtain.

        Yields
        ------
        results : dict
            The results from a MD step. This contains the state of the system
            and order parameter(s) and energies (if calculated).

        """
        logger.debug('Integrating with GROMACS')
        # Dump the initial config:
        system = ensemble['system']
        order_function = ensemble.get('order_function')
        initial_file = self.dump_frame(system)
        self.energy_terms = self.select_energy_terms(thermo)
        if order_function:
            order = self.calculate_order(ensemble)
        else:
            order = None
        name = 'pyretis-gmx'
        # 1) Create mdp_file with updated number of steps:
        # Note the -1 here due do different numbering in GROMACS and PyRETIS.
        settings = {'nsteps': (steps - 1) * self.subcycles,
                    'continuation': 'no'}
        mdp_file = os.path.join(self.exe_dir, f'{name}.mdp')
        self._modify_input(self.input_files['input'], mdp_file, settings,
                           delim='=')
        # 2) Run GROMACS preprocessor:
        out_files = self._execute_grompp(mdp_file, initial_file, name)
        # Generate some names that will be created by mdrun:
        confout = f'{name}.{self.ext}'
        out_files['conf'] = confout
        out_files['cpt_prev'] = f'{name}_prev.cpt'
        for key in ('cpt', 'edr', 'log', 'trr'):
            out_files[key] = f'{name}.{key}'
        # Remove some of these files if present (e.g. left over from a
        # crashed simulation). This is so that GromacsRunner will not
        # start reading a .trr left from a previous simulation.
        remove = [val for key, val in out_files.items() if key != 'tpr']
        self._remove_files(self.exe_dir, remove)
        tpr_file = out_files['tpr']
        trr_file = os.path.join(self.exe_dir, out_files['trr'])
        edr_file = os.path.join(self.exe_dir, out_files['edr'])
        cmd = shlex.split(self.mdrun.format(tpr_file, name, confout))
        # 3) Fire off GROMACS mdrun:
        logger.debug('Executing GROMACS.')
        with GromacsRunner(cmd, trr_file, edr_file, self.exe_dir) as gro:
            for i, data in enumerate(gro.get_gromacs_frames()):
                system.particles.pos = data['x']
                system.particles.vel = data.get('v', None)
                length = box_matrix_to_list(data['box'])
                system.update_box(length)
                results = {}
                if order:
                    results['order'] = order
                if order_function:
                    order = order_function.calculate(system)
                time1 = (i * self.timestep * self.subcycles -
                         0.1 * self.timestep)
                time2 = ((i + 1) * self.timestep * self.subcycles +
                         0.1 * self.timestep)
                energy = self.get_energies(out_files['edr'], begin=time1,
                                           end=time2)
                results['thermo'] = self.rename_energies(energy)
                yield results
        logger.debug('GROMACS execution done.')
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

