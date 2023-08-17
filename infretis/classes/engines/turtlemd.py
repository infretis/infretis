# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
# Distributed under the LGPLv2.1+ License. See LICENSE for more info.
"""A TurtleMD integrator interface.

This module defines a class for using the TurtleMD.

Important classes defined here
------------------------------

CP2KEngine (:py:class:`.CP2KEngine`)
    A class responsible for interfacing CP2K.
"""
from collections import defaultdict

import logging
import os
import numpy as np
from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    read_xyz_file,
    write_xyz_trajectory,
    convert_snapshot
)
from numpy.random import default_rng
from turtlemd.potentials.lennardjones import LennardJonesCut
from turtlemd.potentials.jax_well import *
from turtlemd.system.particles import generate_maxwell_velocities, Particles
from turtlemd.system.box import Box
from turtlemd.system.system import System
from turtlemd.integrators import VelocityVerlet
from turtlemd.integrators import LangevinIntertia
from turtlemd.simulation import MDSimulation
from infretis.classes.engines.cp2k import (
    kinetic_energy,
    reset_momentum
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class TurtleMDEngine(EngineBase):
    """

    """
    def __init__(self, turtlemd, input_path, timestep, subcycles):
        self.temperature=1/2.5*100
        self.boltzmann = 1
        self.beta = 1/self.temperature 
        self.timestep = 0.1
        self.gamma = 2.5
        self.subcycles = 1
        self.dim = 2
        self.mass = np.array([1])

        self.particles = Particles(dim=self.dim)
        self.box = Box(periodic=[False,False])
        self.particles.add_particle(np.zeros(2))
        self.particles.pos = np.array([[-0.9, 0.]])
        self.potentials = [MyWell()]
        self.system = System(self.box, self.particles, self.potentials)
        self.integrator = LangevinIntertia


        super().__init__('TurtleMD internal engine', self.timestep,
                         self.subcycles)

    def old__init__(self, turtlemd, input_path, timestep, subcycles): 
        """
        """
        mass = 1.008
        name = "H"
        self.timestep = timestep # ps
        self.subcycles = subcycles
        super().__init__('TurtleMD internal engine', timestep,
                         subcycles)
        self.boltzmann = 8.3145e-3 # kJ/mol
        self.temperature = 300 # K 
        self.beta = 1/(self.boltzmann*self.temperature)
        positions = [
            [0.303, 0.400, 0.372],
            [0.370, 0.424, 0.443],
        ]

        particles = Particles(dim=self.dim)
        for pos in positions:
            particles.add_particle(pos, mass=mass, name=name)

        potentials = [LennardJonesCut(dim=3, shift=False, mixing="geometric")]
        parameters = {
            0: {"sigma": 0.3, "epsilon": 25.0, "rcut": 1.2},
            1: {"sigma": 0.3, "epsilon": 25.0, "rcut": 1.2},
        }
        potentials[0].set_parameters(parameters)

        generate_maxwell_velocities(
            particles,
            rgen=default_rng(),
            temperature=self.temperature,
            boltzmann= self.boltzmann,
            momentum=True,
        )

        box = Box(
            low=[0, 0, 0], high=[3.0, 3.0, 3.0], periodic=[True, True, True]
        )

        self.dim = len(box.low)
        self.mass = particles.mass
        self.particles = particles
        self.potentials = potentials
        self.box = box
        self.system = System(box=box, particles=particles, potentials=potentials)
        self.integrator = VelocityVerlet

    def old__init__(self, turtlemd, input_path, timestep, subcycles): 
        """
        """

        mass = 1.008 # g/mol
        name = "H"
        self.timestep = timestep # ps
        self.subcycles = subcycles
        super().__init__('TurtleMD internal engine', timestep,
                         subcycles)
        self.boltzmann = 8.3145e-3 # kJ/mol
        self.temperature = 300 # K 
        self.beta = 1/(self.boltzmann*self.temperature)
        positions = [
            [0.303, 0.400, 0.372],
            [0.370, 0.424, 0.443],
            [0.432, 0.353, 0.408],
            [0.496, 0.424, 0.378],
        ]

        particles = Particles(dim=self.dim)
        for pos in positions:
            particles.add_particle(pos, mass=mass, name=name)

        generate_maxwell_velocities(
            particles,
            rgen=default_rng(),
            temperature=self.temperature,
            boltzmann= self.boltzmann,
            momentum=True,
        )

        box = Box(
            low=[0, 0, 0], high=[0.8, 0.8, 0.8], periodic=[True, True, True]
        )

        # force field
        potentials = [
            BondedInteractions(
                bonds=[
                    # k         b0   i  j
                    [200000.0, 0.1, 0, 1],
                    [200000.0, 0.1, 1, 2],
                    [200000.0, 0.1, 2, 3],
                ],
                angles=[
                    # k      ang0           i  j  k
                    [400.0, 1.57079632679, 0, 1, 2],
                    [400.0, 1.57079632679, 1, 2, 3],
                ],
                dihedrals=[
                    # k    ang0           n    i  j  k  l
                    [8.0, 0.78539816339, 2.0, 0, 1, 2, 3]
                ],
            )
        ]

        self.dim = len(box.low)
        self.mass = particles.mass
        self.particles = particles
        self.potentials = potentials
        self.box = box
        self.system = System(box=box, particles=particles, potentials=potentials)
        self.integrator = VelocityVerlet

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
                    logger.debug('TurtleMD will overwrite %s', out_file)
                write_xyz_trajectory(out_file, xyz, vel, names, box,
                                     append=False)
                return
        logger.error('TurtleMD could not extract index %i from %s!',
                     idx, traj_file)

    def _propagate_from(self, name, path, system, ens_set, msg_file, reverse=False):
        """
        We assume the following:
            * Box does not change (constant volume simulation)
            * Box is orthogonal
        """
        status = f'propagating with TurtleMD (reverse = {reverse})'
        interfaces = ens_set['interfaces']
        logger.debug(status)
        success = False
        left, _, right = interfaces
        # Get positions and velocities from the input file.
        initial_conf = system.config[0]
        # these variables will be used later
        box, pos, vel, atoms = self._read_configuration(initial_conf)
        # inititalize turtlemd system
        particles = Particles(dim=self.dim)
        for i in range(self.particles.npart):
            particles.add_particle(pos[i][:self.dim], vel=vel[i][:self.dim],
                                   mass=self.particles.mass[i], name=self.particles.name[i]
                                   )
        tmd_system = System(box = self.box, particles = particles,
                            potentials = self.potentials
                            )
        tmd_simulation = MDSimulation(
            system=tmd_system, integrator=self.integrator(self.timestep, gamma=self.gamma, beta=self.beta, seed = np.random.randint(42000)),
            steps=path.maxlen*self.subcycles
            )
        order = self.calculate_order(system, xyz=pos,
                                    vel=vel, box=tmd_system.box.length)
        
        traj_file = os.path.join(self.exe_dir, f'{name}.{self.ext}')
        # Create a message file with some info about this run:
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        msg_file.write(f'# Trajectory file is: {traj_file}')
        logger.debug('Running TurtleMD')
        step_nr = 0
        # dict for storing ene  rgies
        thermo = defaultdict(list)
        # loop over n subcycles
        # The first step of the loop is the initial phase point, i.e., for i=0
        # turtlemd does not integrate the equations of motion, it just
        # returns the initial system
        for i,step in enumerate(tmd_simulation.run()):
            tmd_system.particles.vel = reset_momentum(tmd_system.particles.vel, tmd_system.particles.mass)
            if (i)%(self.subcycles)==0:
                thermoi = step.thermo(self.boltzmann)
                for key, val in thermoi.items():
                    thermo[key].append(val)
                # update coordinates, velocities and box
                # for the relevant dimensions. We need this here
                # because we use xyz format for trajecories, which has 3 dimensions
                # for coords, vel and the box.
                pos[:,:self.dim] = tmd_system.particles.pos
                vel[:,:self.dim] = tmd_system.particles.vel
                box[:self.dim] = tmd_system.box.length  
                write_xyz_trajectory(traj_file, pos, vel,
                                        atoms, box, step = step_nr)
                order = self.calculate_order(system, xyz=tmd_system.particles.pos,
                                             vel=tmd_system.particles.vel,
                                             box=tmd_system.box.length)
                msg_file.write(f'{step_nr} {" ".join([str(j) for j in order])}')
                snapshot = {'order': order, 'config': (traj_file, step_nr),
                            'vel_rev': reverse}
                phase_point = self.snapshot_to_system(system, snapshot)
                status, success, stop, add = self.add_to_path(path, phase_point,
                                                    left, right)
                
                if stop:    
                    logger.debug('TurtleMD propagation ended at %i. Reason: %s',
                    step_nr, status)
                    break
                step_nr += 1

        msg_file.write('# Propagation done.')
        ekin = np.array(thermo['ekin'])*tmd_system.particles.npart
        vpot = np.array(thermo['vpot'])*tmd_system.particles.npart
        path.update_energies(ekin,
                             vpot)
        return success, status

    def step(self, system, name):
        raise NotImplementedError("Surprise, step not implemented!")

    @staticmethod
    def _read_configuration(filename):
        """
        Read TurtleMD output configuration.

        This method is used when we calculate the order parameter.

        Parameters
        ----------
        filename : string
            The file to read the configuration from.

        Returns
        -------
        box : numpy.array
            The box dimensions if we manage to read it.
        xyz : numpy.arrayo
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
        write_xyz_trajectory(outfile, xyz, -1.0*vel, names, box, append=False)

    def modify_velocities(self, system, vel_settings=None):
        """
        Modfy the velocities of all particles. Note that default
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
    	# to-do: retrieve system.vpot from previous energy file.
        print(rescale, system.vpot)
        print(vel_settings)
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
        else:
            dvel, _ = rgen.draw_maxwellian_velocities(vel, mass, beta, sigma_v=vel_settings['sigma_v'])
            vel += dvel
        # make reset momentum the default
        if vel_settings.get('zero_momentum', True):
            vel = reset_momentum(vel, mass)
        if do_rescale:
            #system.rescale_velocities(rescale, external=True)
            raise NotImplementedError("Option 'rescale_energy' is not implemented yet.")
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
        return dek, kin_new
