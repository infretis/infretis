from infretis.classes.box import create_box, box_from_restart
from infretis.classes.particles import particles_from_restart, Particles

import colorama
import numpy as np
from collections import deque
from copy import copy as copy0
import os

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_PRINT_COLORS = {
    'error': colorama.Fore.RED,
    'info': colorama.Fore.BLUE,
    'warning': colorama.Fore.YELLOW,
    'message': colorama.Fore.CYAN,
    'success': colorama.Fore.GREEN
}

CAL = 4184
CONSTANTS = { }
CONSTANTS['kB'] = {
    'eV/K': 8.61733e-05,
    'J/K': 1.38065e-23,
    'kJ/mol/K': 0.00831446 }
CONSTANTS['kB']['J/mol/K'] = CONSTANTS['kB']['kJ/mol/K'] * 1000
CONSTANTS['kB']['kJ/K'] = CONSTANTS['kB']['J/K'] / 1000
CONSTANTS['kB']['kcal/K'] = CONSTANTS['kB']['J/K'] / CAL
CONSTANTS['kB']['kcal/mol/K'] = CONSTANTS['kB']['J/mol/K'] / CAL
CONSTANTS['NA'] = {
    '1/mol': 6.02214e+23 }
CONSTANTS['c0'] = {
    'm/s': 2.99792e+08 }
CONSTANTS['mu0'] = {
    'H/m': 4 * np.pi * 1e-07 }
CONSTANTS['e'] = {
    'C': 1.60218e-19 }
CONSTANTS['e0'] = {
    'F/m': 1 / CONSTANTS['mu0']['H/m'] * CONSTANTS['c0']['m/s'] ** 2 }
CONSTANTS['kB']['lj'] = 1
CONSTANTS['kB']['reduced'] = 1
CONSTANTS['kB']['si'] = CONSTANTS['kB']['J/K']
CONSTANTS['kB']['real'] = CONSTANTS['kB']['kcal/mol/K']
CONSTANTS['kB']['metal'] = CONSTANTS['kB']['eV/K']
CONSTANTS['kB']['au'] = 1
CONSTANTS['kB']['electron'] = 3.16682e-06
CONSTANTS['kB']['gromacs'] = CONSTANTS['kB']['kJ/mol/K']
CONSTANTS['kB']['cp2k'] = 3.16682e-06
DIMENSIONS = { 'length', 'mass', 'time', 'energy', 'velocity', 'charge',
               'temperature', 'pressure', 'force'}
CONVERT = {key: {} for key in DIMENSIONS}
UNITS = {key: {} for key in DIMENSIONS}

UNITS['length'] = { 'A', 'nm', 'bohr', 'm'}
CONVERT['length'][('A', 'nm')] = 0.1
CONVERT['length'][('A', 'bohr')] = 1.88973
CONVERT['length'][('A', 'm')] = 1e-10
UNITS['mass'] = {
    'g/mol',
    'g',
    'kg'}
CONVERT['mass'][('g', 'kg')] = 0.001
CONVERT['mass'][('g/mol', 'g')] = 1 / CONSTANTS['NA']['1/mol']
CONVERT['mass'][('g/mol', 'kg')] = CONVERT['mass'][('g', 'kg')] / CONSTANTS['NA']['1/mol']
UNITS['time'] = {
    's',
    'ps',
    'fs',
    'ns',
    'us',
    'ms'}
CONVERT['time'][('s', 'ps')] = 1e+12
CONVERT['time'][('s', 'fs')] = 1e+15
CONVERT['time'][('s', 'ns')] = 1e+09
CONVERT['time'][('s', 'us')] = 1e+06
CONVERT['time'][('s', 'ms')] = 1000
UNITS['energy'] = {
    'kcal',
    'kcal/mol',
    'J',
    'J/mol',
    'kJ/mol',
    'eV',
    'hartree'}
CONVERT['energy'][('kcal', 'kcal/mol')] = CONSTANTS['NA']['1/mol']
CONVERT['energy'][('kcal', 'J')] = CAL
CONVERT['energy'][('kcal', 'J/mol')] = CONSTANTS['NA']['1/mol'] * CONVERT['energy'][('kcal', 'J')]
CONVERT['energy'][('kcal', 'kJ/mol')] = CONVERT['energy'][('kcal', 'J/mol')] * 0.001
CONVERT['energy'][('kcal', 'eV')] = CONSTANTS['kB']['eV/K'] / CONSTANTS['kB']['kcal/K']
CONVERT['energy'][('kcal', 'hartree')] = CONVERT['energy'][('kcal', 'eV')] * 0.0367493
UNITS['velocity'] = {
    'm/s',
    'nm/ps',
    'A/fs',
    'A/ps'}
CONVERT['velocity'][('m/s', 'nm/ps')] = 0.001
CONVERT['velocity'][('m/s', 'A/fs')] = 1e-05
CONVERT['velocity'][('m/s', 'A/ps')] = 0.01
UNITS['charge'] = {
    'e',
    'C'}
CONVERT['charge'][('e', 'C')] = CONSTANTS['e']['C']
CONVERT['charge'][('C', 'e')] = 1 / CONSTANTS['e']['C']
UNITS['pressure'] = {
    'Pa',
    'bar',
    'atm'}
CONVERT['pressure'][('Pa', 'bar')] = 1e-05
CONVERT['pressure'][('Pa', 'atm')] = 9.86923e-06
UNITS['temperature'] = {
    'K'}
UNITS['force'] = {
    'N',
    'pN',
    'dyn'}
CONVERT['force'][('N', 'pN')] = 1e+12
CONVERT['force'][('N', 'dyn')] = 100000
for i in DIMENSIONS:
    for j in UNITS[i]:
        CONVERT[i][(j, j)] = 1
UNIT_SYSTEMS = {
    'lj': { },
    'real': { },
    'metal': { },
    'au': { },
    'electron': { },
    'si': { },
    'gromacs': { },
    'reduced': { } }
UNIT_SYSTEMS['lj'] = {
    'length': (3.405, 'A'),
    'energy': (119.8, 'kB'),
    'mass': (39.948, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['reduced'] = {
    'length': (1, 'A'),
    'energy': (1, 'kB'),
    'mass': (1, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['real'] = {
    'length': (1, 'A'),
    'energy': (1, 'kcal/mol'),
    'mass': (1, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['metal'] = {
    'length': (1, 'A'),
    'energy': (1, 'eV'),
    'mass': (1, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['au'] = {
    'length': (1, 'bohr'),
    'energy': (1, 'hartree'),
    'mass': (9.10938e-31, 'kg'),
    'charge': 'e' }
UNIT_SYSTEMS['electron'] = {
    'length': (1, 'bohr'),
    'energy': (1, 'hartree'),
    'mass': (1, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['si'] = {
    'length': (1, 'm'),
    'energy': (1, 'J'),
    'mass': (1, 'kg'),
    'charge': 'e' }
UNIT_SYSTEMS['gromacs'] = {
    'length': (1, 'nm'),
    'energy': (1, 'kJ/mol'),
    'mass': (1, 'g/mol'),
    'charge': 'e' }
UNIT_SYSTEMS['cp2k'] = {
    'length': (1, 'A'),
    'energy': (1, 'hartree'),
    'mass': (9.10938e-31, 'kg'),
    'charge': 'e' }

class System:
    """This class defines a generic system for simulations.

    Attributes
    ----------
    box : object like :py:class:`.Box`
        Defines the simulation box.
    temperature : dict
        This dictionary contains information on the temperature. The
        following information is stored:

        * `set`: The set temperature, ``T``, (if any).
        * `beta`: The derived property ``1.0/(k_B*T)``.
        * `dof`: Information about the degrees of freedom for the
          system.
    order : tuple
        The order parameter(s) for the current state of the system (if
        they have been calculated).
    particles : object like :py:class:`.Particles`
        Defines the particle list which represents the particles and the
        properties of the particles (positions, velocities, forces etc.).
    post_setup : list of tuples
        This list contains extra functions that should be called when
        preparing to run a simulation. This is typically functions that
        should only be called after the system is fully set up. The
        tuples should correspond to ('function', args) where
        such that ``system.function(*args)`` can be called.
    units : string
        Units to use for the system/simulation. Should match the defined
        units in :py:mod:`pyretis.core.units`.

    """

    def __init__(self, units='lj', box=None, temperature=None):
        """Initialise the system.

        Parameters
        ----------
        units : string, optional
            The system of units to use in the simulation box.
        box : object like :py:class:`.Box`, optional
            This variable represents the simulation box. It is used to
            determine the number of dimensions.
        temperature : float, optional
            The (desired) temperature of the system, if applicable.

        Note
        ----
        `self.temperature` is defined as a dictionary. This is just
        because it's convenient to include information about the
        degrees of freedom of the system here.

        """
        self.units = units
        self.temperature = {'set': temperature, 'dof': None, 'beta': None}
        self.box = box
        self._adjust_dof_according_to_box()
        self.particles = None
        self.post_setup = []
        self.order = None
        self.temperature['beta'] = self.calculate_beta()

    def adjust_dof(self, dof):
        """Adjust the degrees of freedom to neglect in the system.

        Parameters
        ----------
        dof : numpy.array
            The degrees of freedom to neglect, in addition to the ones
            we already have neglected.

        """
        if self.temperature['dof'] is None:
            self.temperature['dof'] = np.array(dof)
        else:
            self.temperature['dof'] += np.array(dof)

    def _adjust_dof_according_to_box(self):
        """Adjust the dof according to the box connected to the system.

        For each 'True' in the periodic settings of the box, we subtract
        one degree of freedom for that dimension.

        """
        try:
            dof = []
            all_false = True
            for peri in self.box.periodic:
                dof.append(1 if peri else 0)
                all_false = all_false and not peri
            # If all items in self.box.periodic are false, then we
            # will not bother setting the dof to just zeros
            if not all_false:
                self.adjust_dof(dof)
            return True
        except AttributeError:
            return False

    def get_boltzmann(self):
        """Return the Boltzmann constant in correct units for the system.

        Returns
        -------
        out : float
            The Boltzmann constant.

        """
        return CONSTANTS['kB'][self.units]

    def get_dim(self):
        """Return the dimensionality of the system.

        The value is obtained from the box. In other words, it is the
        box object that defines the dimensionality of the system.

        Returns
        -------
        out : integer
            The number of dimensions of the system.

        """
        try:
            return self.box.dim
        except AttributeError:
            logger.warning(
                'Box dimensions are not set. Setting dimensions to "1"'
            )
            return 1

    def calculate_beta(self, temperature=None):
        r"""Return the so-called beta factor for the system.

        Beta is defined as :math:`\beta = 1/(k_\text{B} \times T)`
        where :math:`k_\text{B}` is the Boltzmann constant and the
        temperature `T` is either specified in the parameters or assumed
        equal to the set temperature of the system.

        Parameters
        ----------
        temperature : float, optional
            The temperature of the system. If the temperature
            is not given, `self.temperature` will be used.

        Returns
        -------
        out : float
            The calculated beta factor, or None if no temperature data
            is available.

        """
        if temperature is None:
            if self.temperature['set'] is None:
                return None
            temperature = self.temperature['set']
        return 1.0 / (temperature * CONSTANTS['kB'][self.units])

    def add_particle(self, pos, vel=None, force=None,
                     mass=1.0, name='?', ptype=0):
        """Add a particle to the system.

        Parameters
        ----------
        pos : numpy.array,
            Position of the particle.
        vel : numpy.array, optional
            The velocity of the particle. If not given numpy.zeros will be
            used.
        force : numpy.array, optional
            Force on the particle. If not given np.zeros will be used.
        mass : float, optional
            Mass of the particle, the default is 1.0.
        name : string, optional
            Name of the particle, the default is '?'.
        ptype : integer, optional
            Particle type, the default is 0.

        Returns
        -------
        out : None
            Does not return anything, but updates :py:attr:`~particles`.

        """
        if vel is None:
            vel = np.zeros_like(pos)
        if force is None:
            force = np.zeros_like(pos)
        self.particles.add_particle(pos, vel, force, mass=mass,
                                    name=name, ptype=ptype)

    def generate_velocities(self, rgen=None, seed=0, momentum=True,
                            temperature=None, distribution='maxwell'):
        """Set velocities for the particles according to a given temperature.

        The temperature can be specified, or it can be taken from
        `self.temperature['set']`.

        Parameters
        ----------
        rgen : string, optional
            This string can be used to select a particular random
            generator. Typically this is only useful for testing.
        seed : int, optional
            The seed for the random generator.
        momentum : boolean, optional
            Determines if the momentum should be reset.
        temperature : float, optional
            The desired temperature to set.
        distribution : str, optional
            Selects a distribution for generating the velocities.

        Returns
        -------
        out : None
            Does not return anything, but updates
            `system.particles.vel`.

        """
        rgen_settings = {'seed': seed, 'rgen': rgen}
        rgen = create_random_generator(rgen_settings)
        if temperature is None:
            temperature = self.temperature['set']
        dof = self.temperature['dof']
        if distribution.lower() == 'maxwell':
            rgen.generate_maxwellian_velocities(self.particles,
                                                CONSTANTS['kB'][self.units],
                                                temperature,
                                                dof, momentum=momentum)
        else:
            msg = 'Distribution "{}" not defined! Velocities not set!'
            msg = msg.format(distribution)
            logger.error(msg)

    def calculate_temperature(self):
        """Calculate the temperature of the system.

        It is included here for convenience since the degrees of freedom
        are easily accessible and it's a very common calculation to
        perform, even though it might be cleaner to include it as a
        particle function.

        Returns
        -------
        out : float
            The temperature of the system.

        """
        dof = self.temperature['dof']
        _, temp, _ = calculate_kinetic_temperature(self.particles,
                                                   CONSTANTS['kB'][self.units],
                                                   dof=dof)
        return temp

    def extra_setup(self):
        """Perform extra set-up for the system.

        The extra set-up will typically be tasks that can only
        be performed after the system is fully set-up, for instance
        after the force field is properly defined.
        """
        for func_name, args in self.post_setup:
            func = getattr(self, func_name, None)
            if func is not None:
                func(*args)

    def rescale_velocities(self, energy, external=False):
        """Re-scale the kinetic energy to a given total energy.

        Parameters
        ----------
        energy : float
            The desired energy.
        energy : boolean, optional
            If True, self.particles.vpot will be used as the potential energy.

        Returns
        -------
        None, but updates the velocities of the particles.

        """
        if not external:
            vpot = self.potential()
        else:
            vpot = self.particles.vpot
        ekin, _ = calculate_kinetic_energy(self.particles)
        ekin_new = energy - vpot
        if ekin_new < 0:
            logger.warning(('Can not re-scale velocities. '
                            'Target energy: %f, Potential: %f'), energy, vpot)
        else:
            logger.debug('Re-scaled energies to ekin: %f', ekin_new)
            alpha = np.sqrt(ekin_new / ekin)
            self.particles.vel = self.particles.vel * alpha

    def restart_info(self):
        """Return a dictionary with restart information."""
        info = {}
        for attr in ('units', 'temperature', 'post_setup', 'order'):
            info[attr] = getattr(self, attr, None)
        # Collect some more info:
        try:
            info['box'] = self.box.restart_info()
        except AttributeError:
            pass
        try:
            info['particles'] = self.particles.restart_info()
        except AttributeError:
            pass
        return info

    def load_restart_info(self, info):
        """Load restart information.

        Parameters
        ----------
        info : dict
            The dictionary with the restart information, should be
            similar to the dict produced by :py:func:`.restart_info`.

        """
        for attr in ('units', 'temperature', 'post_setup', 'order'):
            if attr in info:
                setattr(self, attr, info[attr])

        self.box = box_from_restart(info)
        self.particles = particles_from_restart(info)

    def update_box(self, length):
        """Update the system box, create if needed.

        Parameters
        ----------
        length : numpy.array, list or iterable.
            The box vectors represented as a list.

        """
        if self.box is None:
            self.box = create_box(cell=length)
        else:
            self.box.update_size(length)

    def copy(self):
        """Return a copy of the system.

        This copy is useful for storing snapshots obtained during
        a simulation.

        Returns
        -------
        out : object like :py:class:`.System`
            A copy of the system.

        """
        system_copy = System()
        for attr in {'units', 'temperature', 'post_setup', 'order'}:
            try:
                val = getattr(self, attr)
                if val is None:
                    setattr(system_copy, attr, None)
                else:
                    setattr(system_copy, attr, copy0(val))
            except AttributeError:
                logger.warning(
                    'Missing attribute "%s" when copying system', attr
                )
        for attr in ('box', 'particles'):
            val = getattr(self, attr)
            if val is None:
                setattr(system_copy, attr, None)
            else:
                setattr(system_copy, attr, val.copy())
        # We do not copy the force field here and assume that
        # systems that are copies should share the same force field,
        # that is, if the force field were to change for some reason,
        # then that change should be mediated to all copies of the
        # system.
        return system_copy

    def __eq__(self, other):
        """Compare two system objects."""
        # Note: We do not check the order parameter here as this
        # depends on the choice of the order parameter function.
        attrs = ('units', 'post_setup', 'box', 'particles')
        check = compare_objects(self, other, attrs, numpy_attrs=None)
        # todo To be re-introduced if forcefields get a __eq_ function
        # check = check and self.forcefield is other.forcefield
        # For the temperature, one key may give some trouble:
        check = check and len(self.temperature) == len(other.temperature)
        for key in ('set', 'beta'):
            check = check and self.temperature[key] == other.temperature[key]
        check = check and numpy_allclose(self.temperature['dof'],
                                         other.temperature['dof'])
        return check

    def __ne__(self, other):
        """Check if two systems are not equal."""
        return not self == other

    def __str__(self):
        """Just print some basic info about the system."""
        msg = ['PyRETIS System',
               'Order parameter: {}'.format(self.order),
               'Box:']
        msg.append('{}'.format(self.box))
        msg.append('Particles:')
        msg.append('{}'.format(self.particles))
        return '\n'.join(msg)

def print_to_screen(txt=None, level=None):  # pragma: no cover
    """Print output to standard out.

    This method is included to ensure that output from PyRETIS to the
    screen is written out in a uniform way across the library and
    application(s).

    Parameters
    ----------
    txt : string, optional
        The text to write to the screen.
    level : string, optional
        The level can be used to color the output.

    """
    if txt is None:
        print()
    else:
        out = '{}'.format(txt)
        color = _PRINT_COLORS.get(level, None)
        if color is None:
            print(out)
        else:
            print(color + out)
