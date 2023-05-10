from infretis.classes.external.gromacs import GromacsEngine
from infretis.classes.particles import Particles
from infretis.classes.system import System
from infretis.classes.randomgen import create_random_generator
from infretis.classes.tasks import OutputTask
from infretis.classes.fileio import FileIO
from infretis.classes.screen import ScreenOutput
from infretis.classes.formatter import (
    EnergyFormatter,
    OrderFormatter,
    CrossFormatter,
    SnapshotFormatter,
    ThermoTableFormatter,
    PathTableFormatter,
    EnergyPathFormatter,
    OrderPathFormatter,
    PathEnsembleFormatter,
    RETISResultFormatter,
    PathStorage,
)
from collections import deque
import os
import errno
import colorama
import re
import copy
import ast
import sys
import pickle
import numpy as np
import inspect
import importlib
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
SECTIONS = {}

PROGRAM_NAME = 'PyRETIS'
URL = 'http://www.pyretis.org'
GIT_URL = 'https://gitlab.com/pyretis/pyretis'
CITE = """
[1] A. Lervik, E. Riccardi and T. S. van Erp, J. Comput. Chem., 2017
    doi: https://dx.doi.org/10.1002/jcc.24900
[2] E. Riccardi, A. Lervik, S. Roet, O. Aarøen and T. S. van Erp,
    J. Comput. Chem., 2019, doi: https://dx.doi.org/10.1002/jcc.26112

"""
LOGO = r"""

  _______                                     _______________
 /   ___ \       _____  ____  ____________   / __        ___/
/_/\ \_/ /_  __  \  _ \/ __/ /_  _/ / ___/  /_/ / // // /
   /  __  / / /  / /_)  _/    / // /\ \        / // // / __
  / /   \ \/ /  / _  \ /__   / // /__\ \   ___/ // // /_/ /
 /_/    _\  /  /_/ |_|___/  /_//_/_____/  /______________/.beta
       /___/
"""
TITLE = f'{PROGRAM_NAME} input settings'
HEADING = '{}\n{}\nFor more info, please see: {}\nHave Fun!'
SECTIONS['heading'] = {'text': HEADING.format(TITLE, '=' * len(TITLE), URL)}
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
    'umbrella': None,
    'overlap': None,
    'maxdx': None,
    'mincycle': None
}

SECTIONS['system'] = {
    'dimensions': 3,
    'input_file': None,
    'temperature': 1.0,
    'units': 'lj',
}

SECTIONS['unit-system'] = {
    'charge': None,
    'energy': None,
    'length': None,
    'mass': None,
    'name': None,
}

SECTIONS['engine'] = {
    'class': None,
    'exe_path': HERE,
    'module': None,
    'rgen': 'rgen',
}

SECTIONS['box'] = {
    'cell': None,
    'high': None,
    'low': None,
    'periodic': None,
}

SECTIONS['particles'] = {
    'mass': None,
    'name': None,
    'npart': None,
    'position': None,
    'ptype': None,
    'type': 'internal',
    'velocity': None,
}

SECTIONS['forcefield'] = {
    'description': None
}

SECTIONS['potential'] = {
    'class': None,
    'parameter': None
}

SECTIONS['orderparameter'] = {
    'class': None,
    'module': None,
    'name': 'Order Parameter'
}

SECTIONS['collective-variable'] = {
    'class': None,
    'module': None,
    'name': None
}

SECTIONS['output'] = {
    'backup': 'append',
    'cross-file': 1,
    'energy-file': 1,
    'pathensemble-file': 1,
    'prefix': None,
    'order-file': 1,
    'restart-file': 1,
    'screen': 10,
    'trajectory-file': 100,
}

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
    'zero_momentum': False,
    'mirror_freq': 0,
    'target_freq': 0,
    'target_indices': [],
}

SECTIONS['initial-path'] = {
    'method': None
}

SECTIONS['retis'] = {
    'nullmoves': None,
    'relative_shoots': None,
    'rgen': None,
    'seed': None,
    'swapfreq': None,
    'swapsimul': None,
}

SECTIONS['ensemble'] = {
    'interface': None
}


SPECIAL_KEY = {'parameter'}

# This dictionary contains sections where the keywords
# can not be defined before we parse the input. The reason
# for this is that we support user-defined external modules
# and that the user should have the freedom to define keywords
# for these modules:
ALLOW_MULTIPLE = {
    'collective-variable',
    'engine',
    'ensemble',
    'initial-path',
    'orderparameter',
    'potential',
}

# This dictionary contains sections that can be defined
# multiple times. When parsing, these sections will be
# prefixed with a number to distinguish them.
SPECIAL_MULTIPLE = {
    'collective-variable',
    'ensemble',
    'potential',
}

OUTPUT_TASKS = {}
OUTPUT_TASKS['energy'] = {
    'target': 'file',
    'filename': 'energy.txt',
    'result': ('thermo',),
    'when': 'energy-file',
    'formatter': EnergyFormatter,
}
OUTPUT_TASKS['order'] = {
    'target': 'file',
    'filename': 'order.txt',
    'result': ('order',),
    'when': 'order-file',
    'formatter': OrderFormatter,
}
OUTPUT_TASKS['cross'] = {
    'target': 'file',
    'filename': 'cross.txt',
    'result': ('cross',),
    'when': 'cross-file',
    'formatter': CrossFormatter,
}
OUTPUT_TASKS['traj-txt'] = {
    'target': 'file',
    'filename': 'traj.txt',
    'result': ('system',),
    'when': 'trajectory-file',
    'formatter': SnapshotFormatter,
}
OUTPUT_TASKS['traj-xyz'] = {
    'target': 'file',
    'filename': 'traj.xyz',
    'result': ('system',),
    'when': 'trajectory-file',
    'formatter': SnapshotFormatter,
}
OUTPUT_TASKS['thermo-screen'] = {
    'target': 'screen',
    'result': ('thermo',),
    'when': 'screen',
    'formatter': ThermoTableFormatter,
}
OUTPUT_TASKS['thermo-file'] = {
    'target': 'file',
    'filename': 'thermo.txt',
    'result': ('thermo',),
    'when': 'energy-file',
    'formatter': ThermoTableFormatter,
}
OUTPUT_TASKS['pathensemble'] = {
    'target': 'file',
    'filename': 'pathensemble.txt',
    'result': ('pathensemble',),
    'when': 'pathensemble-file',
    'formatter': PathEnsembleFormatter,
}
OUTPUT_TASKS['pathensemble-screen'] = {
    'target': 'screen',
    'result': ('pathensemble',),
    'when': 'screen',
    'formatter': PathTableFormatter,
}
OUTPUT_TASKS['pathensemble-retis-screen'] = {
    'target': 'screen',
    'result': ('pathensemble',),
    'when': 'screen',
    'formatter': RETISResultFormatter,
}
OUTPUT_TASKS['path-order'] = {
    'target': 'file',
    'filename': 'order.txt',
    'result': ('path', 'status'),
    'when': 'order-file',
    'formatter': OrderPathFormatter,
}
OUTPUT_TASKS['path-energy'] = {
    'target': 'file',
    'filename': 'energy.txt',
    'result': ('path', 'status'),
    'when': 'energy-file',
    'formatter': EnergyPathFormatter,
}
OUTPUT_TASKS['path-traj-ext'] = {
    'target': 'file-archive',
    'filename': 'traj.txt',
    'result': ('path', 'status', 'pathensemble'),
    'when': 'trajectory-file',
    'writer': PathStorage,
}
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

def units_from_settings(settings):
    """Set up units from given input settings.

    Parameters
    ----------
    settings : dict
        A dict defining the units.

    Returns
    -------
    msg : string
        Just a string with some information about the units
        created. This can be used for printing out some info to
        the user.

    """
    unit = settings['system']['units'].lower().strip()
    if 'unit-system' in settings:
        try:
            unit2 = settings['unit-system']['name'].lower()
        except KeyError as err:
            msg = 'Could not find "name" setting for section "unit-system"!'
            logger.error(msg)
            raise ValueError(msg) from err
        if not unit2 == unit:
            msg = f'Inconsistent unit settings "{unit}" != "{unit2}".'
            logger.error(msg)
            raise ValueError(msg)
        setts = {}
        for key in ('length', 'energy', 'mass', 'charge'):
            try:
                setts[key] = settings['unit-system'][key]
            except KeyError as err:
                msg = f'Could not find "{key}" for section "unit-system"!'
                logger.error(msg)
                raise ValueError(msg) from err
        logger.debug('Creating unit system: "%s".', unit)
        create_conversion_factors(unit, **setts)
        msg = f'Created unit system: "{unit}".'
    else:
        logger.debug('Creating units: "%s".', unit)
        create_conversion_factors(unit)
        msg = f'Created units: "{unit}".'
    return msg

def create_conversion_factors(unit, length=None, energy=None, mass=None,
                              charge=None):
    """Set up conversion factors for a system of units.

    Parameters
    ----------
    unit : string
        A name for the system of units
    length : tuple, optional
        This is the length unit given as (float, string) where the
        float is the numerical value and the string the unit,
        e.g. `(1.0, nm)`.
    energy : tuple, optional
        This is the energy unit given as (float, string) where the
        float is the numerical value and the string the unit,
        e.g. `(1.0, eV)`.
    mass : tuple, optional
        This is the mass unit given as (float, string) where the
        float is the numerical value and the string the unit,
        e.g. `(1.0, g/mol)`.
    charge : string, optional
        This is the unit of charge given as a string, e.g. 'e' or 'C'.

    Returns
    -------
    None but will update `CONVERT` so that the conversion factors are
    available.

    """
    # First just set up conversions for the base units:
    for key in DIMENSIONS:
        convert_bases(key)
    # Check inputs and generate factors:
    length = _check_input_unit(unit, 'length', length)
    energy = _check_input_unit(unit, 'energy', energy)
    mass = _check_input_unit(unit, 'mass', mass)
    if charge is None:
        try:
            charge = UNIT_SYSTEMS[unit]['charge']
        except KeyError as err:
            msg = f'Undefined charge unit for {unit}'
            raise ValueError(msg) from err
    else:
        if charge not in UNITS['charge']:
            msg = f'Unknown charge unit "{charge}" requested.'
            raise ValueError(msg)
    generate_conversion_factors(unit, length, energy, mass, charge=charge)

def generate_conversion_factors(unit, distance, energy, mass, charge='e'):
    u"""Create conversions for a system of units from fundamental units.

    This will create a system of units from the three fundamental units
    distance, energy and mass.

    Parameters
    ----------
    unit : string
        This is a label for the unit
    distance : tuple
        This is the distance unit. The form is assumed to be
        `(value, unit)` where the unit is one of the known distance
        units, 'nm', 'A', 'm'.
    energy : tuple
        This is the energy unit. The form is assumed to be
        `(value, unit)` where unit is one of the known energy
        units, 'J', 'kcal', 'kcal/mol', 'kb'.
    mass : tuple
        This is the mass unit. The form is assumed to be `(value, unit)`
        where unit is one of the known mass units, 'g/mol', 'kg', 'g'.
    charge : string, optional
        This selects the base charge. It can be 'C' or 'e' for Coulomb
        or the electron charge. This will determine how we treat
        Coulomb's constant.

    """
    CONVERT['length'][unit, distance[1]] = distance[0]
    CONVERT['mass'][unit, mass[1]] = mass[0]
    if energy[1] == 'kB':  # in case the energy is given in units of kB.
        CONVERT['energy'][unit, 'J'] = energy[0] * CONSTANTS['kB']['J/K']
    else:
        CONVERT['energy'][unit, energy[1]] = energy[0]
        # let us also check if we can define kB now:
        if unit not in CONSTANTS['kB']:
            try:
                kboltz = CONSTANTS['kB'][f'{energy[1]}/K'] / energy[0]
                CONSTANTS['kB'][unit] = kboltz
            except KeyError as err:
                msg = f'For "{unit}" you need to define kB'
                raise ValueError(msg) from err
    # First, set up simple conversions:
    for dim in ('length', 'energy', 'mass'):
        _generate_conversion_for_dim(CONVERT, dim, unit)
    # We can now set up time conversions (since it's using length, mass and
    # energy:
    try:
        value = (CONVERT['length'][unit, 'm']**2 *
                 CONVERT['mass'][unit, 'kg'] /
                 CONVERT['energy'][unit, 'J'])**0.5
    except KeyError as err:
        keys = [(unit, 'm'), (unit, 'kg'), (unit, 'J')]
        dims = ['length', 'mass', 'energy']
        msg = ""
        for key, dim in zip(keys, dims):
            if key not in CONVERT[dim]:
                msg += 'Missing {key} in CONVERT["{dim}"]. '
        raise ValueError(msg) from err
    _add_conversion_and_inverse(CONVERT['time'], value, unit, 's')
    # And velocity (since it's using time and length):
    value = CONVERT['length'][unit, 'm'] / CONVERT['time'][unit, 's']
    _add_conversion_and_inverse(CONVERT['velocity'], value, unit, 'm/s')
    # And pressure (since it's using energy and length):
    value = CONVERT['energy'][unit, 'J'] / CONVERT['length'][unit, 'm']**3
    _add_conversion_and_inverse(CONVERT['pressure'], value, unit, 'Pa')
    # And force (since it's using energy and length):
    value = CONVERT['energy'][unit, 'J'] / CONVERT['length'][unit, 'm']
    _add_conversion_and_inverse(CONVERT['force'], value, unit, 'N')
    # Generate the rest of the conversions:
    for dim in ('time', 'velocity', 'pressure', 'force'):
        _generate_conversion_for_dim(CONVERT, dim, unit)
    # Now, figure out the Temperature conversion:
    kboltz = CONSTANTS['kB']['J/K'] * CONVERT['energy']['J', unit]
    # kboltz in now in units of 'unit'/K, temperature conversion is:
    value = CONSTANTS['kB'][unit] / kboltz
    _add_conversion_and_inverse(CONVERT['temperature'], value, unit, 'K')
    # convert permittivity:
    if charge == 'C':
        CONSTANTS['e0'][unit] = CONSTANTS['e0']['F/m']
    else:
        CONSTANTS['e0'][unit] = (CONSTANTS['e0']['F/m'] *
                                 CONVERT['charge']['C', 'e']**2 /
                                 (CONVERT['force']['N', unit] *
                                  CONVERT['length']['m', unit]**2))
    value = np.sqrt(4.0 * np.pi * CONSTANTS['e0'][unit])
    _add_conversion_and_inverse(CONVERT['charge'], value, unit, charge)
    # convert [charge] * V/A to force, in case it's needed in the future:
    # qE = CONVERT['energy']['J', unit] / CONVERT['charge']['C', 'e']
    _generate_conversion_for_dim(CONVERT, 'charge', unit)

def convert_bases(dimension):
    """Create all conversions between base units.

    This function will generate all conversions between base units
    defined in a `UNITS[dimension]` dictionary. It assumes that one of
    the bases have been used to defined conversions to all other bases.

    Parameters
    ----------
    dimension : string
        The dimension to convert for.

    """
    convert = CONVERT[dimension]
    # start by generating inverses
    generate_inverse(convert)
    for key1 in UNITS[dimension]:
        for key2 in UNITS[dimension]:
            if key1 == key2:
                continue
            value = bfs_convert(convert, key1, key2)[1]
            if value is not None:
                unit1 = (key1, key2)
                unit2 = (key2, key1)
                convert[unit1] = value
                convert[unit2] = 1.0 / convert[unit1]
            else:
                logger.warning(('Could not convert base %s -> %s for '
                                'dimension %s'), key1, key2, dimension)

def generate_inverse(conversions):
    """Generate all simple inverse conversions.

    A simple inverse conversion is something we can obtain by doing
    a ``1 / unit`` type of conversion.

    Parameters
    ----------
    conversions : dictionary
        The unit conversions as `convert[quantity]`.
        Note that this variable will be updated by this function.

    Returns
    -------
    out : None
        Will not return anything, but will update the given parameter
        `conversions`.

    """
    newconvert = {}
    for unit in conversions:
        unit_from, unit_to = unit
        newunit = (unit_to, unit_from)
        if newunit not in conversions:
            newconvert[newunit] = 1.0 / conversions[unit]
    for newunit, value in newconvert.items():
        conversions[newunit] = value

def bfs_convert(conversions, unit_from, unit_to):
    """Generate unit conversion between the provided units.

    The unit conversion can be obtained given that a "path" between
    these units exist. This path is obtained by a Breadth-first search.

    Parameters
    ----------
    conversions : dictionary
        The unit conversion as `convert[quantity]`.
    unit_from : string
        Starting unit.
    unit_to : string
        Target unit.

    Returns
    -------
    out[0] : tuple
        A tuple containing the two units: `(unit_from, unit_to)`.
    out[1] : float
        The conversion factor.
    out[2] : list of tuples
        The 'path' between the units, i.e. the traversal from
        `unit_from` to `unit_to`. `out[2][i]` gives the
        `(unit_from, unit_to)` tuple for step `i` in the conversion.

    """
    if unit_from == unit_to:
        return (unit_from, unit_to), 1.0, None
    if (unit_from, unit_to) in conversions:
        return (unit_from, unit_to), conversions[unit_from, unit_to], None
    que = deque([unit_from])
    visited = [unit_from]
    parents = {unit_from: None}
    while que:
        node = que.popleft()
        if node == unit_to:
            break
        for unit in conversions:
            unit1, unit2 = unit
            if unit1 != node:
                continue
            if unit2 not in visited:
                visited.append(unit2)
                que.append(unit2)
                parents[unit2] = node
    path = []
    if unit_to not in parents:
        logger.warning('Could not determine conversion %s -> %s', unit_from,
                       unit_to)
        return (unit_from, unit_to), None, None
    node = unit_to
    while parents[node]:
        new = [None, node]
        node = parents[node]
        new[0] = node
        path.append(tuple(new))
    factor = 1
    for unit in path[::-1]:
        factor *= conversions[unit]
    return (unit_from, unit_to), factor, path[::-1]

def _check_input_unit(unit, dim, input_unit):
    """Check input units for :py:func:`.create_conversion_factors`.

    Parameters
    ----------
    unit : string
        Name for the unit system we are dealing with.
    dim : string
        The dimension we are looking at, typically 'length', 'mass' or
        'energy'.
    input_unit : tuple
        This is the input unit on the form (value, string) where the
        value is the numerical value and the string the unit,
        e.g. `(1.0, nm)`.

    Returns
    -------
    out : tuple
        The `input_unit` if it passes the tests, otherwise an exception
        will be raised. If the `input_unit` is `None` the default values
        from `UNIT_SYSTEMS` will be returned if they have been defined.

    Raises
    ------
    ValueError
        If the unit in `input_unit` is unknown or malformed.

    """
    if input_unit is not None:
        try:
            value, unit_dim = input_unit
            if unit_dim not in UNITS[dim] and unit_dim != 'kB':
                msg = f'Unknown {dim} unit: {unit_dim}'
                raise LookupError(msg)
            return value, unit_dim
        except TypeError as err:
            msg = f'Could not understand {dim} unit: {input_unit}'
            raise TypeError(msg) from err
    else:  # Try do get values from default:
        try:
            value, unit_dim = UNIT_SYSTEMS[unit][dim]
            return value, unit_dim
        except KeyError as err:
            msg = f'Could not determine {dim} unit for {unit}'
            raise ValueError(msg) from err

def _generate_conversion_for_dim(conv_dict, dim, unit):
    """Generate conversion factors for the specified dimension.

    It will generate all conversions from a specified unit for a given
    dimension considering all other units defined in `UNITS`.

    Parameters
    ----------
    conv_dict : dict
        A dictionary with conversions which we wish to update.
    dim : string
        The dimension to consider
    unit : string
        The unit we create conversions for.

    Returns
    -------
    None, but updates the given `conv_dict`

    """
    convertdim = conv_dict[dim]
    for unit_to in UNITS[dim]:
        if unit == unit_to:  # just skip
            continue
        value = bfs_convert(convertdim, unit, unit_to)[1]
        if value is not None:
            _add_conversion_and_inverse(convertdim, value, unit, unit_to)
        else:
            logger.warning('Could not convert %s -> %s for dimension %s',
                           unit, unit_to, dim)


def _add_conversion_and_inverse(conv_dict, value, unit1, unit2):
    """Add a specific conversion and it's inverse.

    This function is mainly here to ensure that we don't forget to add
    the inverse conversions.

    Parameters
    ----------
    conv_dict : dict
        This is where we store the conversion.
    value : float
        The conversion factor to add
    unit1 : string
        The unit we are converting from.
    unit2 : string
        The unit we are converting to.

    Returns
    -------
    None, but updates the given `conv_dict`.

    """
    conv_dict[unit1, unit2] = value
    conv_dict[unit2, unit1] = 1.0 / conv_dict[unit1, unit2]


def prepare_system(settings):
    """Create a system from given settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    syst : object like :py:class:`.syst`
        This object will correspond to the selected simulation type.

    """
    if settings.get('system', {}).get('obj', False):
        return settings['system']['obj']

    logtxt = 'Initializing unit system.'
    print_to_screen(logtxt, level='info')
    logger.info(logtxt)
    units_from_settings(settings)

    logtxt = 'Creating system from settings.'
    print_to_screen(logtxt, level='info')
    logger.info(logtxt)
    system = create_system(settings)

    system.extra_setup()
    return system

def create_system(settings):
    """Create a system from input settings.

    Parameters
    ----------
    settings : dict
        The dict with the simulation settings.

    Returns
    -------
    system : object like :py:class:`.System`
        The system object we create here.

    """
    if 'restart' in settings:
        system = System()
        system.load_restart_info(settings['restart']['system'])
        return system

    vel = None
    particles = Particles(dim=3)
    if 'input_path' in settings['engine']:
        required_file = {}
        # Get the engine input files
        if 'cp2k' in settings['engine']['class'].lower():
            required_file = {'conf': 'initial.{}'.format(
                settings['engine'].get('cp2k_format', 'xyz'))}
        elif 'gromacs' in settings['engine']['class'].lower():
            required_file = {
                'conf': 'conf.{}'.format(
                    settings['engine'].get('gmx_format', 'gro'))}

        input_file = look_for_input_files(settings['engine']['input_path'],
                                          required_file)
        particles.set_pos((input_file['conf'], None))
        # gromacs tests fail without this if-statement.
        # _get_snapshot_from_file() has issues with reading .g96 format.
        if settings['engine']['class'].lower() == 'cp2k':
            particles.mass, particles.imass = _assign_mass_from_file(
                input_file['conf'],
                settings['system']['units'])
    else:
        particles.set_pos((None, None))
    particles.set_vel(False)
    box = None

    system = System(
        temperature=settings['system']['temperature'],
        units=settings['system']['units'],
        box=box
        )
    system.particles = particles

    # figure out what to do with velocities:
    if 'position' in settings['particles'] and \
            'input_file' not in settings['particles']['position']:
        vel_gen = create_velocities(system, settings, vel)
        if not (vel_gen or vel):
            logger.warning('Velocities not created/read: Set to zero!')

    return system

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

def create_external(settings, key, factory, required_methods,
                    key_settings=None):
    """Create external objects from settings.

    This method will handle the creation of objects from settings. The
    requested objects can be PyRETIS internals or defined in external
    modules.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.
    key : string
        The setting we are creating for.
    factory : callable
        A method to call that can handle the creation of internal
        objects for us.
    required_methods : list of strings
        The methods we need to have if creating an object from external
        files.
    key_settings : dict, optional
        This dictionary contains the settings for the specific key we
        are processing. If this is not given, we will try to obtain
        these settings by `settings[key]`. The reason why we make it
        possible to pass these as settings is in case we are processing
        a key which does not give a simple setting, but a list of settings.
        It that case `settings[key]` will give a list to process. That list
        is iterated somewhere else and `key_settings` can then be used to
        process these elements.

    Returns
    -------
    out : object
        This object represents the class we are requesting here.

    """
    if key_settings is None:
        print('clown 1')
        try:
            key_settings = settings[key]
        except KeyError:
            logger.debug('No "%s" setting found. Skipping set-up', key)
            return None
    print('clown 2')
    module = key_settings.get('module', None)
    print(key_settings)
    klass = None
    try:
        klass = key_settings['class']
    except KeyError:
        logger.debug('No "class" setting for "%s" specified. Skipping set-up',
                     key)
        return None
    print('clown 3')
    if module is None:
        print('clown 4')
        return factory(key_settings)
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is ok or if we should include
    # the 'exe_path' from settings.
    # 1) Check if we can find the module:
    print('clown 5')
    if os.path.isfile(module):
        print('clown 6')
        obj = import_from(module, klass)
    else:
        print('clown 7')
        if 'exe_path' in settings['simulation']:
            module = os.path.join(settings['simulation']['exe_path'],
                                  module)
            obj = import_from(module, klass)
        else:
            msg = 'Could not find module "{}" for {}!'.format(module, key)
            raise ValueError(msg)
    # run some checks:
    print('clown 8')
    for function in required_methods:
        objfunc = getattr(obj, function, None)
        if not objfunc:
            msg = 'Could not find method {}.{}'.format(klass,
                                                       function)
            logger.critical(msg)
            raise ValueError(msg)
        else:
            if not callable(objfunc):
                msg = 'Method {}.{} is not callable!'.format(klass,
                                                             function)
                logger.critical(msg)
                raise ValueError(msg)
    return initiate_instance(obj, key_settings)

def generic_factory(settings, object_map, name='generic'):
    """Create instances of classes based on settings.

    This method is intended as a semi-generic factory for creating
    instances of different objects based on simulation input settings.
    The input settings define what classes should be created and
    the object_map defines a mapping between settings and the
    class.

    Parameters
    ----------
    settings : dict
        This defines how we set up and select the order parameter.
    object_map : dict
        Definitions on how to initiate the different classes.
    name : string, optional
        Short name for the object type. Only used for error messages.

    Returns
    -------
    out : instance of a class
        The created object, in case we were successful. Otherwise we
        return none.

    """
    try:
        klass = settings['class'].lower()
    except KeyError:
        msg = 'No class given for %s -- could not create object!'
        logger.critical(msg, name)
        return None
    if klass not in object_map:
        logger.critical('Could not create unknown class "%s" for %s',
                        settings['class'], name)
        return None
    cls = object_map[klass]['cls']
    return initiate_instance(cls, settings)

def initiate_instance(klass, settings):
    """Initialise a class with optional arguments.

    Parameters
    ----------
    klass : class
        The class to initiate.
    settings : dict
        Positional and keyword arguments to pass to `klass.__init__()`.

    Returns
    -------
    out : instance of `klass`
        Here, we just return the initiated instance of the given class.

    """
    args, kwargs = _pick_out_arg_kwargs(klass, settings)
    # Ready to initiate:
    msg = 'Initiated "%s" from "%s" %s'
    name = klass.__name__
    mod = klass.__module__
    if not args:
        if not kwargs:
            logger.debug(msg, name, mod, 'without arguments.')
            return klass()
        logger.debug(msg, name, mod, 'with keyword arguments.')
        return klass(**kwargs)
    if not kwargs:
        logger.debug(msg, name, mod, 'with positional arguments.')
        return klass(*args)
    logger.debug(msg, name, mod,
                 'with positional and keyword arguments.')
    return klass(*args, **kwargs)

def _pick_out_arg_kwargs(klass, settings):
    """Pick out arguments for a class from settings.

    Parameters
    ----------
    klass : class
        The class to initiate.
    settings : dict
        Positional and keyword arguments to pass to `klass.__init__()`.

    Returns
    -------
    out[0] : list
        A list of the positional arguments.
    out[1] : dict
        The keyword arguments.

    """
    info = inspect_function(klass.__init__)
    used, args, kwargs = set(), [], {}
    for arg in info['args']:
        if arg == 'self':
            continue
        try:
            args.append(settings[arg])
            used.add(arg)
        except KeyError:
            msg = f'Required argument "{arg}" for "{klass}" not found!'
            raise ValueError(msg)
    for arg in info['kwargs']:
        if arg == 'self':
            continue
        if arg in settings:
            kwargs[arg] = settings[arg]
    return args, kwargs

def inspect_function(function):
    """Return arguments/kwargs of a given function.

    This method is intended for use where we are checking that we can
    call a certain function. This method will return arguments and
    keyword arguments a function expects. This method may be fragile -
    we assume here that we are not really interested in args and
    kwargs and we do not look for more information about these here.

    Parameters
    ----------
    function : callable
        The function to inspect.

    Returns
    -------
    out : dict
        A dict with the arguments, the following keys are defined:

        * `args` : list of the positional arguments
        * `kwargs` : list of keyword arguments
        * `varargs` : list of arguments
        * `keywords` : list of keyword arguments

    """
    out = {'args': [], 'kwargs': [],
           'varargs': [], 'keywords': []}
    arguments = inspect.signature(function)  # pylint: disable=no-member
    for arg in arguments.parameters.values():
        kind = _arg_kind(arg)
        if kind is not None:
            out[kind].append(arg.name)
        else:  # pragma: no cover
            logger.critical('Unknown variable kind "%s" for "%s"',
                            arg.kind, arg.name)
    return out

def _arg_kind(arg):
    """Determine kind for a given argument.

    This method will help :py:func:`.inspect_function` to determine
    the correct kind for arguments.

    Parameters
    ----------
    arg : object like :py:class:`inspect.Parameter`
        The argument we will determine the type of.

    Returns
    -------
    out : string
        A string we use for determine the kind.

    """
    kind = None
    if arg.kind == arg.POSITIONAL_OR_KEYWORD:
        if arg.default is arg.empty:
            kind = 'args'
        else:
            kind = 'kwargs'
    elif arg.kind == arg.POSITIONAL_ONLY:
        kind = 'args'
    elif arg.kind == arg.VAR_POSITIONAL:
        kind = 'varargs'
    elif arg.kind == arg.VAR_KEYWORD:
        kind = 'keywords'
    elif arg.kind == arg.KEYWORD_ONLY:
        # We treat these as keyword arguments:
        kind = 'kwargs'
    return kind


def generate_file_name(basename, directory, settings):
    """Generate file name for an output task, from settings.

    Parameters
    ----------
    basename : string
        The base file name to use.
    directory : string
        A directory to output to. Can be None to output to the
        current working directory.
    settings : dict
        The input settings

    Returns
    -------
    filename : string
        The file name to use.

    """
    prefix = settings['output'].get('prefix', None)
    if prefix is not None:
        filename = f'{prefix}{basename}'
    else:
        filename = basename
    filename = add_dirname(filename, directory)
    return filename

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

def import_from(module_path, function_name):
    """Import a method/class from a module.

    This method will dynamically import a specified method/object
    from a module and return it. If the module can not be imported or
    if we can't find the method/class in the module we will raise
    exceptions.

    Parameters
    ----------
    module_path : string
        The path/filename to load from.
    function_name : string
        The name of the method/class to load.

    Returns
    -------
    out : object
        The thing we managed to import.

    """
    try:
        module_name = os.path.basename(module_path)
        module_name = os.path.splitext(module_name)[0]
        spec = importlib.util.spec_from_file_location(module_name,
                                                      module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        logger.debug('Imported module: %s', module)
        return getattr(module, function_name)
    except (ImportError, IOError):
        msg = f'Could not import module: {module_path}'
        logger.critical(msg)
    except AttributeError:
        msg = f'Could not import "{function_name}" from "{module_path}"'
        logger.critical(msg)
    raise ValueError(msg)


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

def soft_partial_exit(exe_path=''):
    """Check the presence of the EXIT file.

    Parameters
    ----------
    exe_path: string, optional
        Path for the EXIT file.

    Returns
    -------
    out : boolean
        True if EXIT is present.
        False if EXIT in not present.

    """
    exit_file = 'EXIT'
    if exe_path:
        exit_file = os.path.join(exe_path, exit_file)
    if os.path.isfile(exit_file):
        logger.info('Exit file found - will exit between steps.')
        print_to_screen('Exit file found - will exit between steps.',
                        level='warning')

        return True
    return False

def task_from_settings(task, settings, directory, engine, progress=False):
    """Create output task from simulation settings.

    Parameters
    ----------
    task : dict
        Settings for creating a task. This dict contains the type and
        name of the task to create. It can also contain overrides to
        the default settings in :py:data:`OUTPUT_TASKS`.
    settings : dict
        Settings for the simulation.
    directory : string
        The directory to write output files to.
    engine : object like :py:class:`.EngineBase`
        This object is used to determine if we need to do something
        special for external engines. If no engine is given, we do
        not do anything special.
    progress : boolean, optional
        For some simulations, the user may select to display a
        progress bar. We will then just disable the other screen
        output.

    Returns
    -------
    out : object like :py:class:`.OutputTask`
        An output task we can use in the simulation.

    """
    task_type = get_task_type(task, engine)
    task_settings = OUTPUT_TASKS[task_type].copy()
    # Override defaults if any:
    for key in task_settings:
        if key in task:
            task_settings[key] = task[key]

    when = {'every': settings['output'][task_settings['when']]}
    if when['every'] < 1:
        logger.info('Skipping output task %s (freq < 1)', task_type)
        return None

    target = task_settings['target']
    if target == 'screen' and progress:
        logger.info(
            'Disabling output to screen %s since progress bar is ON',
            task['name'],
        )
        return None
    formatter = None
    # Initiate the formatter, note that here we can customize the formatter
    # by supplying arguments to it. This was supported in a previous version
    # of PyRETIS, but for now, none of the formatters needs settings to be
    # created.
    if task_settings.get('formatter', None) is not None:
        formatter = initiate_instance(
            task_settings['formatter'],
            task_settings.get('formatter-settings', {}),
        )
    # Create writer:
    writer = None
    if target == 'screen':
        klass = task_settings.get('writer', ScreenOutput)
        writer = klass(formatter)
    if target in ('file', 'file-archive'):
        filename = generate_file_name(task_settings['filename'], directory,
                                      settings)
        file_mode = get_file_mode(settings)
        if target == 'file':
            klass = task_settings.get('writer', FileIO)
            writer = klass(filename, file_mode, formatter, backup=True)
        if target == 'file-archive':
            klass = task_settings.get('writer', PathStorage)
            writer = klass()
    # Finally make the output task:
    if writer is not None:
        return OutputTask(task['name'], task_settings['result'], writer, when)
    logger.warning('Unknown target "%s". Ignoring task: %s',
                   target, task_type)
    return None

def write_restart_file(filename, simulation):
    """Write restart info for a simulation.

    Parameters
    ----------
    filename : string
        The file we are going to write to.
    simulation : object like :py:class:`.Simulation`
        A simulation object we will get information from.

    """
    info = simulation.restart_info()

    with open(filename, 'wb') as outfile:
        pickle.dump(info, outfile)

def write_ensemble_restart(ensemble, config, save):
    """Write a restart file for a path ensemble.

    Parameters
    ----------
    ensemble : dict
        it contains:

        * `path_ensemble` : object like :py:class:`.PathEnsemble`
          The path ensemble we are writing restart info for.
        * ` system` : object like :py:class:`.System`
          System is used here since we need access to the temperature
          and to the particle list.
        * `order_function` : object like :py:class:`.OrderParameter`
          The class used for calculating the order parameter(s).
        * `engine` : object like :py:class:`.EngineBase`
          The engine to use for propagating a path.

    settings_ens : dict
        A dictionary with the ensemble settings.

    """
    info = {}
    info['rgen'] = ensemble.rgen.get_state()

    filename = os.path.join(
        os.getcwd(),
        config['simulation']['load_dir'],
        save,
        'ensemble.restart')

    with open(filename, 'wb') as outfile:
        toprint = os.path.join(save, 'ensemble.restart')
        pickle.dump(info, outfile)

def get_initiation_method(settings):
    """Return the initiation method from given settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the initiation.

    Returns
    -------
    out : callable
        The method to be used for the initiation.

    """
    _methods = {
        'restart': initiate_restart,
    }
    method = settings['initial-path']['method'].lower()
    if method not in _methods:
        logger.error('Unknown initiation method "%s" requested', method)
        logger.error('Known methods: %s', _methods.keys())
        raise ValueError('Unknown initiation method requested!')
    print_to_screen('Will initiate paths using method "{}".'.format(method))
    logger.info('Initiation method "%s" selected', method)
    return _methods[method]


def initiate_path_simulation(simulation, settings):
    """Initialise a path simulation.

    Parameters
    ----------
    simulation : object like :py:class:`.PathSimulation`
        The simulation we are doing the initiation for.
    settings : dict
        A dictionary with settings for the initiation.

    Returns
    -------
    out : callable
        The method to be used for the initiation.

    """
    cycle = simulation.cycle['step']
    method = get_initiation_method(settings)
    return method(simulation, settings, cycle)

def initiate_restart(simulation, settings, cycle):
    """Initialise paths by loading restart data.

    Parameters
    ----------
    simulation : object like :py:class:`.Simulation`
        The simulation we are setting up.
    settings : dictionary
        A dictionary with settings for the initiation.
    cycle : integer
        The simulation cycles we are starting at.

    """
    for idx, ensemble in enumerate(simulation.ensembles):
        path_ensemble = ensemble['path_ensemble']
        name = path_ensemble.ensemble_name
        logger.info('Loading restart data for path ensemble %s:', name)
        print_to_screen(
            'Loading restart data for path ensemble {}:'.format(name),
            level='warning'
        )
        restart_file = os.path.join('trajs',
            f'e{idx}',
            'ensemble.restart')
        restart_file2 = os.path.join('trajs',
            str(path_ensemble.path_number),
            'ensemble.restart')

        restart_info = read_restart_file(restart_file)
        restart_info2 = read_restart_file(restart_file2)
        logger.info('go restart ensemble' + restart_file)
        logger.info('go restart path' + restart_file2)
        ensemble['engine'].load_restart_info(restart_info['engine'])
        ensemble['engine'].exe_dir = path_ensemble.directory['generate']
        ensemble['system'].load_restart_info(restart_info['system'])
        ensemble['rgen'] = create_random_generator(restart_info['rgen'])
        ensemble['order_function'].load_restart_info(
            restart_info.get('order_function', []))

        if settings['engine']['type'] != 'internal':
            for pp in restart_info2['path_ensemble']['last_path']['phasepoints']:
                bname = os.path.basename(pp['particles']['config'][0])
                pname = os.path.join('trajs', str(path_ensemble.path_number), 'accepted', bname)
                idx = pp['particles']['config'][1]
                pp['particles']['config'] = (pname, idx)

        restart_info2['path_ensemble']
        path_ensemble.load_restart_info(restart_info2['path_ensemble'],
                                        cycle)
        # This allows ensemble renumbering
        if settings['simulation']['task'] == 'retis':
            path_ensemble.ensemble_number = idx

        # The Force field is not part of the restart, just explicitly
        # set it:
        path = path_ensemble.last_path

        yield True, path, path.status, path_ensemble

def read_restart_file(filename):
    """Read restart info for a simulation.

    Parameters
    ----------
    filename : string
        The file we are going to read from.

    """
    with open(filename, 'rb') as infile:
        info = pickle.load(infile)
    return info

def task_from_settings(task, settings, directory, engine, progress=False):
    """Create output task from simulation settings.

    Parameters
    ----------
    task : dict
        Settings for creating a task. This dict contains the type and
        name of the task to create. It can also contain overrides to
        the default settings in :py:data:`OUTPUT_TASKS`.
    settings : dict
        Settings for the simulation.
    directory : string
        The directory to write output files to.
    engine : object like :py:class:`.EngineBase`
        This object is used to determine if we need to do something
        special for external engines. If no engine is given, we do
        not do anything special.
    progress : boolean, optional
        For some simulations, the user may select to display a
        progress bar. We will then just disable the other screen
        output.

    Returns
    -------
    out : object like :py:class:`.OutputTask`
        An output task we can use in the simulation.

    """
    task_type = get_task_type(task, engine)
    task_settings = OUTPUT_TASKS[task_type].copy()
    # Override defaults if any:
    for key in task_settings:
        if key in task:
            task_settings[key] = task[key]

    when = {'every': settings['output'][task_settings['when']]}
    if when['every'] < 1:
        logger.info('Skipping output task %s (freq < 1)', task_type)
        return None

    target = task_settings['target']
    if target == 'screen' and progress:
        logger.info(
            'Disabling output to screen %s since progress bar is ON',
            task['name'],
        )
        return None
    formatter = None
    # Initiate the formatter, note that here we can customize the formatter
    # by supplying arguments to it. This was supported in a previous version
    # of PyRETIS, but for now, none of the formatters needs settings to be
    # created.
    if task_settings.get('formatter', None) is not None:
        formatter = initiate_instance(
            task_settings['formatter'],
            task_settings.get('formatter-settings', {}),
        )
    # Create writer:
    writer = None
    if target == 'screen':
        klass = task_settings.get('writer', ScreenOutput)
        writer = klass(formatter)
    if target in ('file', 'file-archive'):
        filename = generate_file_name(task_settings['filename'], directory,
                                      settings)
        file_mode = get_file_mode(settings)
        if target == 'file':
            klass = task_settings.get('writer', FileIO)
            writer = klass(filename, file_mode, formatter, backup=True)
        if target == 'file-archive':
            klass = task_settings.get('writer', PathStorage)
            writer = klass()
    # Finally make the output task:
    if writer is not None:
        return OutputTask(task['name'], task_settings['result'], writer, when)
    logger.warning('Unknown target "%s". Ignoring task: %s',
                   target, task_type)
    return None

def get_task_type(task, engine):
    """Do additional handling for a path task.

    The path task is special since we do very different things for
    external paths. The set-up required to do this is handled here.

    Parameters
    ----------
    task : dict
        Settings related to the specific task.
    engine : object like :py:class:`.EngineBase`
        This object is used to determine if we need to do something
        special for external engines. If no engine is given, we do
        not do anything special.

    Returns
    -------
    out : string
        The task type we are going to be creating for.

    """
    if task['type'] == 'path-traj-{}':
        if engine is None or engine.engine_type in ('internal', 'openmm'):
            fmt = 'int'
        else:
            fmt = 'ext'
        return task['type'].format(fmt)
    return task['type']

def add_dirname(filename, dirname):
    """Add a directory as a prefix to a filename, i.e. `dirname/filename`.

    Parameters
    ----------
    filename : string
        The filename.
    dirname : string
        The directory we want to prefix. It can be None, in which
        case we ignore it.

    Returns
    -------
    out : string
        The path to the resulting file.

    """
    if dirname is not None:
        return os.path.join(dirname, filename)
    return filename

def get_file_mode(settings):
    """Determine if we should append or backup existing files.

    This method translates the backup settings into a file mode string.
    We assume here that the file is opened for writing.

    Parameters
    ----------
    settings : dict
        The simulation settings.

    Returns
    -------
    file_mode : string
        A string representing the file mode to use.

    """
    file_mode = 'w'
    try:
        old = settings['output']['backup'].lower()
        if old == 'append':
            logger.debug('Will append to existing files.')
            file_mode = 'a'
    except AttributeError:
        logger.warning('Could not understand setting for "backup"'
                       ' in "output" section.')
        old = 'backup'
        logger.warning('Handling of existing files is set to: "%s"', old)
        settings['output']['backup'] = old
    return file_mode


def make_dirs(dirname):
    """Create directories for path simulations.

    This function will create a folder using a specified path.
    If the path already exists and if it's a directory, we will do
    nothing. If the path exists and is a file we will raise an
    `OSError` exception here.

    Parameters
    ----------
    dirname : string
        This is the directory to create.

    Returns
    -------
    out : string
        A string with some info on what this function did. Intended for
        output.

    """
    try:
        os.makedirs(dirname)
        msg = f'Created directory: "{dirname}"'
    except OSError as err:
        if err.errno != errno.EEXIST:  # pragma: no cover
            raise err
        if os.path.isfile(dirname):
            msg = f'"{dirname}" is a file. Will abort!'
            raise OSError(errno.EEXIST, msg) from err
        if os.path.isdir(dirname):
            msg = f'Directory "{dirname}" already exist.'
    return msg

def parse_settings_file(filename, add_default=True):
    """Parse settings from a file name.

    Here, we read the file line-by-line and check if the current line
    contains a keyword, if so, we parse that keyword.

    Parameters
    ----------
    filename : string
        The file to parse.
    add_default : boolean
        If True, we will add default settings as well for keywords
        not found in the input.

    Returns
    -------
    settings : dict
        A dictionary with settings for PyRETIS.

    """
    with open(filename, 'r', encoding='utf-8') as fileh:
        raw_sections = _parse_sections(fileh)
    settings = _parse_all_raw_sections(raw_sections)
    if add_default:
        logger.debug('Adding default settings')
        add_default_settings(settings)
        add_specific_default_settings(settings)
    if settings['simulation']['task'] in {'retis', 'tis',
                                          'explore', 'make-tis-files'}:
        fill_up_tis_and_retis_settings(settings)
        # Set up checks before to continue. This section shall GROW.
        check_interfaces(settings)
        check_for_bullshitt(settings)
    return _clean_settings(settings)

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
    if settings.get('initial-path', {}).get('method') == 'restart':
        if settings['simulation'].get('restart') is None:
            settings['simulation']['restart'] = 'pyretis.restart'

    for sec, sec_val in SECTIONS.items():
        if sec not in settings:
            settings[sec] = {}
        for key, val in sec_val.items():
            if val is not None and key not in settings[sec]:
                settings[sec][key] = val
    to_remove = [key for key in settings if len(settings[key]) == 0]
    for key in to_remove:
        settings.pop(key, None)


def add_specific_default_settings(settings):
    """Add specific default settings for each simulation task.

    Parameters
    ----------
    settings : dict
        The current input settings.

    Returns
    -------
    None, but this method might add data to the input settings.

    """
    task = settings['simulation'].get('task')
    if task not in settings:
        settings[task] = {}

    if 'exp' in task:
        settings['tis']['shooting_move'] = 'exp'

    if task in {'tis', 'make-tis-files'}:
        if 'flux' not in settings['simulation']:
            settings['simulation']['flux'] = False
        if 'zero_ensemble' not in settings['simulation']:
            settings['simulation']['zero_ensemble'] = False

    if task == 'retis':
        if 'flux' not in settings['simulation']:
            settings['simulation']['flux'] = True
        if 'zero_ensemble' not in settings['simulation']:
            settings['simulation']['zero_ensemble'] = True

    eng_name = settings['engine'].get('class', 'NoneEngine')
    if eng_name[:7].lower() in {'gromacs', 'cp2k', 'lammps'}:
        settings['particles']['type'] = 'external'
        settings['engine']['type'] = 'external'
        input_path = os.path.join(settings['engine'].get('exe_path', '.'),
                                  settings['engine'].get('input_path', '.'))
        engine_checker = {'gromacs': gromacs_settings,
                          'cp2k': cp2k_settings}
        # Checks engine specific settings
        if engine_checker.get(eng_name[:7].lower()):
            engine_checker[eng_name[:7].lower()](settings, input_path)
    else:
        settings['particles']['type'] = 'internal'
        settings['engine']['type'] = settings['engine'].get('type', 'internal')

def check_for_bullshitt(settings):
    """Do what is stated.

    Just for the input settings.

    Parameters
    ----------
    settings : dict
        The current input settings.

    """
    if (settings['simulation']['task'] in {'retis', 'tis'} and
            len(settings['simulation']['interfaces']) < 3):
        msg = "Insufficient number of interfaces for "\
            f"{settings['simulation']['task']}"

    elif settings['simulation']['task'] in {'tis', 'retis'}:
        if not is_sorted(settings['simulation']['interfaces']):
            msg = "Interface lambda positions in the simulation "\
                "entry are NOT sorted (small to large)"

        if 'ensemble' in settings:
            savelambda = []
            for i_ens, ens in enumerate(settings['ensemble']):
                if 'ensemble_number' not in ens and \
                        'interface' not in ens:
                    msg = "An ensemble has been introduced without "\
                        "references (interface in ensemble settings)"
                else:
                    savelambda.append(settings['simulation']['interfaces']
                                      [i_ens])
                    if 'interface' in ens and ens['interface'] \
                            not in settings['simulation']['interfaces']:
                        msg = "An ensemble has been introduced with an "\
                            "interface not listed in the simulation interfaces"

    if 'msg' in locals():
        raise ValueError(msg)

def check_interfaces(settings):
    """Check that the interfaces are properly defined.

    Parameters
    ----------
    settings : dict
        The current input settings.

    """
    msg = []
    if settings['simulation'].get('flux', False) and \
            not settings['simulation']['zero_ensemble']:
        msg += ['Settings for flux and zero_ensemble are inconsistent.']

    if settings['simulation']['task'] in ['retis', 'tis']:
        if len(settings['simulation']['interfaces']) < 3:
            msg += ['Insufficient number of interfaces for {}'
                    .format(settings['simulation']['task'])]

        if not is_sorted(settings['simulation']['interfaces']):
            msg += ['Interface positions in the simulation interfaces ']
            msg += ['input are NOT properly sorted (ascending order)']

    if msg:
        msgtxt = '\n'.join(msg)
        logger.critical(msgtxt)
        return False

    return True

def _parse_sections(inputtxt):
    """Find sections in the input file with raw data.

    This method will find sections in the input file and
    collect the corresponding raw data.

    Parameters
    ----------
    inputtxt : list of strings or iterable file object
        The raw data to parse.

    Returns
    -------
    raw_data : dict
        A dictionary with keys corresponding to the sections found
        in the input file. `raw_data[key]` contains the raw data
        for the section corresponding to `key`.

    """
    multiple = {key: 0 for key in SPECIAL_MULTIPLE}
    raw_data = {'heading': []}
    previous_line = None
    add_section = 'heading'
    data = []
    for lines in inputtxt:
        current_line, _, _ = lines.strip().partition('#')
        if not current_line:
            continue
        if current_line.startswith('---'):
            if previous_line is None:
                continue
            section_title = previous_line.split()[0].lower()
            if section_title in SPECIAL_MULTIPLE:
                new_section_title = f'{section_title}{multiple[section_title]}'
                multiple[section_title] += 1
                section_title = new_section_title
            if section_title not in raw_data:
                raw_data[section_title] = []
            raw_data[add_section].extend(data[:-1])
            data = []
            add_section = section_title
        else:
            data += [current_line]
        previous_line = current_line
    if add_section is not None:
        raw_data[add_section].extend(data)
    return raw_data


def _parse_section_heading(raw_section):
    """Parse the heading section.

    Parameters
    ----------
    raw_section : list of strings
        The text data for a given section which will be parsed.

    Returns
    -------
    setting : dict
        A dict with keys corresponding to the settings.

    """
    if not raw_section:
        return None
    return {'text': '\n'.join(raw_section)}


def _merge_section_text(raw_section):
    """Merge text for settings that are split across lines.

    This method supports keyword settings that are split across several
    lines. Here we merge these lines by assuming that keywords separate
    different settings.

    Parameters
    ----------
    raw_section : string
        The text we will merge.

    """
    merged = []
    for line in raw_section:
        _, _, found_keyword = look_for_keyword(line)
        if found_keyword or not merged:
            merged.append(line)
        else:
            merged[-1] = ''.join((merged[-1], line))
    return merged


def _parse_section_default(raw_section):
    """Parse a raw section.

    This is the default parser for sections.

    Parameters
    ----------
    raw_section : list of strings
        The text data for a given section which will be parsed.

    Returns
    -------
    setting : dict
        A dict with keys corresponding to the settings.

    """
    merged = _merge_section_text(raw_section)
    setting = {}
    for line in merged:
        match, keyword, found_keyword = look_for_keyword(line)
        if found_keyword:
            raw = line[len(match):].strip()
            parsed, success = parse_primitive(raw)
            if success:
                special = None
                for skey in SPECIAL_MULTIPLE:
                    # To avoid a false True for ensemble_number
                    if keyword.startswith(skey) and keyword[len(skey)] != '_':
                        special = skey

                if special is not None:
                    var = [''.join(line.split(keyword.split()[0])[1])]
                    new_setting = _parse_section_default(var)
                    var = line.split(special)[1].split()[0]
                    num = 0 if not var.isdigit() else int(var)

                    if special not in setting:
                        setting[special] = [{}]
                    while num >= len(setting[special]):
                        setting[special].append({})
                    setting[special][num].update(new_setting)

                elif keyword in SPECIAL_KEY:
                    if keyword not in setting:
                        setting[keyword] = {}
                    var = line.split(keyword)[1].split()[0]
                    # Yes, in some cases we really want an integer.
                    # Note: This will only work for positive numbers
                    # (which we are assuming here).
                    if var.isdigit():
                        setting[keyword][int(var)] = parsed
                    else:
                        setting[keyword][var] = parsed

                elif len(keyword.split()) > 1:
                    key_0 = match.split()[0]
                    var = [' '.join(line.split()[1:])]
                    new_setting = _parse_section_default(var)
                    if key_0 not in setting:
                        setting[key_0] = {}
                    for key, val in new_setting.items():
                        if key in setting[key_0]:
                            setting[key_0][key].update(val)
                        else:
                            setting[key_0][key] = val
                else:
                    setting[keyword] = parsed

            else:  # pragma: no cover
                msg = [f'Could read keyword {keyword}']
                msg += ['Keyword was skipped, please check your input!']
                msg += [f'Input setting: {raw}']
                msgtxt = '\n'.join(msg)
                logger.critical(msgtxt)
    return setting


def _parse_raw_section(raw_section, section):
    """Parse the raw data from a section.

    Parameters
    ----------
    raw_section : list of strings
        The text data for a given section which will be parsed.
    section : string
        A text identifying the section we are parsing for. This is
        used to get a list over valid keywords for the section.

    Returns
    -------
    out : dict
        A dict with keys corresponding to the settings.

    """
    if section not in SECTIONS:
        # Unknown section, just ignore it and give a warning.
        msgtxt = f'Ignoring unknown input section "{section}"'
        logger.warning(msgtxt)
        return None
    if section == 'heading':
        return _parse_section_heading(raw_section)
    return _parse_section_default(raw_section)


def _parse_all_raw_sections(raw_sections):
    """Parse all raw sections.

    This method is helpful for running tests etc.

    Parameters
    ----------
    raw_sections : dict
        The dictionary with the raw data in sections.

    Returns
    -------
    settings : dict
        The parsed settings, with one key for each section parsed.

    """
    settings = {}
    for key, val in raw_sections.items():
        special = None
        for i in SPECIAL_MULTIPLE:
            if key.startswith(i):
                special = i
        if special is not None:
            new_setting = _parse_raw_section(val, special)
            if special not in settings:
                settings[special] = []
            settings[special].append(new_setting)
        else:
            new_setting = _parse_raw_section(val, key)
            if new_setting is None:
                continue
            if key not in settings:
                settings[key] = {}
            for sub_key in new_setting:
                settings[key][sub_key] = new_setting[sub_key]
    return settings

def look_for_keyword(line):
    """Search for a keyword in the given string.

    A string is assumed to define a keyword if the keyword appears as
    the first word in the string, ending with a `=`.

    Parameters
    ----------
    line : string
        A string to check for a keyword.

    Returns
    -------
    out[0] : string
        The matched keyword. It may contain spaces and it will also
        contain the matched `=` separator.
    out[1] : string
        A lower-case, stripped version of `out[0]`.
    out[2] : boolean
        `True` if we found a possible keyword.

    """
    # Match a word followed by a '=':
    key = re.match(r'(.*?)=', line)
    if key:
        keyword = ''.join([key.group(1), '='])
        keyword_low = key.group(1).strip().lower()
        for i in SPECIAL_KEY:
            if keyword_low.startswith(i):
                return keyword, i, True

        # Here we assume that keys with len One or Two are Atoms names
        if len(keyword_low) <= 2:
            keyword_low = key.group(1).strip()

        return keyword, keyword_low, True
    return None, None, False

def parse_primitive(text):
    """Parse text to Python using the ast module.

    Parameters
    ----------
    text : string
        The text to parse.

    Returns
    -------
    out[0] : string, dict, list, boolean, or other type
        The parsed text.
    out[1] : boolean
        True if we managed to parse the text, False otherwise.

    """
    parsed = None
    success = False
    try:
        parsed = ast.literal_eval(text.strip())
        success = True
    except SyntaxError:
        parsed = text.strip()
        success = True
    except ValueError:
        parsed = text.strip()
        success = True
    return parsed, success

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

def cp2k_settings(settings, input_path):
    """Read and processes cp2k settings.

    Parameters
    ----------
    settings : dict
        The current input settings..
    input_path : string
        The CP2K input path

    """
    ext = settings['engine'].get('cp2k_format', 'xyz')
    default_files = {'conf': f'initial.{ext}',
                     'template': 'cp2k.inp'}
    settings['engine']['input_files'] = {}
    for key in ('conf', 'template'):
        # Add input path and the input files if input is no given:
        settings['engine']['input_files'][key] = \
            settings['engine'].get(key,
                                   os.path.join(input_path,
                                                default_files[key]))
    nodes = read_cp2k_input(
        settings['engine']['input_files']['template'])
    MD_data = set_parents(nodes)['MOTION->MD'].data

    # Checks temperature
    cp2k_temp_inp = False
    for lines in MD_data:
        if lines.startswith('TEMPERATURE'):
            cp2k_temp_inp = True
            cp2k_temp = float(lines.split()[1])
            if 'temperature' not in settings['system']:
                settings['system']['temperature'] = cp2k_temp
            elif abs(cp2k_temp - settings['system']['temperature']) < 10e-8:
                pass
            else:
                msg = 'Inequal Pyretis and CP2K input temperatures!' +\
                        ' Temperature is set to CP2K input temperature.'
                logger.warning(msg)
                settings['system']['temperature'] = cp2k_temp
            break
    if not cp2k_temp_inp:
        if 'temperature' not in settings['system']:
            msg = 'Temperature not set in CP2K file!' +\
                ' Temperature is set to CP2K default temperature.'
            logger.warning(msg)
        elif abs(300.0 - settings['system']['temperature']) > 10e-8:
            msg = 'Temperature not set in CP2K file!' +\
                ' And temperature in input rst file does not equal 300.0 K!' +\
                ' Temperature is set to CP2K default temperature.'
            logger.warning(msg)
        # CP2K defaults to 300K when temperature is unspecified.
        settings['system']['temperature'] = 300.0

def fill_up_tis_and_retis_settings(settings):
    """Make the life of sloppy users easier.

    The full input set-up will be here completed.

    Parameters
    ----------
    settings : dict
        The current input settings.

    Returns
    -------
    None, but this method might add data to the input settings.

    """
    create_empty_ensembles(settings)
    ensemble_save = copy.deepcopy(settings['ensemble'])

    # The previously constructed dictionary is inserted in the settings.
    # This is done such that the specific input given per ensemble
    # OVERWRITES the general input.
    for i_ens, val in enumerate(ensemble_save):
        for key in settings:
            if key in val:
                if key not in SPECIAL_MULTIPLE:
                    val[key] = {**copy.deepcopy(settings[key]),
                                **copy.deepcopy(val[key])}
                else:
                    for i_sub, sub in enumerate(settings[key]):
                        while len(val[key]) < len(settings[key]):
                            val[key].append({})
                        val[key][i_sub] = {
                            **copy.deepcopy(sub),
                            **copy.deepcopy(val[key][i_sub])
                            }

        ensemble_save[i_ens] = {**copy.deepcopy(settings),
                                **copy.deepcopy(val)}
        del ensemble_save[i_ens]['ensemble']

    for i_ens, ens in enumerate(ensemble_save):
        add_default_settings(settings)
        add_specific_default_settings(settings)
        settings['ensemble'][i_ens] = copy.deepcopy(ens)
        if 'make-tis-files' in settings['simulation']['task']:
            settings['ensemble'][i_ens]['simulation']['task'] = 'tis'
    if settings['tis'].get('shooting_moves', False):
        for i_ens, ens_set in enumerate(settings['ensemble']):
            ens_set['tis']['shooting_move'] = \
                settings['tis']['shooting_moves'][i_ens]

def create_empty_ensembles(settings):
    """Create missing ensembles in the settings.

    Checks the input and allocate it to the right ensemble. In theory
    inouts shall include all these info, but it is not practical.

    Parameters
    ----------
    settings : dict
        The current input settings.

    Returns
    -------
    None, but this method might add data to the input settings.

    """
    ints = settings['simulation']['interfaces']

    # Determine how many ensembles are needed.
    add0 = 0
    if not settings['simulation'].get('flux', False):
        add0 += 1
    if not settings['simulation'].get('zero_ensemble', True):
        add0 += 1

    # if some ensembles have inputs, they need to be kept.
    if 'ensemble' in settings:
        orig_set = settings['ensemble'].copy()
    else:
        orig_set = []

    settings['ensemble'] = []
    add = add0

    # if in the main settings an ensemble_number is defined, then only
    # that ensemble will be considered.
    if 'tis' in settings:
        idx = settings['tis'].get('ensemble_number')
        detect = settings['tis'].get('detect', ints[2])
        if idx is not None:
            settings['ensemble'].append({'interface': ints[1],
                                         'tis': {'ensemble_number': idx,
                                                 'detect': detect}})
            for sav in orig_set:
                settings['ensemble'][0] = {**settings['ensemble'][0], **sav}
            return
    # if one wants to compute the flux, the 000 ensemble is for it
    # todo remove this labelling mismatch, and give to the flux
    # a flux name folder (instead of 000), and leave 000 for the O^+ ens.
    if settings['simulation'].get('flux', False):
        settings['ensemble'].append({'interface': ints[0],
                                     'tis': {'ensemble_number': 0,
                                             'detect': ints[1]}})
        add = add0 + 1
    for i in range(add, len(ints)):
        settings['ensemble'].append({'interface': ints[i - 1],
                                     'tis': {'ensemble_number': i,
                                             'detect': ints[i]}})

    # create the ensembles in setting, keeping eventual inputs.
    # nb. in the settings, specific input for an ensemble can be now given.
    for i_ens, ens in enumerate(settings['ensemble']):
        for sav in orig_set:
            if 'tis' in sav and 'ensemble_number' in sav['tis']:
                if ens['tis']['ensemble_number'] ==\
                        sav['tis']['ensemble_number']:
                    settings['ensemble'][i_ens].update(sav)
            elif ens['interface'] == sav['interface']:
                settings['ensemble'][i_ens].update(sav)

    return

def is_sorted(lll):
    """Check if a list is sorted."""
    return all(aaa <= bbb for aaa, bbb in zip(lll[:-1], lll[1:]))

def _clean_settings(settings):
    """Clean up input settings.

    Here, we attempt to remove unwanted stuff from the input settings.

    Parameters
    ----------
    settings : dict
        The current input settings.

    Returns
    -------
    settingsc : dict
        The cleaned input settings.

    """
    settingc = {}
    # Add other sections:
    for sec in settings:
        if sec not in SECTIONS:  # Well, ignore unknown ones:
            msgtxt = f'Ignoring unknown section "{sec}"'
            logger.warning(msgtxt)
            continue
        if sec in SPECIAL_MULTIPLE:
            settingc[sec] = list(settings[sec])
        else:
            settingc[sec] = {}
            if sec in ALLOW_MULTIPLE:  # Here, just add multiple sections:
                for key in settings[sec]:
                    settingc[sec][key] = settings[sec][key]
            else:
                for key in settings[sec]:
                    if key not in SECTIONS[sec]:  # Ignore junk:
                        msgtxt = f'Ignoring unknown "{key}" in "{sec}"'
                        logger.warning(msgtxt)
                    else:
                        settingc[sec][key] = settings[sec][key]
    to_remove = [key for key, val in settingc.items() if len(val) == 0]
    for key in to_remove:
        settingc.pop(key, None)
    return settingc
def compare_objects(obj1, obj2, attrs, numpy_attrs=None):
    if not obj1.__class__ == obj2.__class__:
        logger.debug(
            'The classes are different %s != %s',
            obj1.__class__, obj2.__class__
        )
        return False
    if not len(obj1.__dict__) == len(obj2.__dict__):
        logger.debug('Number of attributes differ.')
        return False
    # Compare the requested attributes:
    for key in attrs:
        try:
            val1 = getattr(obj1, key)
            val2 = getattr(obj2, key)
        except AttributeError:
            logger.debug('Failed to compare attribute "%s"', key)
            return False
        if numpy_attrs and key in numpy_attrs:
            if not numpy_allclose(val1, val2):
                logger.debug('Attribute "%s" differ.', key)
                return False
        else:
            if not val1 == val2:
                logger.debug('Attribute "%s" differ.', key)
                return False
    return True
