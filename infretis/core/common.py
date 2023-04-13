from infretis.classes.external.gromacs import GromacsEngine
from infretis.classes.particles import Particles
from infretis.classes.system import System

from pyretis.inout.screen import print_to_screen
from collections import deque
import os
import sys
import numpy as np
import inspect
import importlib
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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


def check_engine(settings):
    """Check the engine settings.

    Checks that the input engine settings are correct, and
    automatically determine the 'internal' or 'external'
    engine setting.

    Parameters
    ----------
    settings : dict
        The current input settings.

    """
    msg = []
    if 'engine' not in settings:
        msg += ['The section engine is missing']

    elif settings['engine'].get('type') == 'external':

        if 'input_path' not in settings['engine']:
            msg += ['The section engine requires an input_path entry']

        if 'gmx' in settings['engine'] and \
                'gmx_format' not in settings['engine']:
            msg += ['File format is not specified for the engine']
        elif 'cp2k' in settings['engine'] and \
                'cp2k_format' not in settings['engine']:
            msg += ['File format is not specified for the engine']

    if msg:
        msgtxt = '\n'.join(msg)
        logger.critical(msgtxt)
        return False

    return True

def prepare_engine(settings):
    """Create an engine from given settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    engine : object like :py:class:`.engine`
        This object will correspond to the selected simulation type.

    """
    if settings.get('engine', {}).get('obj', False):
        return settings['engine']['obj']

    logtxt = units_from_settings(settings)
    print_to_screen(logtxt, level='info')
    logger.info(logtxt)

    check_engine(settings)
    engine = create_engine(settings)
    logtxt = f'Created engine "{engine}" from settings.'
    print_to_screen(logtxt, level='info')
    logger.info(logtxt)
    return engine

def create_engine(settings):
    """Create an engine from settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    out : object like :py:class:`.EngineBase`
        This object represents the engine.

    """
    engine = create_external(settings, 'engine', engine_factory,
                             ['integration_step'])
    if not engine:
        raise ValueError('Could not create engine from settings!')
    return engine


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
        try:
            key_settings = settings[key]
        except KeyError:
            logger.debug('No "%s" setting found. Skipping set-up', key)
            return None
    module = key_settings.get('module', None)
    klass = None
    try:
        klass = key_settings['class']
    except KeyError:
        logger.debug('No "class" setting for "%s" specified. Skipping set-up',
                     key)
        return None
    if module is None:
        return factory(key_settings)
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is ok or if we should include
    # the 'exe_path' from settings.
    # 1) Check if we can find the module:
    if os.path.isfile(module):
        obj = import_from(module, klass)
    else:
        if 'exe_path' in settings['simulation']:
            module = os.path.join(settings['simulation']['exe_path'],
                                  module)
            obj = import_from(module, klass)
        else:
            msg = 'Could not find module "{}" for {}!'.format(module, key)
            raise ValueError(msg)
    # run some checks:
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

def engine_factory(settings):
    """Create an engine according to the given settings.

    This function is included as a convenient way of setting up and
    selecting an engine. It will return the created engine.

    Parameters
    ----------
    settings : dict
        This defines how we set up and select the engine.

    Returns
    -------
    out : object like :py:class:`.EngineBase`
        The object representing the engine to use in a simulation.

    """
    engine_map = {
        'gromacs2': {'cls': GromacsEngine},
    }
    return generic_factory(settings, engine_map, name='engine')

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
