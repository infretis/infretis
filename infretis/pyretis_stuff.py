from pyretis.core.retis import make_retis_step
from pyretis.core.random_gen import create_random_generator
from pyretis.core.common import soft_partial_exit, priority_checker
from pyretis.inout.common import make_dirs
from pyretis.initiation import initiate_path_simulation
from pyretis.inout.restart import write_ensemble_restart
from pyretis.inout.screen import print_to_screen
from pyretis.inout.simulationio import task_from_settings
from pyretis.simulation.simulation import Simulation
from datetime import datetime
import signal
from time import sleep
import shlex
import shutil
import subprocess
import os
import struct
import re
import collections
import inspect
import ast
from copy import copy as copy0
from copy import deepcopy
import importlib
import sys
import logging
from abc import ABCMeta, abstractmethod
from collections import deque
import numpy as np
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Define formats for the trajectory output:
TRR_HEAD_SIZE = 1000
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
TRR_DATA_ITEMS = ('box_size', 'vir_size', 'pres_size',
                  'x_size', 'v_size', 'f_size')
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
    'umbrella': None,
    'overlap': None,
    'maxdx': None,
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
SECTIONS['analysis'] = {
    'blockskip': 1,
    'bins': 100,
    'maxblock': 1000,
    'maxordermsd': -1,
    'ngrid': 1001,
    'plot': {
        'plotter': 'mpl',
        'output': 'png',
        'style': 'pyretis' },
    'report': [
        'latex',
        'rst',
        'html'],
    'report-dir': None,
    'skipcross': 1000,
    'txt-output': 'txt.gz',
    'tau_ref_bin': [],
    'skip': 0 }
SPECIAL_KEY = {
    'parameter'}
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

_STATUS = {
    'ACC': 'The path has been accepted',
    'MCR': 'Momenta change rejection',
    'BWI': 'Backward trajectory end at wrong interface',
    'BTL': 'Backward trajectory too long (detailed balance condition)',
    'BTX': 'Backward trajectory too long (max-path exceeded)',
    'BTS': 'Backward trajectory too short',
    'EWI': 'Initial path ends at wrong interface',
    'EXP': 'Exploration path',
    'FTL': 'Forward trajectory too long (detailed balance condition)',
    'FTX': 'Forward trajectory too long (max-path exceeded)',
    'FTS': 'Forward trajectory too short',
    'HAS': 'High Acceptance Swap rejection for SS/WF detailed balance',
    'KOB': 'Kicked outside of boundaries',
    'NCR': 'No crossing with middle interface',
    'NSG': 'Path has no suitable segments',
    'NSS': 'No one-step crossing in stone skipping',
    'SSA': 'Stone Skipping super detailed balance rejection',
    'WFA': 'Wire Fencing super detailed balance rejection',
    'SWI': 'Initial path starts at wrong interface',
    'WTA': 'Web Throwing super detailed balance rejection',
    'TSS': 'Target swap selection rejection',
    'TSA': 'Target swap detailed balance rejection',
    'XSS': 'SS sub path too long in stone skipping',
    '0-L': 'Path in the {0-} ensemble ends at the left interface',
}

_GENERATED = {
    'sh': 'Path was generated with a shooting move',
    'is': 'Path was generated by shooting initially prior to Stone Skipping',
    'tr': 'Path was generated with a time-reversal move',
    'ki': 'Path was generated by integration after kicking',
    're': 'Path was loaded from formatted external file(s)',
    'ld': 'Path was loaded from unformatted external file(s)',
    's+': 'Path was generated by a swapping move from +',
    's-': 'Path was generated by a Swapping move from -',
    'ss': 'Path was generated by Stone Skipping',
    'wt': 'Path was generated by Web Throwing',
    'wf': 'Path was generated by Wire Fencing',
    '00': 'Path was generated by a null move',
    'mr': 'Path was generated by a mirror move',
    'ts': 'Path was generated by a target swap move',
}


# Short versions of the moves:
_GENERATED_SHORT = {
    'sh': 'Shoot',
    'tr': 'Time-reversal',
    'ki': '"Kick" initiation',
    're': 'Formatted ext file load',
    'ld': 'Unformatted ext file load',
    's+': 'Swap from +',
    's-': 'Swap from -',
    'ss': 'Stone skipping',
    'wt': 'Web Throwing',
    'wf': 'Wire Fencing',
    '00': 'Null',
    'mr': 'mirror',
    'ts': 'target swap',
}

def read_gromacs_generic(filename):
    '''Read GROMACS files.

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

    '''
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

    Examples
    --------
    >>> from pyretis.inout.formats.gromacs import read_gromacs_file
    >>> for snapshot in read_gromacs_file('traj.gro'):
    ...     print(snapshot['x'][0])

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
    --------
    >>> from pyretis.inout.formats.xyz import read_xyz_file
    >>> for snapshot in read_xyz_file('traj.xyz'):
    ...     print(snapshot['x'][0])

    Note
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

def get_box_from_header(header):
    """Get box lengths from a text header.

    Parameters
    ----------
    header : string
        Header from which we will extract the box.

    Returns
    -------
    out : numpy.array or None
        The box lengths.

    """
    low = header.lower()
    if low.find('box:') != -1: 
        txt = low.split('box:')[1].strip()
        return np.array([float(i) for i in txt.split()])
    return None


def list_get(input_list, index):
    """Get an item from a list and handle out-of bounds errors.

    This method is intended to be used when we are picking items from
    a list and possibly we want a number of items which is larger than
    the number of items in the list. Here, we then just return the last
    element.

    Parameters
    ----------
    input_list : list
        The list to pick from.
    index : integer
        The index to pick.

    """
    try:
        return input_list[index]
    except IndexError:
        return input_list[-1]


def array_to_box_matrix(cell):
    """Return a box matrix corresponding to a cell array.

    Parameters
    ----------
    cell : list or numpy.array
        An (1D) array containing 1, 2, 3, 6 or 9 items. These are
        the xx, yy, zz, xy, xz, yx, yz, zx, zy elements. Setting
        x = 0, y = 1 and z = 2 will give the indices in the matrix,
        e.g. yx -> (1, 0) will correspond to the item in row 1 and
        column 0.

    Returns
    -------
    box : numpy.array (2D)
        The box vector on matrix form.

    """
    if len(cell) == 1:
        return 1.0 * np.array([cell[0]])
    if len(cell) == 2:
        return 1.0 * np.array([[cell[0], 0.0],
                               [0.0, cell[1]]])
    if len(cell) == 3:
        return 1.0 * np.array([[cell[0], 0.0, 0.0],
                               [0.0, cell[1], 0.0],
                               [0.0, 0.0, cell[2]]])
    if len(cell) == 6:
        return 1.0 * np.array([[cell[0], cell[3], cell[4]],
                               [0.0, cell[1], cell[5]],
                               [0.0, 0.0, cell[2]]])
    if len(cell) == 9:
        return 1.0 * np.array([[cell[0], cell[3], cell[4]],
                               [cell[5], cell[1], cell[6]],
                               [cell[7], cell[8], cell[2]]])
    logger.error(
        '%d box parameters given, need 1, 2, 3, 6, or 9.', len(cell)
    )
    raise ValueError('Incorrect number of box-parameters!')

def check_consistency(low, high, length):
    """Check that given box bounds are consistent.

    Parameters
    ----------
    low : numpy.array
        The lower bounds for the box.
    high : numpy.array
        The upper bounds for the box.
    length : numpy.array
        The lengths of the box.

    """
    length_re = high - low
    if any(i <= 0 for i in length_re):
        logger.error('Check box settings. Found high <= low.')
        raise ValueError('Incorrect box: high <= low.')
    if not all(np.isclose(length, length_re)):
        logger.error('Check box settings: length != high - low.')
        raise ValueError('Incorrect box: length != high - low.')

def _get_low_high_length(low, high, length, periodic):
    """Determine box cell parameters from input.

    This method will consider the following cases:

    1) We are given low, high and length.
    2) We are given high and length, and determine the low values.
    3) We are given low and length, and determine the high values.
    4) We are given length, assume low to be zero and determine high.
    5) We are given low and high, and determine the length.
    6) We are given just high, assume low to be zero and
       determine the length.
    7) We are just given low, and assume high and length to be infinite.
    8) We are given none of the values, and assume all to infinite.

    Parameters
    ----------
    low : numpy.array or None
        The lower bounds for the box.
    high : numpy.array or None
        The upper bounds for the box.
    length : numpy.array or None
        The lengths of the box.
    periodic : list of boolean or None
        We will assume a periodic box for dimensions where
        this list is True.

    Returns
    -------
    out[0] : numpy.array
        The updated lower bounds for the box.
    out[1] : numpy.array
        The updated upper bounds for the box.
    out[2] : numpy.array
        The updated lengths of the box.
    out[3] : list of boolean
        The updated periodic settings for the box.

    """
    case = (length is not None, low is not None, high is not None)
    if low is not None:
        low = np.array(low)
    if high is not None:
        high = np.array(high)
    if length is not None:
        length = np.array(length)
    if case == (True, True, True):
        # 1) We have given length, low and high.
        pass
    elif case == (True, False, True):
        # 2) Length & high has been given, just determine low:
        low = high - length
    elif case == (True, True, False):
        # 3) Length and low was given, determine high:
        high = low + length
    elif case == (True, False, False):
        # 4) Length is given, set low to 0 and high to low + length:
        low = np.zeros_like(length)
        high = low + length
    elif case == (False, True, True):
        # 5) Low and high is given, determine length:
        length = high - low
    elif case == (False, False, True):
        # 6) High is given, assume low and determine length:
        low = np.zeros_like(high)
        length = high - low
    elif case == (False, True, False):
        # 7) Low given. High and length to be determined.
        # This is not enough info, so we assume an infinite box:
        length = float('inf') * np.ones_like(low)
        high = float('inf') * np.ones_like(low)
    elif case == (False, False, False):
        # Not much info is given. We let the box be similar
        # in shape to the input periodic settings:
        if periodic is None:
            logger.info(
                'Too few settings for the box is given. A 1D box is assumed.'
            )
            periodic = [False]
        low = np.array([-float('inf') for _ in periodic])
        high = float('inf') * np.ones_like(low)
        length = float('inf') * np.ones_like(low)
    return low, high, length, periodic

def create_box(low=None, high=None, cell=None, periodic=None):
    """Set up and create a box.

    Parameters
    ----------
    low : numpy.array, optional
        1D array containing the lower bounds of the cell.
    high : numpy.array, optional
        1D array containing the higher bounds of the cell.
    cell : numpy.array, optional
        1D array, a flattened version of the simulation box matrix.
        This array is expected to contain 1, 2, 3, 6 or 9 items.
        These are the xx, yy, zz, xy, xz, yx, yz, zx, zy elements,
        respectively.
    periodic : list of boolean, optional
        If `periodic[i]` then we should apply periodic boundaries
        to dimension `i`.

    Returns
    -------
    out : object like :py:class:`.BoxBase`
        The object representing the simulation box.

    """
    # If the cell is given, the we should be able to determine
    # the length:
    if cell is not None:
        # make sure the cell does not become an array of objects
        cell = np.array(cell, dtype=float)

        if len(cell) <= 3:
            length = np.array([i for i in cell])
        else:  # Use the xx, yy and zz parameters:
            length = np.array([i for i in cell[:3]])
        # Determine low, high, length and possible periodic:
        low, high, length, periodic = _get_low_high_length(
            low, high, length, periodic
        )
    else:  # Here, cell was not given. Try to obtain it from the length:
        low, high, length, periodic = _get_low_high_length(
            low, high, cell, periodic
        )
        cell = np.array([i for i in length])
    # We can still have periodic not set:
    if periodic is None:
        logger.info(
            'Periodic settings not given. Assumed True for all directions.'
        )
        periodic = [True] * len(length)
    else:
        # If for some reason the periodic settings have wrong length:
        if len(periodic) < len(length):
            logger.info('Setting missing periodic settings to True.')
            for _ in range(len(length) - len(periodic)):
                periodic.append(True)
        elif len(periodic) > len(length):
            logger.error('Too many periodic settings given.')
            raise ValueError('Too many periodic settings given.')
    # Here, everything should be set:
    check_consistency(low, high, length)
    # Create the box:
    obj = TriclinicBox
    if len(cell) <= 3:
        obj = RectangularBox
    return obj(low, high, length, periodic, cell)

class BoxBase(metaclass=ABCMeta):
    """Class for a generic simulation box.

    This class defines a generic simulation box.

    Attributes
    ----------
    low : numpy.array
        1D array containing the lower bounds of the cell.
    high : numpy.array
        1D array containing the higher bounds of the cell.
    length : numpy.array
        1D array containing the length of the sides of the
        simulation box.
    ilength : numpy.array
        1D array containing the inverse box lengths for the
        simulation box.
    periodic : list of boolean
        If `periodic[i]` then we should apply periodic boundaries
        to dimension `i`.
    box_matrix : numpy.array
        2D matrix, representing the simulation cell.
    cell : numpy.array
        1D array representing the simulation cell (flattened
        version of the 2D box matrix).

    """

    def __init__(self, low, high, length, periodic, cell):
        """Initialise the BoxBase class."""
        self.low = low 
        self.high = high
        self.length = length
        self.periodic = periodic
        self.cell = cell
        # Create box matrix from the given cell:
        self.box_matrix = array_to_box_matrix(self.cell)
        self.ilength = 1.0 / self.length
        self.dim = len(self.length)

    def update_size(self, new_size):
        """Update the box size.

        Parameters
        ----------
        new_size : list, tuple, numpy.array, or other iterable
            The new box size.

        """
        if new_size is None:
            logger.warning(
                'Box update ignored: Tried to update with empty size!'
            )
        else:
            try:
                size = new_size.size
            except AttributeError:
                size = len(new_size)
            if size <= 3:
                if size == len(self.cell):
                    for i in range(self.dim):
                        self.length[i] = new_size[i]
                        self.high[i] = self.low[i] + new_size[i]
                        self.cell[i] = new_size[i]
                    self.ilength = 1.0 / self.length
            else:
                try:
                    self.box_matrix = array_to_box_matrix(new_size)
                    self.cell = [i for i in new_size]
                    self.length = np.array([float(i) for i in self.cell[:3]])
                    self.high = self.low + self.length
                    self.ilength = 1.0 / self.length
                except ValueError:
                    logger.critical('Box update failed!')

    def bounds(self):
        """Return the boundaries of the box (low, high) as an array."""
        return [(i, j) for i, j in zip(self.low, self.high)]

    @abstractmethod
    def calculate_volume(self):
        """Return the volume of the box."""
        return

    @abstractmethod
    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only.

        For the given positions, this function will apply periodic
        boundary conditions to one dimension only. This can be useful
        for instance in connection with order parameters.

        Parameters
        ----------
        pos : float
            Coordinate to wrap.
        dim : int
            This selects the dimension to consider.

        """
        return

    @abstractmethod
    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position.

        Parameters
        ----------
        pos : nump.array
            Positions to apply periodic boundaries to.

        Returns
        -------
        out : numpy.array, same shape as parameter `pos`
            The periodic-boundary wrapped positions.

        """
        return
    @abstractmethod
    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector.

        Parameters
        ----------
        distance : numpy.array
            The distance vectors.

        Returns
        -------
        out : numpy.array, same shape as parameter `distance`
            The periodic-boundary-wrapped distances.

        """
        return

    @abstractmethod
    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance.

        This will apply periodic boundaries to a distance. Note that the
        distance can be a vector, but not a matrix of distance vectors.

        Parameters
        ----------
        distance : numpy.array with shape `(self.dim,)`
            A distance vector.

        Returns
        -------
        out : numpy.array, same shape as parameter `distance`
            The periodic-boundary wrapped distance vector.

        """
        return

    def print_length(self, fmt=None):
        """Return a string with box lengths. Can be used for output."""
        if fmt is None:
            return ' '.join(('{}'.format(i) for i in self.cell))
        return ' '.join((fmt.format(i) for i in self.cell))

    def restart_info(self):
        """Return a dictionary with restart information."""
        info = {
            'length': self.length,
            'periodic': self.periodic,
            'low': self.low,
            'high': self.high,
            'cell': self.cell,
        }
        return info
    def copy(self):
        """Return a copy of the box.

        Returns
        -------
        out : object like :py:class:`.BoxBase`
            A copy of the box.

        """
        box_copy = self.__class__(
            np.copy(self.low),
            np.copy(self.high),
            np.copy(self.length),
            copy0(self.periodic),
            np.copy(self.cell)
        )
        return box_copy

    def load_restart_info(self, info):
        """Read the restart information."""
        self.length = info.get('length')
        self.periodic = info.get('periodic')
        self.low = info.get('low')
        self.high = info.get('high')
        self.cell = info.get('cell')

    def __str__(self):
        """Return a string describing the box.

        Returns
        -------
        out : string
            String with the type of box, the extent of the box and
            information about the periodicity.

        """
        boxstr = []
        if len(self.cell) <= 3:
            boxstr.append('Orthogonal box:')
        else:
            boxstr.append('Triclinic box:')
        for i, periodic in enumerate(self.periodic):
            low = self.low[i]
            high = self.high[i]
            msg = 'Dim: {}, Low: {}, high: {}, periodic: {}'
            boxstr.append(msg.format(i, low, high, periodic))
        cell = self.print_length()
        boxstr.append('Cell: {}'.format(cell))
        return '\n'.join(boxstr)

    def __eq__(self, other):
        """Compare two box objects."""
        attrs = {'low', 'high', 'length', 'ilength', 'box_matrix', 'cell',
                 'periodic', 'dim'}
        numpy_attrs = {'low', 'high', 'length', 'ilength', 'box_matrix',
                       'cell'}
        return compare_objects(self, other, attrs, numpy_attrs)
    def __ne__(self, other):
        """Compare two box objects."""
        return not self == other

class RectangularBox(BoxBase):
    """An orthogonal box."""

    def calculate_volume(self):
        """Calculate the volume of the box.

        Returns
        -------
        out : float
            The volume of the box.

        """
        return product(self.length)

    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only.

        For the given positions, this function will apply periodic
        boundary conditions to one dimension only. This can be useful
        for instance in connection with order parameters.

        Parameters
        ----------
        pos : float
            Coordinate to wrap around.
        dim : int
            This selects the dimension to consider.

        """
        if self.periodic[dim]:
            low, length = self.low[dim], self.length[dim]
            ilength = self.ilength[dim]
            relpos = pos - low
            delta = relpos
            if relpos < 0.0 or relpos >= length:
                delta = relpos - np.floor(relpos * ilength) * length
            return delta + low
        return pos

    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position.

        Parameters
        ----------
        pos : nump.array
            Positions to apply periodic boundaries to.

        Returns
        -------
        out : numpy.array, same shape as parameter `pos`
            The periodic-boundary wrapped positions.

        """
        pbcpos = np.zeros(pos.shape)
        for i, periodic in enumerate(self.periodic):
            if periodic:
                low = self.low[i]
                length = self.length[i]
                ilength = self.ilength[i]
                relpos = pos[:, i] - low
                delta = np.where(
                    np.logical_or(relpos < 0.0, relpos >= length),
                    relpos - np.floor(relpos * ilength) * length,
                    relpos
                    )
                pbcpos[:, i] = delta + low
            else:
                pbcpos[:, i] = pos[:, i]
        return pbcpos

    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector.

        Parameters
        ----------
        distance : numpy.array
            The distance vectors.

        Returns
        -------
        out : numpy.array, same shape as the `distance` parameter
            The pbc-wrapped distances.

        Note
        ----
        This will modify the given input matrix inplace. This can be
        changed by setting ``pbcdist = np.copy(distance)``.

        """
        pbcdist = distance
        for i, (periodic, length, ilength) in enumerate(zip(self.periodic,
                                                            self.length,
                                                            self.ilength)):
            if periodic:
                dist = pbcdist[:, i]
                high = 0.5 * length
                k = np.where(np.abs(dist) >= high)[0]
                dist[k] -= np.rint(dist[k] * ilength) * length
        return pbcdist

    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance.

        This will apply periodic boundaries to a distance. Note that the
        distance can be a vector, but not a matrix of distance vectors.

        Parameters
        ----------
        distance : numpy.array with shape `(self.dim,)`
            A distance vector.

        Returns
        -------
        out : numpy.array, same shape as the `distance` parameter
            The periodic-boundary wrapped distance vector.

        """
        pbcdist = np.zeros(distance.shape)
        for i, (periodic, length, ilength) in enumerate(zip(self.periodic,
                                                            self.length,
                                                            self.ilength)):
            if periodic and np.abs(distance[i]) > 0.5*length:
                pbcdist[i] = (distance[i] -
                              np.rint(distance[i] * ilength) * length)
            else:
                pbcdist[i] = distance[i]
        return pbcdist

class TriclinicBox(BoxBase):
    """This class represents a triclinic box."""

    def calculate_volume(self):
        """Calculate and return the volume of the box.

        Returns
        -------
        out : float
            The volume of the box.

        """
        return det(self.box_matrix)

    def pbc_coordinate_dim(self, pos, dim):
        """Apply periodic boundaries to a selected dimension only."""
        raise NotImplementedError

    def pbc_wrap(self, pos):
        """Apply periodic boundaries to the given position."""
        raise NotImplementedError

    def pbc_dist_matrix(self, distance):
        """Apply periodic boundaries to a distance matrix/vector."""
        raise NotImplementedError

    def pbc_dist_coordinate(self, distance):
        """Apply periodic boundaries to a distance."""
        raise NotImplementedError

def set_up_box(settings, boxs, dim=3):
    """Set up a box from given settings.

    Parameters
    ----------
    settings : dict
        The dict with the simulation settings.
    boxs : dict or None
        If no box settings are given, we can still create a box,
        inferred from the positions of the particles. This dict
        contains the settings to do so.
    dim : integer, optional
        Number of dimensions for the box. This is used only as a last
        resort when no information about the box is given.

    Returns
    -------
    box : object like :py:class:`.BoxBase` or None
        The box if we managed to create it, otherwise None.

    """
    msg = 'Box created {}:\n{}'
    box = None
    if settings.get('box', None) is not None:
        box = create_box(**settings['box'])
        msgtxt = msg.format('from settings', box)
        logger.info(msgtxt)
        debugtxt = 'Settings used:\n{}'.format(settings['box'])
        logger.debug(debugtxt)
    else:
        if boxs is not None:
            box = create_box(**boxs)
            msgtxt = msg.format('from initial positions', box)
            logger.info(msgtxt)
            msgwarn = 'The box was assumed periodic in all directions.'
            logger.warning(msgwarn)
        else:
            if dim > 0:
                box = create_box(periodic=[False]*dim)
                msgtxt = msg.format('without specifications', box)
                logger.info(msgtxt)
                msgwarn = 'The box was assumed nonperiodic in all directions.'
                logger.warning(msgwarn)
    return box

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


class PathEnsemble:
    """Representation of a path ensemble.

    This class represents a collection of paths in a path ensemble.
    In general, paths may be "long and complicated" so here, we really
    just store a simplified abstraction of the path, which is obtained
    by the `Path.get_path_data()` function for a given `Path` object.
    The returned dictionary is stored in the list `PathEnsemble.paths`.
    The only full path we store is the last accepted path. This is
    convenient for the RETIS method where paths may be swapped between
    path ensembles.

    Attributes
    ----------
    ensemble_number : integer
        This integer is used to represent the path ensemble, for RETIS
        simulations it's useful to identify the path ensemble. The path
        ensembles are numbered sequentially 0, 1, 2, etc. This
        corresponds to ``[0^-]``, ``[0^+]``, ``[1^+]``, etc.
    ensemble_name : string
        A string which can be used for printing the ensemble name.
        This is of form ``[0^-]``, ``[0^+]``, ``[1^+]``, etc.
    ensemble_name_simple : string
        A string with a simpler representation of the ensemble name,
        can be used for creating output files etc.
    interfaces : list of floats
        Interfaces, specified with the values for the
        order parameters: `[left, middle, right]`.
    paths : list
        This list contains the stored information for the paths. Here
        we only store the data returned by calling the `get_path_data()`
        function of the `Path` object.
    nstats : dict of ints
        This dict store some statistics for the path ensemble. The keys
        are:

        * npath : The number of paths stored.
        * nshoot : The number of accepted paths generated by shooting.
        * ACC, BWI, ... : Number of paths with given status (from `_STATUS`).
    maxpath : int
        The maximum number of paths to store.
    last_path : object like :py:class:`.PathBase`
        This is the last **accepted** path.

    """

    def __init__(self, ensemble_number, interfaces,
                 rgen=None, maxpath=10000, exe_dir=None):
        """Initialise the PathEnsemble object.

        Parameters
        ----------
        ensemble_number : integer
            An integer used to identify the ensemble.
        interfaces : list of floats
            These are the interfaces specified with the values
            for the order parameters: ``[left, middle, right]``.
        rgen : object like :py:class:`.RandomGenerator`, optional
            The random generator that will be used for the
            paths that required random numbers.
        maxpath : integer, optional
            The maximum number of paths to store information for in memory.
            Note, that this will not influence the analysis as long as
            you are using the output files when running the analysis.
        exe_dir : string, optional
            The base folder where the simulation was executed from.
            This is used to set up output directories for the path
            ensemble.

        """
        if rgen is None:
            rgen = create_random_generator()
        self.rgen = rgen
        self.ensemble_number = ensemble_number
        self.path_number = None
        self.interfaces = tuple(interfaces)  # Should not change interfaces.
        self.last_path = None
        self.nstats = {'npath': 0, 'nshoot': 0, 'ACC': 0}
        self.paths = []
        self.maxpath = maxpath
        if self.ensemble_number == 0:
            self.ensemble_name = '[0^-]'
            self.start_condition = 'R'
        else:
            ensemble_number = self.ensemble_number - 1
            self.ensemble_name = f'[{ensemble_number}^+]'
            self.start_condition = 'L'
        self.ensemble_name_simple = generate_ensemble_name(
            self.ensemble_number
        )
        self.directory = collections.OrderedDict()
        self.directory['path_ensemble'] = None
        self.directory['accepted'] = None
        self.directory['generate'] = None
        self.directory['traj'] = None
        self.directory['exe_dir'] = exe_dir
        if exe_dir is not None:
            path_dir = os.path.join(exe_dir, self.ensemble_name_simple)
            self.update_directories(path_dir)
            self.directory['home_dir'] = exe_dir


    def __eq__(self, other):
        """Check if two path_ensemble are equal."""
        equal = True
        if self.__class__ != other.__class__:
            logger.debug('%s and %s.__class__ differ', self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug('%s and %s.__dict__ differ', self, other)
            equal = False

        for i in ['directory', 'interfaces', 'nstats', 'paths']:
            if hasattr(self, i):
                for j, k in zip(getattr(self, i), getattr(other, i)):
                    if j != k:
                        logger.debug('%s for %s and %s attributes are %s and '
                                     '%s', i, self, other, j, k)
                        equal = False

        for i in ['ensemble_name',
                  'ensemble_name_simple', 'ensemble_number',
                  'maxpath', 'start_condition']:
            if hasattr(self, i):
                if getattr(self, i) != getattr(other, i):
                    logger.debug('%s for %s and %s, attributes are %s and %s',
                                 i, self, other,
                                 getattr(self, i), getattr(other, i))
                    equal = False

        if hasattr(self, 'last_path'):
            if self.last_path != other.last_path:
                logger.debug('last paths differs')
                equal = False

        if hasattr(self, 'rgen'):
            if self.rgen.__class__ != other.rgen.__class__:
                logger.debug('self.rgen.__class__ differs')
                return False
            if self.rgen.__dict__['seed'] != other.rgen.__dict__['seed']:
                logger.debug('rgen seed differs')
                equal = False
            if not big_fat_comparer(self.rgen.__dict__['rgen'].get_state(),
                                    other.rgen.__dict__['rgen'].get_state()):
                logger.debug('rgen differs')
                equal = False

        return equal

    def __ne__(self, other):
        """Check if two paths are not equal."""
        return not self == other

    def directories(self):
        """Yield the directories PyRETIS should make."""
        for key in self.directory:
            yield self.directory[key]

    def update_directories(self, path):
        """Update directory names.

        This method will not create new directories, but it will
        update the directory names.

        Parameters
        ----------
        path : string
            The base path to set.

        """
        for key, val in self.directory.items():
            if key == 'path_ensemble':
                self.directory[key] = path
            else:
                self.directory[key] = os.path.join(path, key)
            if val is None:
                logger.debug('Setting directory "%s" to %s', key,
                             self.directory[key])
            else:
                logger.debug('Updating directory "%s": %s -> %s',
                             key, val, self.directory[key])

    def reset_data(self):
        """Erase the stored data in the path ensemble.

        It can be used in combination with flushing the data to a
        file in order to periodically write and empty the amount of data
        stored in memory.

        Notes
        -----
        We do not reset `self.last_path` as this might be used in the
        RETIS function.

        """
        self.paths = []
        for key in self.nstats:
            self.nstats[key] = 0

    def store_path(self, path):
        """Store a new accepted path in the path ensemble.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            The path we are going to store.

        Returns
        -------
        None, but we update `self.last_path`.

        """
        self.last_path = path

    def add_path_data(self, path, status, cycle=0):
        """Append data from the given path to `self.path_data`.

        This will add the data from a given` path` to the list path data
        for this ensemble. If will also update `self.last_path` if the
        given `path` is accepted.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the object to store data from.
        status : string
            This is the status of the path. Note that the path object
            also has a status property. However, this one might not be
            set, for instance when the path is just None. We therefore
            use `status` here as a parameter.
        cycle : int, optional
            The current cycle number.

        """
        if len(self.paths) >= self.maxpath:
            # This is just to limit the data we keep in memory in
            # case of really long simulations.
            logger.debug(('Path-data memory storage reset for ensemble %s.\n'
                          'This is just to limit the amount of data we store '
                          'in memory.\nThis will *NOT* influence the '
                          'simulation'), self.ensemble_name)
            self.paths = []
        # Update statistics:
        if path is None:
            # Here we add a dummy path with minimal info. This is because we
            # could not generate a path for some reason which should be
            # specified by the status.
            path_data = {'status': status, 'generated': ('', 0, 0, 0),
                         'weight': 1.}
        else:
            path_data = path.get_path_data(status, self.interfaces)
            if 'EXP' in status:
                path_data['status'] = 'EXP'
            if path_data['status'] in {'ACC', 'EXP'}:  # Store the path:
                self.store_path(path)
                if path_data['generated'][0] in {'sh', 'ss', 'wt', 'wf'}:
                    self.nstats['nshoot'] += 1
        path_data['cycle'] = cycle  # Also store cycle number.
        self.paths.append(path_data)  # Store the new data.
        # Update some statistics:
        # This is to count also for the first occurrence of the status:
        self.nstats[status] = self.nstats.get(status, 0) + 1
        self.nstats['npath'] += 1

    def get_accepted(self):
        """Yield accepted paths from the path ensemble.

        This function will return an iterator useful for iterating over
        accepted paths only. In the path ensemble we store both accepted
        and rejected paths. This function will loop over all paths
        stored and yield the accepted paths the correct number of times.
        """
        last_path = None
        for path in self.paths:
            if path['status'] == 'ACC':
                last_path = path
            yield last_path

    def get_acceptance_rate(self):
        """Return acceptance rate for the path ensemble.

        The acceptance rate is obtained as the fraction of accepted
        paths to the total number of paths in the path ensemble. This
        will only consider the paths that are currently stored in
        `self.paths`.

        Returns
        -------
        out : float
            The acceptance rate.

        """
        acc = 0
        npath = 0
        for path in self.paths:
            if path['status'] == 'ACC':
                acc += 1
            npath += 1
        return float(acc) / float(npath)

    def get_paths(self):
        """Yield the different paths stored in the path ensemble.

        It is included here in order to have a simple compatibility
        between the :py:class:`.PathEnsemble` object and the
        py:class:`.PathEnsembleFile` object. This is useful for the
        analysis.

        Yields
        ------
        out : dict
            This is the dictionary representing the path data.

        """
        for path in self.paths:
            yield path

    def move_path_to_generate(self, path, prefix=None):
        """Move a path for temporary storing."""
        return

    def __str__(self):
        """Return a string with some info about the path ensemble."""
        msg = [f'Path ensemble: {self.ensemble_name}']
        msg += [f'\tInterfaces: {self.interfaces}']
        if self.nstats['npath'] > 0:
            npath = self.nstats['npath']
            nacc = self.nstats.get('ACC', 0)
            msg += [f'\tNumber of paths stored: {npath}']
            msg += [f'\tNumber of accepted paths: {nacc}']
            ratio = float(nacc) / float(npath)
            msg += [f'\tRatio accepted/total paths: {ratio}']
        return '\n'.join(msg)

    def restart_info(self):
        """Return a dictionary with restart information."""
        restart = {
            'nstats': self.nstats,
            'interfaces': self.interfaces,
            'ensemble_number': self.ensemble_number,
        }
        if hasattr(self, 'rgen'):
            restart['rgen'] = self.rgen.get_state()

        if self.last_path:
            restart['last_path'] = self.last_path.restart_info()

        return restart

    def load_restart_info(self, info, cycle=0):
        """Load restart information.

        Parameters
        ----------
        info : dict
            A dictionary with the restart information.
        cycle : integer, optional
            The current simulation cycle.

        """
        self.nstats = info['nstats']
        # for attr in ('interfaces', 'ensemble_number'):
        #     if info[attr] != getattr(self, attr):
        #         logger.warning(
        #             'Inconsistent path ensemble restart info for %s', attr)
        for key in info:
            if key == 'rgen':
                self.rgen = create_random_generator(info[key])
            elif key == 'last_path':
                rgen = create_random_generator(info[key]['rgen'])
                path = Path(rgen=rgen)
                path.load_restart_info(info['last_path'])
                path_data = path.get_path_data('ACC', self.interfaces)
                path_data['cycle'] = cycle
                self.last_path = path
                self.paths.append(path_data)
            elif hasattr(self, key):
                setattr(self, key, info[key])


class PathEnsembleExt(PathEnsemble):
    """Representation of a path ensemble.

    This class is similar to :py:class:`.PathEnsemble` but it is made
    to work with external paths. That is, some extra file handling is
    done when accepting a path.

    """

    @staticmethod
    def _move_path(path, target_dir, prefix=None):
        """Move a path to a given target directory.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to move.
        target_dir : string
            The location where we are moving the path to.
        prefix : string, optional
            To give a prefix to the name of moved files.

        """
        logger.debug('Moving path to %s', target_dir)
        new_pos, source = _generate_file_names(path, target_dir,
                                               prefix=prefix)
        for pos, phasepoint in zip(new_pos, path.phasepoints):
            phasepoint.particles.set_pos(pos)
        for src, dest in source.items():
            if src == dest:
                logger.debug('Skipping move %s -> %s', src, dest)
            else:
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        logger.debug('Removing %s as it exists', dest)
                        os.remove(dest)
                logger.debug('Moving %s -> %s', src, dest)
                os.rename(src, dest)

    @staticmethod
    def _copy_path(path, target_dir, prefix=None):
        """Copy a path to a given target directory.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to copy.
        target_dir : string
            The location where we are copying the path to.
        prefix : string, optional
            To give a prefix to the name of copied files.

        Returns
        -------
        out : object like py:class:`.PathBase`
            A copy of the input path.

        """
        path_copy = path.copy()
        new_pos, source = _generate_file_names(path_copy, target_dir,
                                               prefix=prefix)
        # Update positions:
        for pos, phasepoint in zip(new_pos, path_copy.phasepoints):
            phasepoint.particles.set_pos(pos)
        for src, dest in source.items():
            if src != dest:
                if os.path.exists(dest):
                    if os.path.isfile(dest):
                        logger.debug('Removing %s as it exists', dest)
                        os.remove(dest)
                logger.debug('Copy %s -> %s', src, dest)
                shutil.copy(src, dest)
        return path_copy

    def store_path(self, path):
        """Store a path by explicitly moving it.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path object we are going to store.

        """
        self._move_path(path, self.directory['accepted'])
        self.last_path = path
        for entry in self.list_superfluous():
            try:
                os.remove(entry)
            except OSError:  # pragma: no cover
                pass

    def list_superfluous(self):
        """List files in accepted directory that we do not need."""
        last = set()
        if self.last_path:
            for phasepoint in self.last_path.phasepoints:
                pos_file, _ = phasepoint.particles.get_pos()
                last.add(os.path.split(pos_file)[-1])
        for entry in os.scandir(self.directory['accepted']):
            if entry.is_file() and os.path.split(entry.path)[-1] not in last:
                yield entry.path

    def move_path_to_generate(self, path, prefix=None):
        """Move a path for temporary storing."""
        self._move_path(path, self.directory['generate'], prefix=prefix)

    def load_restart_info(self, info, cycle=0):
        """Load restart for external path."""
        super().load_restart_info(info, cycle=cycle)
        # Update file names:
        directory = self.directory['accepted']
        self._copy_path(self.last_path, self.directory['accepted'])
        for entry in self.list_superfluous():
            try:
                os.remove(entry)
            except OSError:  # pragma: no cover
                pass

        for phasepoint in self.last_path.phasepoints:
            filename = os.path.basename(phasepoint.particles.get_pos()[0])
            new_file_name = os.path.join(directory, filename)
            if not os.path.isfile(new_file_name):
                logger.critical('The restart path "%s" does not exist',
                                new_file_name)
            phasepoint.particles.set_pos((new_file_name,
                                          phasepoint.particles.get_pos()[1]))


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
    path_ensemble = PathEnsembleExt(i_ens, interfaces, rgen=rgen_path, exe_dir=exe_dir)

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
        'gromacs2': {'cls': GromacsEngine2},
    }
    return generic_factory(settings, engine_map, name='engine')


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

class EngineBase(metaclass=ABCMeta):
    """
    Abstract base class for engines.

    The engines perform molecular dynamics (or Monte Carlo) and they
    are assumed to act on a system. Typically they will integrate
    Newtons equation of motion in time for that system.

    Attributes
    ----------
    description : string
        Short string description of the engine. Used for printing
        information about the integrator.
    exe_dir : string
        A directory where the engine is going to be executed.
    engine_type : string or None
        Describe the type of engine as an "internal" or "external"
        engine. If this is undefined, this variable is set to None.
    needs_order : boolean
        Determines if the engine needs an internal order parameter
        or not. If not, it is assumed that the order parameter is
        calculated by the engine.

    """

    engine_type = None
    needs_order = True

    def __init__(self, description):
        """Just add the description."""
        self.description = description
        self._exe_dir = None

    @property
    def exe_dir(self):
        """Return the directory we are currently using."""
        return self._exe_dir

    @exe_dir.setter
    def exe_dir(self, exe_dir):
        """Set the directory for executing."""
        self._exe_dir = exe_dir
        if exe_dir is not None:
            logger.debug('Setting exe_dir to "%s"', exe_dir)
            if self.engine_type == 'external' and not os.path.isdir(exe_dir):
                logger.warning(('"Exe dir" for "%s" is set to "%s" which does'
                                ' not exist!'), self.description, exe_dir)

    @abstractmethod
    def integration_step(self, ensemble):
        """Perform one time step of the integration."""
        return

    @staticmethod
    def add_to_path(path, phase_point, left, right):
        """
        Add a phase point and perform some checks.

        This method is intended to be used by the propagate methods.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            The path to add to.
        phase_point : object like py:class:`.System`
            The phase point to add to the path.
        left : float
            The left interface.
        right : float
            The right interface.

        """
        status = 'Running propagate...'
        success = False
        stop = False
        add = path.append(phase_point)
        if not add:
            if path.length >= path.maxlen:
                status = 'Max. path length exceeded'
            else:  # pragma: no cover
                status = 'Could not add for unknown reason'
            success = False
            stop = True
        if path.phasepoints[-1].order[0] < left:
            status = 'Crossed left interface!'
            success = True
            stop = True
        elif path.phasepoints[-1].order[0] > right:
            status = 'Crossed right interface!'
            success = True
            stop = True
        if path.length == path.maxlen:
            status = 'Max. path length exceeded!'
            success = False
            stop = True
        return status, success, stop, add

    @abstractmethod
    def propagate(self, path, ensemble, reverse=False):
        """Propagate equations of motion."""
        return

    @abstractmethod
    def modify_velocities(self, ensemble, vel_settings):
        """Modify the velocities of the current state.

        Parameters
        ----------
        ensemble: dict
            It contains all the runners:

            * `path` : object like :py:class:`.PathBase`
              This is the path we use to fill in phase-space points.
              We are here not returning a new path - this since we want
              to delegate the creation of the path (type) to the method
              that is running `propagate`.

        vel_settings: dict
            It contains all the info for the velocity:

            * `sigma_v` : numpy.array, optional
              These values can be used to set a standard deviation (one
              for each particle) for the generated velocities.
            * `aimless` : boolean, optional
              Determines if we should do aimless shooting or not.
            * `momentum` : boolean, optional
              If True, we reset the linear momentum to zero after
              generating.
            * `rescale or rescale_energy` : float, optional
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
        return

    @abstractmethod
    def calculate_order(self, ensemble, xyz=None, vel=None, box=None):
        """Obtain the order parameter."""
        return

    @abstractmethod
    def dump_phasepoint(self, phasepoint, deffnm=None):
        """Dump phase point to a file."""
        return

    @abstractmethod
    def kick_across_middle(self, ensemble, middle,
                           tis_settings):
        """Force a phase point across the middle interface."""
        return

    @abstractmethod
    def clean_up(self):
        """Perform clean up after using the engine."""
        return

    @staticmethod
    def snapshot_to_system(system, snapshot):
        """Convert a snapshot to a system object."""
        system_copy = system.copy()
        system_copy.order = snapshot.get('order', None)
        particles = system_copy.particles
        particles.pos = snapshot.get('pos', None)
        particles.vel = snapshot.get('vel', None)
        particles.vpot = snapshot.get('vpot', None)
        particles.ekin = snapshot.get('ekin', None)
        for external in ('config', 'vel_rev', 'top'):
            if hasattr(particles, external) and external in snapshot:
                setattr(particles, external, snapshot[external])
        return system_copy

    def __eq__(self, other):
        """Check if two engines are equal."""
        if self.__class__ != other.__class__:
            logger.debug('%s and %s.__class__ differ', self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug('%s and %s.__dict__ differ', self, other)
            return False

        for i in ['engine_type', 'needs_order',
                  'description', '_exe_dir', 'timestep']:
            if hasattr(self, i):
                if getattr(self, i) != getattr(other, i):
                    logger.debug('%s for %s and %s, attributes are %s and %s',
                                 i, self, other,
                                 getattr(self, i), getattr(other, i))
                    return False

        if hasattr(self, 'rgen'):
            # pylint: disable=no-member
            if (self.rgen.__class__ != other.rgen.__class__
                    or set(self.rgen.__dict__) != set(other.rgen.__dict__)):
                logger.debug('rgen class differs')
                return False

            # pylint: disable=no-member
            for att1, att2 in zip(self.rgen.__dict__, other.rgen.__dict__):
                # pylint: disable=no-member
                if self.rgen.__dict__[att1] != other.rgen.__dict__[att2]:
                    logger.debug('rgen class attribute %s and %s differs',
                                 att1, att2)
                    return False

        return True

    def __ne__(self, other):
        """Check if two engines are not equal."""
        return not self == other

    @classmethod
    def can_use_order_function(cls, order_function):
        """Fail if the engine can't be used with an empty order parameter."""
        if order_function is None and cls.needs_order:
            raise ValueError(
                'No order parameter was defined, but the '
                'engine *does* require it.'
            )

    def restart_info(self):
        """General method.

        Returns the info to allow an engine exact restart.

        Returns
        -------
        info : dict
            Contains all the updated simulation settings and counters.

        """
        info = {'description': self.description}

        return info

    def load_restart_info(self, info=None):
        """Load restart information.

        Parameters
        ----------
        info : dict
            The dictionary with the restart information, should be
            similar to the dict produced by :py:func:`.restart_info`.

        """
        self.description = info.get('description')

    def __str__(self):
        """Return the string description of the integrator."""
        return self.description

class ExternalMDEngine(EngineBase):
    """
    Base class for interfacing external MD engines.

    This class defines the interface to external programs. The
    interface will define how we interact with the external programs,
    how we write input files for them and read output files from them.
    New engines should inherit from this class and implement the
    following methods:

    * :py:meth:`ExternalMDEngine.step`
        A method for performing a MD step with the external
        engine. Note that the MD step can consist of a number
        of subcycles.
    * :py:meth:`ExternalMDEngine._read_configuration`
        For reading output (configurations) from the external engine.
        This is used for calculating the order parameter(s).
    * :py:meth:`ExternalMDEngine._reverse_velocities`
        For reversing velocities in a snapshot. This method
        will typically make use of the method
        :py:meth:`ExternalMDEngine._read_configuration`.
    * :py:meth:`ExternalMDEngine._extract_frame`
        For extracting a single frame from a trajectory.
    * :py:meth:`ExternalMDEngine._propagate_from`
        The method for propagating the equations of motion using
        the external engine.
    * :py:meth:`ExternalMDEngine.modify_velocities`
        The method used for generating random velocities for
        shooting points. Note that this method is defined in
        :py:meth:`EngineBase.modify_velocities`.

    Attributes
    ----------
    description : string
        A string with a description of the external engine.
        This can, for instance, be what program we are interfacing with.
        This is used for outputting information to the user.
    timestep : float
        The time step used for the external engine.
    subcycles : integer
        The number of steps the external step is composed of. That is:
        each external step is really composed of ``subcycles`` number
        of iterations.

    """

    engine_type = 'external'

    def __init__(self, description, timestep, subcycles):
        """
        Set up the external engine.

        Here we just set up some common properties which are useful
        for the execution.

        Parameters
        ----------
        description : string
            A string with a description of the external engine.
            This can, for instance, be what program we are interfacing
            with. This is used for outputting information to the user.
        timestep : float
            The time step used in the simulation.
        subcycles : integer
            The number of sub-cycles each external integration step is
            composed of.

        """
        super().__init__(description)
        self.timestep = timestep
        self.subcycles = subcycles
        self.ext = 'xyz'
        self.input_files = {}

    def integration_step(self, ensemble):
        """
        Perform a single time step of the integration.

        For external engines, it does not make much sense to run single
        steps unless we absolutely have to. We therefore just fail here.
        I.e. the external engines are not intended for performing pure
        MD simulations.

        If it's absolutely needed, there is a :py:meth:`self.step()`
        method which can be used, for instance in the initialisation.

        Parameters
        ----------
        system : object like :py:class:`.System`
            A system to run the integration step on.

        """
        msg = 'External engine does **NOT** support "integration_step()"!'
        logger.error(msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def step(self, system, name):
        """Perform a single step with the external engine.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.
        name : string
            To name the output files from the external engine.

        Returns
        -------
        out : string
            The name of the output configuration, obtained after
            completing the step.

        """
        return

    @abstractmethod
    def _read_configuration(self, filename):
        """Read output configuration from external software.

        Parameters
        ----------
        filename : string
            The file to open and read a configuration from.

        Returns
        -------
        out[0] : numpy.array
            The dimensions of the simulation box.
        out[1] : numpy.array
            The positions found in the given filename.
        out[2] : numpy.array
            The velocities found in the given filename.

        """
        return

    @abstractmethod
    def _reverse_velocities(self, filename, outfile):
        """Reverse velocities in a given snapshot.

        Parameters
        ----------
        filename : string
            Input file with velocities.
        outfile : string
            File to write with reversed velocities.

        """
        return

    @staticmethod
    def _modify_input(sourcefile, outputfile, settings, delim='='):
        """
        Modify input file for external software.

        Here we assume that the input file has a syntax consisting of
        ``keyword = setting``. We will only replace settings for
        the keywords we find in the file that is also inside the
        ``settings`` dictionary.

        Parameters
        ----------
        sourcefile : string
            The path of the file to use for creating the output.
        outputfile : string
            The path of the file to write.
        settings : dict
            A dictionary with settings to write.
        delim : string, optional
            The delimiter used for separation keywords from settings.

        """
        reg = re.compile(fr'(.*?){delim}')
        written = set()
        with open(sourcefile, 'r', encoding='utf-8') as infile, \
                open(outputfile, 'w', encoding='utf-8') as outfile:
            for line in infile:
                to_write = line
                key = reg.match(line)
                if key:
                    keyword = ''.join([key.group(1), delim])
                    keyword_strip = key.group(1).strip()
                    if keyword_strip in settings:
                        to_write = f'{keyword} {settings[keyword_strip]}\n'
                    written.add(keyword_strip)
                outfile.write(to_write)
            # Add settings not yet written:
            for key, value in settings.items():
                if key not in written:
                    outfile.write(f'{key} {delim} {value}\n')

    @staticmethod
    def _read_input_settings(sourcefile, delim='='):
        """
        Read input settings for simulation input files.

        Here we assume that the input file has a syntax consisting of
        ``keyword = setting``, where ``=`` can be any string given
        in the input parameter ``delim``.

        Parameters
        ----------
        sourcefile : string
            The path of the file to use for creating the output.
        delim : string, optional
            The delimiter used for separation keywords from settings.

        Returns
        -------
        settings : dict of strings
            The settings found in the file.

        Note
        ----
        Important: We are here assuming that there will *ONLY* be one
        keyword per line.

        """
        reg = re.compile(fr'(.*?){delim}')
        settings = {}
        with open(sourcefile, 'r', encoding='utf-8') as infile:
            for line in infile:
                key = reg.match(line)
                if key:
                    keyword_strip = key.group(1).strip()
                    settings[keyword_strip] = line.split(delim)[1].strip()
        return settings

    def execute_command(self, cmd, cwd=None, inputs=None):
        """
        Execute an external command for the engine.

        We are here executing a command and then waiting until it
        finishes. The standard out and standard error are piped to
        files during the execution and can be inspected if the
        command fails. This method returns the return code of the
        issued command.

        Parameters
        ----------
        cmd : list of strings
            The command to execute.
        cwd : string or None, optional
            The current working directory to set for the command.
        inputs : bytes or None, optional
            Additional inputs to give to the command. These are not
            arguments but more akin to keystrokes etc. that the external
            command may take.

        Returns
        -------
        out : int
            The return code of the issued command.

        """
        cmd2 = ' '.join(cmd)
        logger.debug('Executing: %s', cmd2)
        if inputs is not None:
            logger.debug('With input: %s', inputs)

        out_name = 'stdout.txt'
        err_name = 'stderr.txt'

        if cwd:
            out_name = os.path.join(cwd, out_name)
            err_name = os.path.join(cwd, err_name)

        return_code = None

        with open(out_name, 'wb') as fout, open(err_name, 'wb') as ferr:
            cmd = shlex.split(' '.join(cmd))
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd
            )
            exe.communicate(input=inputs)
            # Note: communicate will wait until process terminates.
            return_code = exe.returncode
            if return_code != 0:
                logger.error('Execution of external program (%s) failed!',
                             self.description)
                logger.error('Attempted command: %s', cmd2)
                logger.error('Execution directory: %s', cwd)
                if inputs is not None:
                    logger.error('Input to external program was: %s', inputs)
                logger.error('Return code from external program: %i',
                             return_code)
                logger.error('STDOUT, see file: %s', out_name)
                logger.error('STDERR, see file: %s', err_name)
                msg = (f'Execution of external program ({self.description}) '
                       f'failed with command:\n {cmd2}.\n'
                       f'Return code: {return_code}')
                raise RuntimeError(msg)
        if return_code is not None and return_code == 0:
            self._removefile(out_name)
            self._removefile(err_name)
        return return_code

    @staticmethod
    def _movefile(source, dest):
        """Move file from source to destination."""
        logger.debug('Moving: %s -> %s', source, dest)
        shutil.move(source, dest)

    @staticmethod
    def _copyfile(source, dest):
        """Copy file from source to destination."""
        logger.debug('Copy: %s -> %s', source, dest)
        shutil.copyfile(source, dest)

    @staticmethod
    def _removefile(filename):
        """Remove a given file if it exist."""
        try:
            os.remove(filename)
            logger.debug('Removing: %s', filename)
        except OSError:
            logger.debug('Could not remove: %s', filename)

    def _remove_files(self, dirname, files):
        """Remove files from a directory.

        Parameters
        ----------
        dirname : string
            Where we are removing.
        files : list of strings
            A list with files to remove.

        """
        for i in files:
            self._removefile(os.path.join(dirname, i))

    def clean_up(self):
        """Will remove all files from the current directory."""
        dirname = self.exe_dir
        logger.debug('Running engine clean-up in "%s"', dirname)
        files = [item.name for item in os.scandir(dirname) if item.is_file()]
        self._remove_files(dirname, files)

    def calculate_order(self, ensemble, xyz=None, vel=None, box=None):
        """
        Calculate order parameter from configuration in a file.

        Note, if ``xyz``, ``vel`` or ``box`` are given, we will
        **NOT** read positions, velocity and box information from the
        current configuration file.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system` : object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.

        xyz : numpy.array, optional
            The positions to use, in case we have already read them
            somewhere else. We will then not attempt to read them again.
        vel : numpy.array, optional
            The velocities to use, in case we already have read them.
        box : numpy.array, optional
            The current box vectors, in case we already have read them.

        Returns
        -------
        out : list of floats
            The calculated order parameter(s).

        """
        # Convert system into an internal representation:
        system = ensemble['system']
        order_function = ensemble['order_function']
        if any((xyz is None, vel is None, box is None)):
            out = self._read_configuration(system.particles.config[0])
            box = out[0]
            xyz = out[1]
            vel = out[2]
        system.particles.pos = xyz
        if system.particles.vel_rev:
            if vel is not None:
                system.particles.vel = -1.0 * vel
        else:
            system.particles.vel = vel
        if box is None and self.input_files.get('template', False):
            # CP2K specific box initiation:
            box, _ = read_cp2k_box(ensemble['engine'].input_files['template'])

        system.update_box(box)
        return order_function.calculate(system)

    def kick_across_middle(self, ensemble, middle, tis_settings):
        """
        Force a phase point across the middle interface.

        This is accomplished by repeatedly kicking the phase point so
        that it crosses the middle interface.

        Parameters
        ----------
        ensemble: dict
            It contains:

            * `system`: object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `rgen`: object like :py:class:`.RandomGenerator`
              This is the random generator that will be used.

        middle : float
            This is the value for the middle interface.
        tis_settings : dict
            This dictionary contains settings for TIS. Explicitly used here:

            * `zero_momentum`: boolean, determines if the momentum is zeroed.
            * `rescale_energy`: boolean, determines if energy is re-scaled.

        Returns
        -------
        out[0] : object like :py:class:`.System`
            The phase-point just before the crossing the interface.
        out[1] : object like :py:class:`.System`
            The phase-point just after crossing the interface.

        Note
        ----
        This function will update the input system state.

        """
        system = ensemble['system'].copy()
        logger.info('Kicking with external integrator: %s', self.description)
        # We search for crossing with the middle interface and do this
        # by sequentially kicking the initial phase point
        # Let's get the starting point:
        initial_file = self.dump_frame(system)
        # Create a "previous file" for storing the state before a new kick:
        prev_file = os.path.join(
            self.exe_dir, f'p_{os.path.basename(initial_file)}'
        )
        msg_file_name = os.path.join(self.exe_dir, 'msg-kick.txt')
        msg_file = FileIO(msg_file_name, 'w', None, backup=False)
        msg_file.open()
        msg_file.write(f'Kick initiation for {self.description}')
        self._copyfile(initial_file, prev_file)
        # Update so that we use the prev_file:
        ensemble['system'].particles.set_pos((prev_file, None, None))
        logger.info('Searching for crossing with: %9.6g', middle)
        print_to_screen(f'Searching for crossing with: {middle}')
        print_to_screen(f'Writing progress to: {msg_file_name}')

        while True:
            msg_file.write('New kick:')
            # Do kick from current state:
            msg_file.write('\tModify velocities...')
            vel_settings = {'sigma_v': None, 'aimless': True,
                            'momentum': False, 'rescale': None}
            self.modify_velocities(ensemble, vel_settings)
            # Update order parameter in case it's velocity dependent:
            curr = self.calculate_order(ensemble)[0]
            msg_file.write(f'\tAfter kick: {curr}')
            previous = ensemble['system'].copy()
            previous.order = [curr]
            # Update system by integrating forward:
            msg_file.write('\tIntegrate forward...')
            conf = self.step(ensemble['system'], 'kicked')
            curr_file = os.path.join(self.exe_dir, conf)
            # Compare previous order parameter and the new one:
            prev = curr
            curr = self.calculate_order(ensemble)[0]
            txt = f'{prev} -> {curr} | {middle}'
            msg_file.write(f'\t{txt}')
            if (prev <= middle < curr) or (curr < middle <= prev):
                logger.info('Crossed middle interface: %s', txt)
                msg_file.write('\tCrossed middle interface!')
                # Middle interface was crossed, just stop the loop.
                self._copyfile(curr_file, prev_file)
                # Update file name after moving:
                system.particles.set_pos((prev_file, None, None))
                break
            if (prev <= curr < middle) or (middle < curr <= prev):
                # Getting closer, keep the new point:
                logger.debug('Getting closer to middle: %s', txt)
                msg_file.write(
                    '\tGetting closer to middle, keeping new point.'
                )
                self._movefile(curr_file, prev_file)
                # Update file name after moving:
                ensemble['system'].particles.set_pos((prev_file, None, None))
            else:  # We did not get closer, fall back to previous point:
                logger.debug('Did not get closer to middle: %s', txt)
                msg_file.write(
                    '\tDid not get closer, fall back to previous point.'
                )
                ensemble['system'] = previous.copy()
                curr = previous.order
                self._removefile(curr_file)
            msg_file.flush()
        msg_file.close()
        self._removefile(msg_file_name)
        return previous, system

    @abstractmethod
    def _extract_frame(self, traj_file, idx, out_file):
        """Extract a frame from a trajectory file.

        Parameters
        ----------
        traj_file : string
            The trajectory file to open.
        idx : integer
            The frame number we look for.
        out_file : string
            The file to dump to.

        """
        return

    def propagate(self, path, ensemble, reverse=False):
        """
        Propagate the equations of motion with the external code.

        This method will explicitly do the common set-up, before
        calling more specialised code for doing the actual propagation.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
            We are here not returning a new path - this since we want
            to delegate the creation of the path to the method
            that is running `propagate`.
        ensemble: dict
            It contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `interfaces` : list of floats
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
        logger.debug('Running propagate with: "%s"', self.description)

        prefix = str(counter())
        if ensemble.get('path_ensemble', False):
            prefix = ensemble['path_ensemble'].ensemble_name_simple + '_' \
                     + prefix

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

        initial_state = ensemble['system'].copy()
        system = ensemble['system']
        initial_file = self.dump_frame(system, deffnm=prefix + '_conf')
        msg_file.write(f'# Initial file: {initial_file}')
        logger.debug('Initial state: %s', system)

        if reverse != system.particles.vel_rev:
            logger.debug('Reversing velocities in initial config.')
            msg_file.write('# Reversing velocities')
            basepath = os.path.dirname(initial_file)
            localfile = os.path.basename(initial_file)
            initial_conf = os.path.join(basepath, f'r_{localfile}')
            self._reverse_velocities(initial_file, initial_conf)
        else:
            initial_conf = initial_file
        msg_file.write(f'# Initial config: {initial_conf}')

        # Update system to point to the configuration file:
        system.particles.set_pos((initial_conf, None))
        system.particles.set_vel(reverse)
        # Propagate from this point:
        msg_file.write(f'# Interfaces: {ensemble["interfaces"]}')
        success, status = self._propagate_from(
            name,
            path,
            ensemble,
            msg_file,
            reverse=reverse
        )
        # Reset to initial state:
        ensemble['system'] = initial_state
        msg_file.close()
        return success, status

    @abstractmethod
    def _propagate_from(self, name, path, ensemble, msg_file, reverse=False):
        """
        Run the actual propagation using the specific engine.

        This method is called after :py:meth:`.propagate`. And we
        assume that the necessary preparations before the actual
        propagation (e.g. dumping of the configuration etc.) is
        handled in that method.

        Parameters
        ----------
        name : string
            A name to use for the trajectory we are generating.
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
        ensemble: dict
            It contains:

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
        return

    def _name_output(self, basename):
        """
        Create a file name for the output file.

        This method is used when we dump a configuration to add
        the correct extension for GROMACS (either gro or g96).

        Parameters
        ----------
        basename : string
            The base name to give to the file.

        Returns
        -------
        out : string
            A file name with an extension.

        """
        out_file = f'{basename}.{self.ext}'
        return os.path.join(self.exe_dir, out_file)

    def dump_config(self, config, deffnm='conf'):
        """Extract configuration frame from a system if needed.

        Parameters
        ----------
        config : tuple
            The configuration given as (filename, index).
        deffnm : string, optional
            The base name for the file we dump to.

        Returns
        -------
        out : string
            The file name we dumped to. If we did not in fact dump, this is
            because the system contains a single frame and we can use it
            directly. Then we return simply this file name.

        Note
        ----
        If the velocities should be reversed, this is handled elsewhere.

        """
        pos_file, idx = config
        out_file = os.path.join(self.exe_dir, self._name_output(deffnm))
        if idx is None:
            if pos_file != out_file:
                self._copyfile(pos_file, out_file)
        else:
            logger.debug('Config: %s', (config, ))
            self._extract_frame(pos_file, idx, out_file)
        return out_file

    def dump_frame(self, system, deffnm='conf'):
        """Just dump the frame from a system object."""
        return self.dump_config(system.particles.config, deffnm=deffnm)

    def dump_phasepoint(self, phasepoint, deffnm='conf'):
        """Just dump the frame from a system object."""
        pos_file = self.dump_config(phasepoint.particles.get_pos(),
                                    deffnm=deffnm)
        phasepoint.particles.set_pos((pos_file, None))


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
    sim_type = settings['simulation']['task'].lower()

    sim_map = {
        'retis': create_retis_simulation,
    }

    # Improve setting quality
    add_default_settings(settings)
    add_specific_default_settings(settings)

    # if settings['simulation'].get('restart', False):
    #     settings, info_restart = settings_from_restart(settings)

    if sim_type not in sim_map:  # TODO put in check_sim_type
        msgtxt = 'Unknown simulation task {}'.format(sim_type)
        logger.error(msgtxt)
        raise ValueError(msgtxt)

    simulation = sim_map[sim_type](settings)
    msgtxt = '{}'.format(simulation)
    logger.info('Created simulation:\n%s', msgtxt)

    # if settings['simulation'].get('restart', False):
    #     simulation.load_restart_info(info_restart)

    return simulation

def create_retis_simulation(settings):
    """Set up and create a RETIS simulation.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    SimulationRETIS : object like :py:class:`.SimulationRETIS`
        The object representing the simulation to run.

    """
    check_ensemble(settings)
    ensembles = create_ensembles(settings)

    key_check('steps', settings)

    controls = {'rgen': create_random_generator(settings['simulation']),
                'steps': settings['simulation']['steps'],
                'startcycle': settings['simulation'].get('startcycle', 0)}

    return SimulationRETIS(ensembles, settings, controls)


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

    def step(self):
        """Perform a TIS/RETIS simulation step.

        Returns
        -------
        out : dict
            This list contains the results of the defined tasks.

        """
        sim_type = self.settings['simulation']['task'].lower()
        sim_map = {'tis': make_tis,
                   'explore': make_tis,
                   'retis': make_retis_step}

        prio_skip = priority_checker(self.ensembles, self.settings)
        if True not in prio_skip:
            self.cycle['step'] += 1
            self.cycle['stepno'] += 1
        msgtxt = f' {sim_type} step. Cycle {self.cycle["stepno"]}'
        logger.info(msgtxt)
        prepare = sim_map[sim_type]
        runner = prepare(self.ensembles, self.rgen,
                         self.settings, self.cycle['step'])
        results = {}
        for i_ens, res in enumerate(runner):
            if prio_skip[i_ens]:
                continue
            idx = res['ensemble_number']
            result = {'cycle': self.cycle}
            result[f'move-{idx}'] = res['mc-move']
            result[f'status-{idx}'] = res['status']
            result[f'path-{idx}'] = res['trial']
            result[f'accept-{idx}'] = res['accept']
            result[f'all-{idx}'] = res
            # This is to fix swaps needs
            idx_ens = i_ens if sim_type in {'tis', 'explore'} else idx
            result[f'pathensemble-{idx}'] = \
                self.ensembles[idx_ens]['path_ensemble']
            for task in self.output_tasks:
                task.output(result)
            results.update(result)
            if soft_partial_exit(self.settings['simulation']['exe_path']):
                self.cycle['endcycle'] = self.cycle['step']
                break
        return results

    def run(self):
        """Run a path simulation.

        The intended usage is for simulations where all tasks have
        been defined in :py:attr:`self.tasks`.

        Note
        ----
        This function will just run the tasks via executing
        :py:meth:`.step` In general, this is probably too generic for
        the simulation you want, if you are creating a custom simulation.
        Please consider customizing the :py:meth:`.run` (or the
        :py:meth:`.step`) method of your simulation class.

        Yields
        ------
        out : dict
            This dictionary contains the results from the simulation.

        """
        while not self.is_finished():
            result = self.step()
            self.write_restart()
            if self.soft_exit():
                yield result
                break
            yield result


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

        if not is_sorted(savelambda):
            msg = "Interface positions in the ensemble simulation "\
                "are NOT properly sorted (ascending order)"

    else:
        msg = "No ensemble in settings"

    if 'msg' in locals():
        raise ValueError(msg)

    return True

def is_sorted(lll):
    """Check if a list is sorted."""
    return all(aaa <= bbb for aaa, bbb in zip(lll[:-1], lll[1:]))

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
    # Engine is not None and not internal => external.
    particles = ParticlesExt(dim=3)
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

def create_initial_positions(settings):
    """Set up the initial positions from the given settings.

    The settings can specify the initial positions as a file or
    to be generated on a lattice by PyRETIS.

    Parameters
    ----------
    settings : dict
        Settings for creating the initial positions.

    Returns
    -------
    out[0] : object like :py:class:`.Particles`
        The particles we created.
    out[1] : list
        The size associated with the particles. Can be used to create a
        box.
    out[2] : boolean
        True if we have read/created velocities different from just
        zeros. This is only True if we have read from a file with
        velocities.

    """
    logger.debug('Settings used for initial positions: %s',
                 settings['particles']['position'])
    particles = None
    if 'generate' in settings['particles']['position']:
        particles, box = initial_positions_lattice(settings)
        return particles, box, False
    if 'input_file' in settings['particles']['position']:
        # First check if we need to add a path to the file:
        filename = settings['particles']['position']['input_file']
        if (not os.path.isfile(filename) and
                'exe_path' in settings['simulation']):
            filename = os.path.join(settings['simulation']['exe_path'],
                                    filename)
            settings['particles']['position']['input_file'] = filename
        particles, box, vel = initial_positions_file(settings)
        return particles, box, vel
    msg = 'Unknown settings for initial positions: {}'
    msgtxt = msg.format(settings['particles']['position'])
    logger.error(msgtxt)
    raise ValueError(msgtxt)

def initial_positions_file(settings):
    """Get initial positions from an input file.

    Parameters
    ----------
    settings : dict
        The input settings for the simulation.

    Returns
    -------
    particles : object like :py:class:`.Particles`
        The particles we created.
    size : list of floats
        A size for the region we created. This can be used to create
        a box.
    vel_read : boolean
        True if we read velocities from the input file.

    """
    ndim = settings['system'].get('dimensions', 3)
    pos_settings = settings['particles']['position']
    ptype = settings['particles'].get('ptype', None)
    pname = settings['particles'].get('name', None)
    pmass = settings['particles'].get('mass', {})
    ptypes = {}  # To automatically set particle types based on name.
    snapshot, convert = _get_snapshot_from_file(pos_settings,
                                                settings['system']['units'])
    vel_read = False
    particles = Particles(dim=ndim)
    for i, atomname in enumerate(snapshot['atomname']):
        pos = []
        vel = []
        for key in ['x', 'y', 'z'][:ndim]:
            pos.append(snapshot[key][i])
            vel_key = 'v{}'.format(key)
            if vel_key in snapshot:
                vel.append(snapshot[vel_key][i])
        pos = np.array(pos) * convert['length']
        if len(vel) != ndim:
            vel = np.zeros_like(pos)
        else:
            vel = np.array(vel) * convert['velocity']
            vel_read = True
        # Get particle type from the atom names or from input list:
        if ptype is None:
            if atomname not in ptypes:
                ptypes[atomname] = len(ptypes)
            particle_type = ptypes[atomname]
        else:
            particle_type = list_get(ptype, i)
        if pname is None:
            particle_name = atomname
        else:
            particle_name = list_get(pname, i)
        # Infer the mass from the input masses, or try to get it
        # from the periodic table:
        try:
            particle_mass = pmass[particle_name]
        except KeyError:
            particle_mass = guess_particle_mass(i + 1, particle_name,
                                                settings['system']['units'])
        particles.add_particle(pos,
                               vel, np.zeros_like(pos),
                               mass=particle_mass, name=particle_name,
                               ptype=particle_type)
    try:
        box = {'cell': [i * convert['length'] for i in snapshot['box']]}
        if ndim < 3:
            box['cell'] = box['cell'][:ndim]
    except (KeyError, IndexError, TypeError) as err:
        logger.debug('No box read from file: %s.', err)
        box = None
    logger.info('Read %d particle(s) from "%s".', particles.npart,
                pos_settings['input_file'])
    if vel_read:
        logger.info('Read velocities from file: "%s".',
                    pos_settings['input_file'])
    return particles, box, vel_read

def _get_snapshot_from_file(pos_settings, units):
    """Get a configuration snapshot from a file.

    This snapshot will be used to set up the initial configuration.

    Parameters
    ----------
    pos_settings : dict
        A dict with information on what we should read.
    units : string
        The internal units.

    Returns
    -------
    snapshot : dict
        The snapshot we found in the file. It will at least have the
        keys with the positions ('x', 'y', 'z') and atom name
        'atomnames'. It may have information about velocities
        ('vx', 'vy', 'vz') and the box ('box').
    convert : dict
        Dictionary with conversion factors to internal units.

    """
    filename = pos_settings.get('input_file', None)
    if filename is None:
        msg = ('Requested reading (initial) configuration from file, '
               'but no input file given!')
        logger.error(msg)
        raise ValueError(msg)
    fmt = pos_settings.get('format', os.path.splitext(filename)[1][1:])
    snaps = []
    convert = None
    if fmt not in READFILE:
        msg = ('Input configuration "{}" has unknown '
               'format "{}".').format(filename, fmt)
        logger.error(msg)
        logger.error('Supported formats are: %s.', [key for key in READFILE])
        raise ValueError(msg)

    reader = READFILE[fmt]['reader']
    read_units = READFILE[fmt]['units']
    if read_units is None:
        convert = {'length': 1.0, 'velocity': 1.0}
    else:
        convert = {
            'length': CONVERT['length'][read_units['length'], units],
            'velocity': CONVERT['velocity'][read_units['velocity'], units]
        }
    logger.info(
        'Reading initial configuration from "%s" (format: "%s").',
        filename,
        fmt,
    )
    snaps = [snap for snap in reader(filename)]

    snapshot = None
    if len(snaps) == 1:
        snapshot = snaps[0]
    elif len(snaps) > 1:
        msg = ('Found several frames ({}) in input file.'
               ' Will use the last one!').format(len(snaps))
        logger.warning(msg)
        snapshot = snaps[-1]
    else:
        msg = ('Could not find any configurations in '
               'input file: {}').format(filename)
        logger.error(msg)
        raise ValueError(msg)
    return snapshot, convert


class Particles:
    """Base class for a collection of particles.

    This is a simple particle list. It stores the positions,
    velocities, forces, masses (and inverse masses) and type
    information for a set of particles. This class also defines a
    method for iterating over pairs of particles, which could be
    useful for implementing neighbour lists. In this particular
    class, this method will just define an all-pairs list.

    Attributes
    ----------
    npart : integer
        Number of particles.
    pos : numpy.array
        Positions of the particles.
    vel : numpy.array
        Velocities of the particles.
    force : numpy.array
        Forces on the particles.
    virial : numpy.array
        The current virial for the particles.
    mass : numpy.array
        Masses of the particles.
    imass : numpy.array
        Inverse masses, `1.0 / self.mass`.
    name : list of strings
        A name for the particle. This may be used as short text
        describing the particle.
    ptype : numpy.array of integers
        A type for the particle. Particles with identical `ptype` are
        of the same kind.
    dim : int
        This variable is the dimensionality of the particle list. This
        should be derived from the box. For some functions, it is
        convenient to be able to access the dimensionality directly
        from the particle list. It is therefore set as an attribute
        here.
    vpot : float
        The potential energy of the particles.
    ekin : float
        The kinetic energy of the particles.

    """

    particle_type = 'internal'

    # Attributes to store when restarting/copying:
    _copy_attr = {'npart', 'name', 'ptype', 'dim'}
    # Attributes which are numpy arrays:
    _numpy_attr = {'pos', 'vel', 'force', 'virial', 'mass', 'imass',
                   'ptype', 'ekin', 'vpot'}

    def __init__(self, dim=1):
        """Initialise the Particle list.

        Here we just create an empty particle list.

        Parameters
        ----------
        dim : integer, optional
            The number of dimensions we are considering for positions,
            velocities and forces.

        """
        self.npart = 0
        self.pos = None
        self.vel = None
        self.vpot = None
        self.ekin = None
        self.force = None
        self.mass = None
        self.imass = None
        self.name = []
        self.ptype = None
        self.virial = None
        self.dim = dim

    def empty_list(self):
        """Reset the particle list.

        This will delete all particles in the list and set other
        variables to `None`.

        Note
        ----
        This is almost `self.__init__` repeated. The reason for this is
        simply that we want to define all attributes in `self.__init__`
        and not get any 'surprise attributes' defined elsewhere.
        Also, note that the dimensionality (`self.dim`) is not changed
        in this method.

        """
        self.npart = 0
        self.pos = None
        self.vpot = None
        self.ekin = None
        self.vel = None
        self.force = None
        self.mass = None
        self.imass = None
        self.name = []
        self.ptype = None
        self.virial = None

    def _copy_attribute(self, attr, copy_function):
        """Copy an attribute.

        Parameters
        ----------
        attr : string
            The attribute to copy.
        copy_function : callable
            The method to use for copying the attribute.

        Returns
        -------
        out : object
            A copy of the selected attribute.

        """
        val = getattr(self, attr, None)
        if val is None:
            return None
        return copy_function(val)

    def copy(self):
        """Return a copy of the particle state.

        Returns
        -------
        out : object like :py:class:`.Particles`
            A copy of the current state of the particles.

        """
        particles_copy = self.__class__(dim=self.dim)
        for attr in self._copy_attr:
            copy_attr = self._copy_attribute(attr, copy0)
            setattr(particles_copy, attr, copy_attr)
        for attr in self._numpy_attr:
            copy_attr = self._copy_attribute(attr, np.copy)
            setattr(particles_copy, attr, copy_attr)
        return particles_copy

    def __eq__(self, other):
        """Compare two particle states."""
        attrs = self._copy_attr.union(self._numpy_attr)
        return compare_objects(self, other, attrs,
                               numpy_attrs=self._numpy_attr)

    def set_pos(self, pos):
        """Set positions for the particles.

        Parameters
        ----------
        pos : numpy.array
            The positions to set.

        """
        self.pos = np.copy(pos)

    def get_pos(self):
        """Return (a copy of) positions."""
        return np.copy(self.pos)

    def set_vel(self, vel):
        """Set velocities for the particles.

        Parameters
        ----------
        vel : numpy.array
            The velocities to set.

        """
        self.vel = np.copy(vel)

    def get_vel(self):
        """Return (a copy of) the velocities."""
        return np.copy(self.vel)

    def set_force(self, force):
        """Set the forces for the particles.

        Parameters
        ----------
        force : numpy.array
            The forces to set.

        """
        self.force = np.copy(force)

    def get_force(self):
        """Return (a copy of) the forces."""
        return np.copy(self.force)

    def add_particle(self, pos, vel, force, mass=1.0,
                     name='?', ptype=0):
        """Add a particle to the system.

        Parameters
        ----------
        pos : numpy.array
            Positions of new particle.
        vel :  numpy.array
            Velocities of new particle.
        force : numpy.array
            Forces on the new particle.
        mass : float, optional
            The mass of the particle.
        name : string, optional
            The name of the particle.
        ptype : integer, optional
            The particle type.

        """
        if self.npart == 0:
            self.name = [name]
            self.ptype = np.array(ptype, dtype=np.int16)
            self.pos = np.zeros((1, self.dim))
            self.pos[0] = pos
            self.vel = np.zeros((1, self.dim))
            self.vel[0] = vel
            self.force = np.zeros((1, self.dim))
            self.force[0] = force
            self.mass = np.zeros((1, 1))  # Column matrix.
            self.mass[0] = mass
            self.imass = 1.0 / self.mass
        else:
            self.name.append(name)
            self.ptype = np.append(self.ptype, ptype)
            self.pos = np.vstack([self.pos, pos])
            self.vel = np.vstack([self.vel, vel])
            self.force = np.vstack([self.force, force])
            self.mass = np.vstack([self.mass, mass])
            self.imass = np.vstack([self.imass, 1.0/mass])
        self.npart += 1

    def get_selection(self, properties, selection=None):
        """Return selected properties for a selection of particles.

        Parameters
        ----------
        properties : list of strings
            The strings represent the properties to return.
        selection : list with indices to return, optional
            If a selection is not given, data for all particles
            are returned.

        Returns
        -------
        out : list
            A list with the properties in the order they were asked for
            in the ``properties`` argument.

        """
        sel_prop = []
        for prop in properties:
            if hasattr(self, prop):
                var = getattr(self, prop)
                if isinstance(var, list):
                    if selection is None:
                        sel_prop.append(var)
                    else:
                        sel_prop.append([var[i] for i in selection])
                else:
                    if selection is None:
                        sel_prop.append(var)
                    else:
                        sel_prop.append(var[selection])
        return sel_prop

    def __iter__(self):
        """Iterate over the particles.

        This function will yield the properties of the different
        particles.

        Yields
        ------
        out : dict
            The information in `self.pos`, `self.vel`, ... etc.

        """
        for i, pos in enumerate(self.pos):
            part = {'pos': pos, 'vel': self.vel[i], 'force': self.force[i],
                    'mass': self.mass[i], 'imass': self.imass[i],
                    'name': self.name[i], 'ptype': self.ptype[i]}
            yield part

    def pairs(self):
        """Iterate over all pairs of particles.

        Yields
        ------
        out[0] : integer
            The index for the first particle in the pair.
        out[1] : integer
            The index for the second particle in the pair.
        out[2] : integer
            The particle type of the first particle.
        out[3] : integer
            The particle type of the second particle.

        """
        for i, itype in enumerate(self.ptype[:-1]):
            for j, jtype in enumerate(self.ptype[i+1:]):
                yield (i, i+1+j, itype, jtype)

    def __str__(self):
        """Print out basic info about the particle list."""
        return 'Particles: {}\nTypes: {}\nNames: {}'.format(
            self.npart, np.unique(self.ptype), set(self.name)
        )

    def restart_info(self):
        """Generate information for saving a restart file."""
        info = {'type': self.particle_type}
        for copy_list in (self._copy_attr, self._numpy_attr):
            for attr in copy_list:
                try:
                    info[attr] = getattr(self, attr)
                except AttributeError:
                    logger.warning(('Missing attribute "%s" when creating'
                                    ' restart information.'), attr)
                    info[attr] = None
        return info

    def load_restart_info(self, info):
        """Load restart information.

        Parameters
        ----------
        info : dict
            Dictionary with the settings to load.

        """
        for attr in self._copy_attr.union(self._numpy_attr):
            if attr in info:
                setattr(self, attr, info[attr])
            else:
                msg = ('Could not set "{}" for particles '
                       'from restart info').format(attr)
                logger.error(msg)
                raise ValueError(msg)

    def reverse_velocities(self):
        """Reverse the velocities in the system."""
        self.vel = self.vel * -1


class ParticlesExt(Particles):
    """A particle list, when positions and velocities are stored in files.

    Attributes
    ----------
    config : tuple
        The file name and index in this file for the configuration
        the particle list is representing.
    vel_rev : boolean
        If this is True, the velocities in the file represeting
        the configuration will have to be reversed before they are
        used.

    """

    particle_type = 'external'

    # Attributes to store when restarting/copying:
    _copy_attr = {'npart', 'name', 'ptype', 'dim',
                  'config', 'vel_rev'}
    # Attributes which are numpy arrays:
    _numpy_attr = {'pos', 'vel', 'force', 'virial', 'mass', 'imass',
                   'ptype', 'ekin', 'vpot'}

    def __init__(self, dim=1):
        """Create an empty ParticleExt list.

        Parameters
        ----------
        dim : integer, optional
            The number of dimensions we are considering for positions,
            velocities and forces.

        """
        super().__init__(dim=dim)
        self.config = (None, None)
        self.vel_rev = False

    def add_particle(self, pos, vel, force, mass=1.0,
                     name='?', ptype=0):
        """Add a particle to the system.

        Parameters
        ----------
        pos : tuple
            Positions of new particle.
        vel : boolean
            Velocities of new particle.
        force : tuple
            Forces on the new particle.
        mass : float, optional
            The mass of the particle.
        name : string, optional
            The name of the particle.
        ptype : integer, optional
            The particle type.

        """
        self.name = [name]
        self.ptype = np.array(ptype, dtype=np.int16)
        self.pos = None
        self.set_pos(pos)
        self.vel = None
        self.set_vel(vel)
        self.force = force
        self.mass = np.zeros((1, 1))  # Column matrix.
        self.mass[0] = mass
        self.imass = 1.0 / self.mass
        self.npart = 1

    def empty_list(self):
        """Just empty the list."""
        super().empty_list()
        self.config = (None, None)
        self.vel_rev = False

    def reverse_velocities(self):
        """Reverse the velocities in the system."""
        self.vel_rev = not self.vel_rev

    def set_pos(self, pos):
        """Set positions for the particles.

        This will copy the input positions, for this class, the
        input positions are assumed to be a file name with a
        corresponding integer which determines the index for the
        positions in the file for cases where the file contains
        several snapshots.

        Parameters
        ----------
        pos : tuple of (string, int)
            The positions to set, this represents the file name and the
            index for the frame in this file.

        """
        self.config = (pos[0], pos[1])

    def get_pos(self):
        """Just return the positions of the particles."""
        return self.config

    def set_vel(self, rev_vel):
        """Set velocities for the particles.

        Here we store information which tells if the
        velocities should be reversed or not.

        Parameters
        ----------
        rev_vel : boolean
            The velocities to set. If True, the velocities should
            be reversed before used.

        """
        self.vel_rev = rev_vel

    def get_vel(self):
        """Return info about the velocities."""
        return self.vel_rev

    def __str__(self):
        """Print out basic info about the particle list."""
        return 'Config: {}\nReverse velocities: {}'.format(
            self.config, self.vel_rev
        )

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


def key_check(key, settings):
    """Check for the presence of a key in settings."""
    # todo These checks shall be done earlier, when cleaning the input.
    if key not in settings['simulation']:
        msgtxt = 'Simulation setting "{}" is missing!'.format(key)
        logger.critical(msgtxt)
        raise ValueError(msgtxt)

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



class MDEngine(EngineBase):
    """Base class for internal MD integrators.

    This class defines an internal MD integrator. This class of
    integrators work with the positions and velocities of the system
    object directly. Further, we make use of the system object in
    order to update forces etc.

    Attributes
    ----------
    timestep : float
        Time step for the integration.
    description : string
        Description of the MD integrator.
    dynamics : str
        A short string to represent the type of dynamics produced
        by the integrator (NVE, NVT, stochastic, ...).

    """

    engine_type = 'internal'

    def __init__(self, timestep, description, dynamics=None):
        """Set up the integrator.

        Parameters
        ----------
        timestep : float
            The time step for the integrator in internal units.
        description : string
            A short description of the integrator.
        dynamics : string or None, optional
            Description of the kind of dynamics the integrator does.

        """
        super().__init__(description)
        self.timestep = timestep
        self.dynamics = dynamics

    def integration_step(self, system):
        """Perform a single time step of the integration.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are acting on.

        Returns
        -------
        out : None
            Does not return anything, in derived classes it will
            typically update the given `System`.

        """
        raise NotImplementedError

    def select_thermo_function(self, thermo='full'):
        """Select function for calculating thermodynamic properties.

        Parameters
        ----------
        thermo : string, or None, optional
            String which selects the kind of thermodynamic output.

        Returns
        -------
        thermo_func : callable or None
            The function matching the requested thermodynamic output.

        """
        thermo_func = None
        if thermo is not None:
            if thermo not in ('full', 'path'):
                logger.debug(
                    'Unknown thermo "%s" requested: Using "path"!',
                    thermo,
                )
                thermo = 'path'
            logger.debug('Thermo output for %s.integrate(): "%s"',
                         self.__class__.__name__, thermo)
            if thermo == 'full':
                thermo_func = calculate_thermo
            elif thermo == 'path':
                thermo_func = calculate_thermo_path
        return thermo_func

    def integrate(self, ensemble, steps, thermo='full'):
        """Perform several integration steps.

        This method will perform several integration steps, but it will
        also calculate order parameter(s) if requested and energy
        terms.

        Parameters
        ----------
        ensemble: dict
            It contains:

            * `system` : object like :py:class:`.System`
              The system we are integrating.
            * `order_function` : object like :py:class:`.OrderParameter`
              An order function can be specified if we want to
              calculate the order parameter along with the simulation.

        steps : integer
            The number of steps we are going to perform. Note that we
            do not integrate on the first step (e.g. step 0) but we do
            obtain the other properties. This is to output the starting
            configuration.
        thermo : string, optional
            Select the thermodynamic properties we are to calculate.

        Yields
        ------
        results : dict
            The result of a MD step. This contains the state of the
            system and also the order parameter(s) (if calculated) and
            the thermodynamic quantities (if calculated).

        """
        thermo_func = self.select_thermo_function(thermo=thermo)
        system = ensemble['system']
        order_function = ensemble.get('order_function')
        system.potential_and_force()
        for i in range(steps):
            if i == 0:
                pass
            else:
                self.integration_step(system)
            results = {'system': system}
            if order_function is not None:
                results['order'] = self.calculate_order(ensemble)
            if thermo_func:
                results['thermo'] = thermo_func(system)
            yield results

    def invert_dt(self):
        """Invert the time step for the integration.

        Returns
        -------
        out : boolean
            True if the time step is positive, False otherwise.

        """
        self.timestep *= -1.0
        return self.timestep > 0.0

    def propagate(self, path, ensemble, reverse=False):
        """Generate a path by integrating until a criterion is met.

        This function will generate a path by calling the function
        specifying the integration step repeatedly. The integration is
        carried out until the order parameter has passed the specified
        interfaces or if we have integrated for more than a specified
        maximum number of steps. The given system defines the initial
        state and the system is reset to its initial state when this
        method is done.

        Parameters
        ----------
        path : object like :py:class:`.PathBase`
            This is the path we use to fill in phase-space points.
            We are here not returning a new path, this since we want
            to delegate the creation of the path (type) to the method
            that is running `propagate`.
        ensemble: dict
            It contains:

            * `system` : object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function` : object like :py:class:`.OrderParameter`
              An order function can be specified if we want to
              calculate the order parameter along with the simulation.
            * `interfaces` : list of floats
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
        status = 'Propagate w/internal engine'
        logger.debug(status)
        success = False
        # Copy the system, so that we can propagate without altering it:
        system = ensemble['system']
        system.potential_and_force()  # Make sure forces are set.
        left, _, right = ensemble['interfaces']
        for i in range(path.maxlen):
            system.order = self.calculate_order(ensemble)
            ekin = calculate_kinetic_energy(system.particles)[0]
            system.particles.ekin = ekin
            status, success, stop, _ = self.add_to_path(path, system.copy(),
                                                        left, right)
            if stop:
                logger.debug('Stopping propagate at step: %i ', i)
                break
            if reverse:
                system.particles.vel *= -1.0
                self.integration_step(system)
                system.particles.vel *= -1.0
            else:
                self.integration_step(system)
        logger.debug('Propagate done: "%s" (success: %s)', status, success)
        return success, status

    @staticmethod
    def modify_velocities(ensemble, vel_settings):
        """Modify the velocities of the current state.

        This method will modify the velocities of a time slice.
        And it is part of the integrator since it, conceptually,
        fits here:  we are acting on the system and modifying it.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system : object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating
            * `rgen` : object like :py:class:`.RandomGenerator`
              This is the random generator that will be used.

        vel_settings: dict.
            It contains all the info for the velocity:

            * `sigma_v` : numpy.array, optional
              These values can be used to set a standard deviation (one
              for each particle) for the generated velocities.
            * `aimless` : boolean, optional
              Determines if we should do aimless shooting or not.
            * `momentum` : boolean, optional
              If True, we reset the linear momentum to zero after
              generating.
            * `rescale or rescale_energy` : float, optional
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
        rgen = ensemble['rgen']
        system = ensemble['system']
        rescale = vel_settings.get('rescale_energy',
                                   vel_settings.get('rescale'))
        particles = system.particles
        if rescale is not None and rescale is not False:
            if rescale > 0:
                kin_old = rescale - particles.vpot
                do_rescale = True
            else:
                logger.warning('Ignored re-scale 6.2%f < 0.0.', rescale)
                return 0.0, calculate_kinetic_energy(particles)[0]
        else:
            kin_old = calculate_kinetic_energy(particles)[0]
            do_rescale = False
        if vel_settings.get('aimless', False):
            vel, _ = rgen.draw_maxwellian_velocities(system)
            particles.vel = vel
        else:  # Soft velocity change, from a Gaussian distribution:
            dvel, _ = rgen.draw_maxwellian_velocities(
                system, sigma_v=vel_settings['sigma_v'])
            particles.vel = particles.vel + dvel
        if vel_settings.get('momentum', False):
            reset_momentum(particles)
        if do_rescale:
            system.rescale_velocities(rescale)
        kin_new = calculate_kinetic_energy(particles)[0]
        dek = kin_new - kin_old
        return dek, kin_new

    def __call__(self, system):
        """To allow calling `MDEngine(system)`.

        Here, we are just calling `self.integration_step(system)`.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The system we are integrating.

        Returns
        -------
        out : None
            Does not return anything, but will update the particles.

        """
        return self.integration_step(system)

    @staticmethod
    def calculate_order(ensemble, xyz=None, vel=None, box=None):
        """Return the order parameter.

        This method is just to help to calculate the order parameter
        in cases where only the engine can do it.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system` : object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.

        xyz : numpy.array, optional
            The positions to use. Typically for internal engines, this
            is not needed. It is included here as it can be used for
            testing and also to be compatible with the generic function
            defined by the parent.
        vel : numpy.array, optional
            The velocities to use.
        box : numpy.array, optional
            The current box vectors.

        Returns
        -------
        out : list of floats
            The calculated order parameter(s).

        """
        system = ensemble['system']
        order_function = ensemble['order_function']
        if any((xyz is None, vel is None, box is None)):
            return order_function.calculate(system)
        system.particles.pos = xyz
        system.particles.vel = vel
        system.box.update_size(box)
        return order_function.calculate(system)

    def kick_across_middle(self, ensemble, middle, tis_settings):
        """Force a phase point across the middle interface.

        This is accomplished by repeatedly kicking the phase point so
        that it crosses the middle interface.

        Parameters
        ----------
        ensemble : dict
            It contains:

            * `system` : object like :py:class:`.System`
              This is the system that contains the particles we are
              investigating
            * `order_function` : object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.
            * `rgen` : object like :py:class:`.RandomGenerator`
              This is the random generator that will be used.

        middle : float
            This is the value for the middle interface.
        tis_settings : dict
            This dictionary contains settings for TIS. Explicitly used here:

            * `zero_momentum`: boolean, determines if the momentum is zeroed.
            * `rescale_energy`: boolean, determines if energy is re-scaled.

        Returns
        -------
        out[0] : object like :py:class:`.System`
            The phase-point just before the interface.
        out[1] : object like :py:class:`.System`
            The phase-point just after the interface.

        Note
        ----
        This function will update the system state.

        """
        # We search for crossing with the middle interface and do this
        # by sequentially kicking the initial phase point:
        system = ensemble['system']
        previous = system.copy()
        system.potential_and_force()  # Make sure forces are set.
        curr = self.calculate_order(ensemble)[0]
        logger.info('Kicking from: %9.6f', curr)
        while True:
            # Save current state:
            previous = system.copy()
            # Modify velocities:
            self.modify_velocities(
                ensemble, {'sigma_v': None,
                           'aimless': True,
                           'momentum': tis_settings['zero_momentum'],
                           'rescale': tis_settings['rescale_energy']})
            # Update order parameter in case it is velocity dependent:
            curr = self.calculate_order(ensemble)[0]
            previous.order = curr
            # Store modified velocities:
            previous.particles.set_vel(system.particles.get_vel())
            # Integrate forward one step:
            self.integration_step(system)
            # Compare previous order parameter and the new one:
            prev = curr
            curr = self.calculate_order(ensemble)[0]
            if curr == middle:
                # By construction we want two points, one left and one
                # right of the interface, and these two points should
                # be connected by a MD step. If we hit exactly on the
                # interface we just fall back:
                system.particles = previous.particles.copy()
                curr = previous.order
                # TODO: This method should be improved and generalized.
                # The generalization should be done so that this method
                # is only defined once and not as it is now - defined
                # for several engines.
            else:
                if (prev <= middle < curr) or (curr < middle <= prev):
                    # Middle interface was crossed, just stop the loop.
                    logger.info('Crossing found: %9.6f %9.6f ', prev, curr)
                    break
                elif (prev <= curr < middle) or (middle < curr <= prev):
                    # We are getting closer, keep the new point.
                    pass
                else:  # We did not get closer, fall back to previous point.
                    system.particles = previous.particles.copy()
                    curr = previous.order
        system.order = curr
        return previous, system

    def dump_phasepoint(self, phasepoint, deffnm=None):
        """For compatibility with external integrators."""
        return

    def clean_up(self):
        """Clean up after using the engine.

        Currently, this is only included for compatibility with external
        integrators.

        """
        return

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

class GromacsEngine(ExternalMDEngine):
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
        path : object like :py:class:`pyretis.core.path.PathBase`
            This is the path we use to fill in phase-space points.
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
        logger.debug(status)
        success = False
        interfaces = ensemble['interfaces']
        left, _, right = interfaces
        # Dumping of the initial config were done by the parent, here
        # we will just use it:
        initial_conf = ensemble['system'].particles.get_pos()[0]
        # Get the current order parameter:
        order = self.calculate_order(ensemble)
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        # In some cases, we don't really have to perform a step as the
        # initial config might be left/right of the interface in
        # question. Here, we will perform a step anyway. This is to be
        # sure that we obtain energies and also a trajectory segment.
        # Note that all the energies are obtained after we are done
        # with the integration from the .edr file of the trajectory.
        msg_file.write('# Running grompp and mdrun (initial step).')
        out_files = self._execute_grompp_and_mdrun(initial_conf, name)
        # Define name of some files:
        tpr_file = out_files['tpr']
        cpt_file = out_files['cpt']
        traj_file = os.path.join(self.exe_dir, out_files['trr'])
        msg_file.write(f'# Trajectory file is: {traj_file}')
        conf_abs = os.path.join(self.exe_dir, out_files['conf'])
        # Note: The order parameter is calculated AT THE END of each iteration.
        msg_file.write('# Starting GROMACS.')
        msg_file.write('# Step order parameter cv1 cv2 ...')
        for i in range(path.maxlen):
            msg_file.write(
                f'{i} {" ".join([str(j) for j in order])}'
            )
            # We first add the previous phase point, and then we propagate.
            snapshot = {'order': order,
                        'config': (traj_file, i),
                        'vel_rev': reverse}
            phase_point = self.snapshot_to_system(ensemble['system'],
                                                  snapshot)
            status, success, stop, _ = self.add_to_path(path, phase_point,
                                                        left, right)
            if stop:
                logger.debug('GROMACS propagation ended at %i. Reason: %s',
                             i, status)
                break
            if i == 0:
                # This step was performed before entering the main loop.
                pass
            elif i > 0:
                out_extnd = self._extend_and_execute_mdrun(tpr_file, cpt_file,
                                                           name)
                out_files.update(out_extnd)
            # Calculate the order parameter using the current system:
            ensemble['system'].particles.set_vel(reverse)
            ensemble['system'].particles.set_pos((conf_abs, None))
            order = self.calculate_order(ensemble)
            # We now have the order parameter, for GROMACS just remove the
            # config file to avoid the GROMACS #conf_abs# backup clutter:
            self._removefile(conf_abs)
            msg_file.flush()
        logger.debug('GROMACS propagation done, obtaining energies')
        msg_file.write('# Propagation done.')
        msg_file.write(f'# Reading energies from: {out_files["edr"]}')
        msg_file.flush()
        energy = self.get_energies(out_files['edr'])
        path.update_energies(energy['kinetic en.'], energy['potential'])
        logger.debug('Removing GROMACS output after propagate.')
        remove = [val for key, val in out_files.items()
                  if key not in ('trr', 'gro', 'g96')]
        self._remove_files(self.exe_dir, remove)
        self._remove_gromacs_backup_files(self.exe_dir)
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
            It contains:

            * `system`: object like :py:class:`.System`
              The system object gives the initial state for the
              integration. The initial state is stored and the system is
              reset to the initial state when the integration is done.
            * `order_function`: object like :py:class:`.OrderParameter`
              The object used for calculating the order parameter.

        steps : integer
            The number of steps we are going to perform. Note that we
            do not integrate on the first step (e.g. step 0) but we do
            obtain the other properties. This is to output the starting
            configuration.
        thermo : string, optional
            Select the thermodynamic properties we are to calculate.

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
        out_files = {}
        conf_abs = None
        self.energy_terms = self.select_energy_terms(thermo)
        # For step zero, obtain the order parameter:
        if order_function:
            order = self.calculate_order(ensemble)
        else:
            order = None

        for i in range(steps):
            if i == 0:
                out_files = self._execute_grompp_and_mdrun(
                    initial_file,
                    'pyretis-gmx'
                )
                conf_abs = os.path.join(self.exe_dir, out_files['conf'])
            elif 0 < i < steps - 1:
                out_extnd = self._extend_and_execute_mdrun(
                    out_files['tpr'],
                    out_files['cpt'],
                    'pyretis-gmx'
                )
                out_files.update(out_extnd)
            else:
                pass
            # Update with results from previous step:
            results = {}
            if order:
                results['order'] = order
            # Update for order parameter:
            if order_function:
                system.particles.set_pos((conf_abs, None, None))
                order = self.calculate_order(ensemble)
            # Obtain latest energies:
            time1 = i * self.timestep * self.subcycles
            time2 = (i + 1) * self.timestep * self.subcycles
            # time1 and time2 should be correct now, but we are victims
            # of floating points. Subtract/add something small so that
            # we round to correct time.
            time1 -= self.timestep * 0.1
            time2 += self.timestep * 0.1
            energy = self.get_energies(out_files['edr'], begin=time1,
                                       end=time2)
            # Rename energies into the PyRETIS convention:
            results['thermo'] = self.rename_energies(energy)
            yield results

class GromacsEngine2(GromacsEngine):
    """
    A class for interfacing GROMACS.

    This class defines an interface to GROMACS. Attributes are similar
    to :py:class:`.GromacsEngine`. In this particular interface,
    GROMACS is executed without starting and stopping and we rely on
    reading the output TRR file from GROMACS while a simulation is
    running.
    """

    def __init__(self, gmx, mdrun, input_path, timestep, subcycles,
                 exe_path=os.path.abspath('.'),
                 maxwarn=0, gmx_format='gro', write_vel=True,
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
        super().__init__(gmx, mdrun, input_path, timestep, subcycles,
                         exe_path=exe_path,
                         maxwarn=maxwarn, gmx_format=gmx_format,
                         write_vel=write_vel, write_force=write_force)

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

def create_orderparameter(settings):
    """Create order parameters from settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    out : object like :py:class:`.OrderParameter`
        This object represents the order parameter.

    """
    main_order = create_external(
        settings,
        'orderparameter',
        order_factory,
        ['calculate'],
    )
    if main_order is None:
        logger.info('No order parameter created')
        return None
    logger.info('Created main order parameter:\n%s', main_order)

    extra_cv = []
    order_settings = settings.get('collective-variable', [])
    for order_setting in order_settings:
        order = create_external(
            settings,
            'collective-variable',
            order_factory,
            ['calculate'],
            key_settings=order_setting
        )
        logger.info('Created additional collective variable:\n%s', order)
        extra_cv.append(order)
    if not extra_cv:
        return main_order
    all_order = [main_order] + extra_cv
    order = CompositeOrderParameter(order_parameters=all_order)
    logger.info('Composite order parameter:\n%s', order)
    return order


def order_factory(settings):
    """Create order parameters according to the given settings.

    This function is included as a convenient way of setting up and
    selecting the order parameter.

    Parameters
    ----------
    settings : dict
        This defines how we set up and select the order parameter.

    Returns
    -------
    out : object like :py:class:`.OrderParameter`
        An object representing the order parameter.

    """
    factory_map = {
        'orderparameter': {
            'cls': OrderParameter
        },
        'position': {
            'cls': Position
        },
        'velocity': {
            'cls': Velocity
        },
        'distance': {
            'cls': Distance
        },
        'distancevel': {
            'cls': Distancevel
        },
        'positionvelocity': {
            'cls': PositionVelocity
        },
        'distancevelocity': {
            'cls': DistanceVelocity
        },
    }
    return generic_factory(settings, factory_map, name='orderparameter')

class OrderParameter:
    """Base class for order parameters.

    This class represents an order parameter and other collective
    variables. The order parameter is assumed to be a function
    that can uniquely be determined by the system object and its
    attributes.

    Attributes
    ----------
    description : string
        This is a short description of the order parameter.
    velocity_dependent : boolean
        This flag indicates whether or not the order parameter
        depends on the velocity direction. If so, we need to
        recalculate the order parameter when reversing trajectories.

    """

    def __init__(self, description='Generic order parameter', velocity=False):
        """Initialise the OrderParameter object.

        Parameters
        ----------
        description : string
            Short description of the order parameter.

        """
        self.description = description
        self.velocity_dependent = velocity
        if self.velocity_dependent:
            logger.debug(
                'Order parameter "%s" was marked as velocity dependent.',
                self.description
            )

    @abstractmethod
    def calculate(self, system):
        """Calculate the main order parameter and return it.

        All order parameters should implement this method as
        this ensures that the order parameter can be calculated.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This object contains the information needed to calculate
            the order parameter.

        Returns
        -------
        out : list of floats
            The order parameter(s). The first order parameter returned
            is used as the progress coordinate in path sampling
            simulations!

        """
        return

    def __str__(self):
        """Return a simple string representation of the order parameter."""
        msg = [
            f'Order parameter: "{self.__class__.__name__}"',
            f'{self.description}'
        ]
        if self.velocity_dependent:
            msg.append('This order parameter is velocity dependent.')
        return '\n'.join(msg)

    def load_restart_info(self, info):
        """Load the orderparameter restart info."""

    def restart_info(self):
        """Save any mutatable parameters for the restart."""

class Velocity(OrderParameter):
    """Initialise the order parameter.

    This class defines a very simple order parameter which is just
    the velocity of a given particle.

    Attributes
    ----------
    index : integer
        This is the index of the atom which will be used, i.e.
        ``system.particles.vel[index]`` will be used.
    dim : integer
        This is the dimension of the coordinate to use.
        0, 1 or 2 for 'x', 'y' or 'z'.

    """

    def __init__(self, index, dim='x'):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the velocity of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.

        """
        txt = f'Velocity of particle {index} (dim: {dim})'
        super().__init__(description=txt, velocity=True)
        self.index = index
        self.dim = {'x': 0, 'y': 1, 'z': 2}.get(dim, None)
        if self.dim is None:
            logger.critical('Unknown dimension %s requested', dim)
            raise ValueError

    def calculate(self, system):
        """Calculate the velocity order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the velocities.

        Returns
        -------
        out : list of floats
            The velocity order parameter.

        """
        return [system.particles.vel[self.index][self.dim]]


def _verify_pair(index):
    """Check that the given index contains a pair."""
    try:
        if len(index) != 2:
            msg = ('Wrong number of atoms for pair definition. '
                   f'Expected 2 got {len(index)}')
            logger.error(msg)
            raise ValueError(msg)
    except TypeError as err:
        msg = 'Atom pair should be defined as a tuple/list of integers.'
        logger.error(msg)
        raise TypeError(msg) from err


class Distance(OrderParameter):
    """A distance order parameter.

    This class defines a very simple order parameter which is just
    the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index, periodic=True):
        """Initialise order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = 'Periodic' if periodic else 'Non-periodic'
        txt = f'{pbc} distance, particles {index[0]} and {index[1]}'
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        self.index = index

    def calculate(self, system):
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The distance order parameter.

        """
        particles = system.particles
        delta = particles.pos[self.index[1]] - particles.pos[self.index[0]]
        if self.periodic:
            delta = system.box.pbc_dist_coordinate(delta)
        lamb = np.sqrt(np.dot(delta, delta))
        return [lamb]

class Position(OrderParameter):
    """A positional order parameter.

    This class defines a very simple order parameter which is just
    the position of a given particle.

    Attributes
    ----------
    index : integer
        This is the index of the atom which will be used, i.e.
        ``system.particles.pos[index]`` will be used.
    dim : integer
        This is the dimension of the coordinate to use.
        0, 1 or 2 for 'x', 'y' or 'z'.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the position or not.

    """

    def __init__(self, index, dim='x', periodic=False, description=None):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the position of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        if description is None:
            description = f'Position of particle {index} (dim: {dim})'
        super().__init__(description=description, velocity=False)
        self.periodic = periodic
        self.index = index
        self.dim = {'x': 0, 'y': 1, 'z': 2}.get(dim, None)
        if self.dim is None:
            msg = f'Unknown dimension {dim} requested'
            logger.critical(msg)
            raise ValueError(msg)

    def calculate(self, system):
        """Calculate the position order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions.

        Returns
        -------
        out : list of floats
            The position order parameter.

        """
        particles = system.particles
        pos = particles.pos[self.index]
        lamb = pos[self.dim]
        if self.periodic:
            lamb = system.box.pbc_coordinate_dim(lamb, self.dim)
        return [lamb]


class Distancevel(OrderParameter):
    """A rate of change of the distance order parameter.

    This class defines a very simple order parameter which is just
    the time derivative of the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index, periodic=True):
        """Initialise the order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = 'Periodic' if periodic else 'Non-periodic'
        txt = (f'{pbc} rate-of-change-distance, particles {index[0]} and '
               f'{index[1]}')
        super().__init__(description=txt, velocity=True)
        self.periodic = periodic
        self.index = index

    def calculate(self, system):
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The rate-of-change of the distance order parameter.

        """
        particles = system.particles
        delta = particles.pos[self.index[1]] - particles.pos[self.index[0]]
        if self.periodic:
            delta = system.box.pbc_dist_coordinate(delta)
        lamb = np.sqrt(np.dot(delta, delta))
        # Add the velocity as an additional collective variable:
        delta_v = particles.vel[self.index[1]] - particles.vel[self.index[0]]
        cv1 = np.dot(delta, delta_v) / lamb
        return [cv1]


class CompositeOrderParameter(OrderParameter):
    """A composite order parameter.

    This class represents a composite order parameter. It does not
    actually calculate order parameters itself, but it has references
    to several objects like :py:class:`.OrderParameter` which it can
    use to obtain the order parameters. Note that the first one of
    these objects will be interpreted as the main progress coordinate
    in path sampling simulations.

    Attributes
    ----------
    extra : list of objects like :py:class:`OrderParameter`
        This is a list of order parameters to calculate.

    """

    def __init__(self, order_parameters=None):
        """Set up the composite order parameter.

        Parameters
        ----------
        order_parameters : list of objects like :py:class:`.OrderParameter`
            A list of order parameters we can add.

        """
        super().__init__(description='Combined order parameter',
                         velocity=False)
        self.order_parameters = []
        if order_parameters is not None:
            for order_function in order_parameters:
                self.add_orderparameter(order_function)

    def calculate(self, system):
        """Calculate the main order parameter and return it.

        This is defined as a method just to ensure that at least this
        method will be defined in the different order parameters.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This object contains the information needed to calculate
            the order parameter.

        Returns
        -------
        out : list of floats
            The order parameter(s). The first order parameter returned
            is assumed to be the progress coordinate for path sampling
            simulations.

        """
        all_order = []
        for order_function in self.order_parameters:
            all_order.extend(order_function.calculate(system))
        return all_order

    def mirror(self):
        """Mirrors all of the functions that allow it."""
        order_p = self.order_parameters[0]
        op_mirror_func = getattr(order_p, 'mirror', None)
        if op_mirror_func is not None:
            op_mirror_func()
        else:
            msg = "Attempting a mirror move, but orderparameter of \n class"
            msg += f" '{type(order_p).__name__}' does not have the function"
            msg += " 'mirror()'.\n"
            msg += "Please use an OP of type 'Permeability' or implement your"
            msg += " own mirror() function"
            logger.warning(msg)
        # This is safe as compound OPs should always have more than 1 OP.
        for order_function in self.order_parameters[1:]:
            mirror_func = getattr(order_function, 'mirror', None)
            if mirror_func is not None:
                mirror_func()

    def restart_info(self):
        """Return the mutable attributes for restart."""
        return [op.restart_info() for op in self.order_parameters]

    def load_restart_info(self, info):
        """Load the mutable attributes for restart."""
        for i, op_info in enumerate(info):
            self.order_parameters[i].load_restart_info(op_info)

    def add_orderparameter(self, order_function):
        """Add an extra order parameter to calculate.

        Parameters
        ----------
        order_function : object like :py:class:`.OrderParameter`
            An object we can use to calculate the order parameter.

        Returns
        -------
        out : boolean
            Return True if we added the function, False otherwise.

        """
        # We check that the ``calculate`` method is present and callable.
        for func in ('calculate', ):
            objfunc = getattr(order_function, func, None)
            name = order_function.__class__.__name__
            if not objfunc:
                msg = f'Missing method "{func}" in order parameter {name}'
                logger.error(msg)
                raise ValueError(msg)
            if not callable(objfunc):
                msg = f'"{func}" in order parameter {name} is not callable!'
                raise ValueError(msg)
        self.velocity_dependent |= order_function.velocity_dependent
        if self.velocity_dependent:
            logger.debug(
                'Order parameter "%s" was marked as velocity dependent.',
                self.description
            )
        self.order_parameters.append(order_function)
        return True

    @property
    def index(self):
        """Get only the index that is tracked by the orderparameter."""
        return self.order_parameters[0].index

    @index.setter
    def index(self, var):
        """Set only the index that is tracked by the orderparameter."""
        self.order_parameters[0].index = var

    def __str__(self):
        """Return a simple string representation of the order parameter."""
        txt = ['Order parameter, combination of:']
        for i, order in enumerate(self.order_parameters):
            txt.append(f'{i}: {str(order)}')
        msg = '\n'.join(txt)
        return msg


class PositionVelocity(CompositeOrderParameter):
    """An order parameter equal to the position & velocity of a given atom."""

    def __init__(self, index, dim='x', periodic=False):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the position
            and velocity of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        position = Position(index, dim=dim, periodic=periodic)
        velocity = Velocity(index, dim=dim)
        orderparameters = [position, velocity]
        super().__init__(order_parameters=orderparameters)


class DistanceVelocity(CompositeOrderParameter):
    """An order parameter equal to a distance and its rate of change."""

    def __init__(self, index, periodic=True):
        """Initialise the order parameter.

        Parameters
        ----------
        index : tuple of integers
            These are the indices used for the two particles.
            `system.particles.pos[index[0]]` and
            `system.particles.pos[index[1]]` will be used.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        position = Distance(index, periodic=periodic)
        velocity = Distancevel(index, periodic=periodic)
        orderparameters = [position, velocity]
        super().__init__(order_parameters=orderparameters)

def box_from_restart(restart):
    """Create a box from restart settings.

    Parameters
    ----------
    restart : dict
        A dictionary with restart settings.

    Returns
    -------
    box : object like :py:class:`.BoxBase`
        The box created from the restart settings.

    """
    restart_box = restart.get('box', None)
    if restart_box is None:
        logger.info('No box created from restart settings.')
        return None
    box = create_box(
        low=restart_box.get('low'),
        high=restart_box.get('high'),
        cell=restart_box.get('cell'),
        periodic=restart_box.get('periodic')
    )
    return box

def particles_from_restart(restart):
    """Create particles from restart information.

    Parameters
    ----------
    restart : dict
        The restart settings.

    Returns
    -------
    particles : object like :py:class:`.Particles.`
        The object created from the restart information.

    """
    restart_particles = restart.get('particles', None)
    if restart_particles is None:
        logger.info('No particles were created from restart information.')
        return None
    particles = ParticlesExt(dim=restart_particles['dim'])
    particles.load_restart_info(restart_particles)
    return particles

class PathBase:
    """Base class for representation of paths.

    This class represents a path. A path consists of a series of
    consecutive snapshots (the trajectory) with the corresponding order
    parameter.

    Attributes
    ----------
    generated : tuple
        This contains information on how the path was generated.
        `generated[0]` : string, as defined in the variable `_GENERATED`
        `generated[1:]` : additional information:
        For ``generated[0] == 'sh'`` the additional information is the
        index of the shooting point on the old path, the new path and
        the corresponding order parameter.
    maxlen : int
        This is the maximum path length. Some algorithms require this
        to be set. Others don't, which is indicated by setting `maxlen`
        equal to None.
    ordermax : tuple
        This is the (current) maximum order parameter for the path.
        `ordermax[0]` is the value, `ordermax[1]` is the index in
        `self.path`.
    ordermin : tuple
        This is the (current) minimum order parameter for the path.
        `ordermin[0]` is the value, `ordermin[1]` is the index in
        `self.path`.
    phasepoints : list of objects like :py:class:`.System`
        The phase points the path is made up of.
    rgen : object like :py:class:`.RandomGenerator`
        This is the random generator that will be used for the
        paths that required random numbers.
    time_origin : int
        This is the location of the phase point `path[0]` relative to
        its parent. This might be useful for plotting.
    status : str or None
        The status of the path. The possibilities are defined
        in the variable `_STATUS`.
    weight : real
        The statistical weight of the path.

    """

    def __init__(self, rgen=None, maxlen=None, time_origin=0):
        """Initialise the PathBase object.

        Parameters
        ----------
        rgen : object like :py:class:`.RandomGenerator`, optional
            This is the random generator that will be used.
        maxlen : int, optional
            This is the max-length of the path. The default value,
            None, is just a path of arbitrary length.
        time_origin : int, optional
            This can be used to store the shooting point of a parent
            trajectory.

        """
        self.maxlen = maxlen
        self.weight = 1.
        self.time_origin = time_origin
        self.status = None
        self.generated = None
        self.phasepoints = []
        self.path_number = None
        self.traj_v = None
        if rgen is None:
            rgen = create_random_generator()
        self.rgen = rgen

    @property
    def length(self):
        """Compute the length of the path."""
        return len(self.phasepoints)

    @property
    def ordermin(self):
        """Compute the minimum order parameter of the path."""
        idx = np.argmin([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    @property
    def ordermax(self):
        """Compute the maximum order parameter of the path."""
        idx = np.argmax([i.order[0] for i in self.phasepoints])
        return (self.phasepoints[idx].order[0], idx)

    def check_interfaces(self, interfaces):
        """Check current status of the path.

        Get the current status of the path with respect to the
        interfaces. This is intended to determine if we have crossed
        certain interfaces or not.

        Parameters
        ----------
        interfaces : list of floats
            This list is assumed to contain the three interface values
            left, middle and right.

        Returns
        -------
        out[0] : str, 'L' or 'R' or None
            Start condition: did the trajectory start at the left ('L')
            or right ('R') interface.
        out[1] : str, 'L' or 'R' or None
            Ending condition: did the trajectory end at the left ('L')
            or right ('R') interface or None of them.
        out[2] str, 'M' or '*'
            'M' if the middle interface is crossed, '*' otherwise.
        out[3] : list of boolean
            These values are given by
            `ordermin < interfaces[i] <= ordermax`.

        """
        if self.length < 1:
            logger.warning('Path is empty!')
            return None, None, None, None
        ordermax, ordermin = self.ordermax[0], self.ordermin[0]
        cross = [ordermin < interpos <= ordermax for interpos in interfaces]
        left, right = min(interfaces), max(interfaces)
        # Check end & start point:
        end = self.get_end_point(left, right)
        start = self.get_start_point(left, right)
        middle = 'M' if cross[1] else '*'
        return start, end, middle, cross

    def get_end_point(self, left, right=None):
        """Return the end point of the path as a string.

        The end point is either to the left of the `left` interface or
        to the right of the `right` interface, or somewhere in between.

        Parameters
        ----------
        left : float
            The left interface.
        right : float, optional
            The right interface, equal to left if not specified.

        Returns
        -------
        out : string
            A string representing where the end point is ('L' - left,
            'R' - right or None).

        """
        if right is None:
            right = left
        assert left <= right

        if self.phasepoints[-1].order[0] <= left:
            end = 'L'
        elif self.phasepoints[-1].order[0] >= right:
            end = 'R'
        else:
            end = None
            logger.debug('Undefined end point.')
        return end

    def get_start_point(self, left, right=None):
        """Return the start point of the path as a string.

        The start point is either to the left of the `left` interface or
        to the right of the `right` interface.

        Parameters
        ----------
        left : float
            The left interface.
        right : float, optional
            The right interface, equal to left if not specified.

        Returns
        -------
        out : string
            A string representing where the start point is ('L' - left,
            'R' - right or None).

        """
        if right is None:
            right = left
        assert left <= right
        if self.phasepoints[0].order[0] <= left:
            start = 'L'
        elif self.phasepoints[0].order[0] >= right:
            start = 'R'
        else:
            start = None
            logger.debug('Undefined starting point.')
        return start

    @abstractmethod
    def get_shooting_point(self):
        """Return a shooting point from the path.

        Returns
        -------
        phasepoint : object like :py:class:`.System`
            A phase point which will be the state to shoot from.
        idx : int
            The index of the shooting point.

        """
        return

    @abstractmethod
    def phasepoint(self, idx):
        """Return a specific phase point.

        Parameters
        ----------
        idx : int
            Index for phase-space point to return.

        Returns
        -------
        out : tuple
            A phase-space point in the path.

        """
        return

    @abstractmethod
    def _append_posvel(self, pos, vel):
        """Append positions and velocities to the path."""
        return

    def append(self, phasepoint):
        """Append a new phase point to the path.

        Parameters
        ----------
        out : object like :py:class:`.System`
            The system information we add to the path.

        """
        if self.maxlen is None or self.length < self.maxlen:
            self.phasepoints.append(phasepoint)
            return True
        logger.debug('Max length exceeded. Could not append to path.')
        return False

    def get_path_data(self, status, interfaces):
        """Return information about the path.

        This information can be stored in a object like
        :py:class:`.PathEnsemble`.

        Parameters
        ----------
        status : string
            This represents the current status of the path.
        interfaces : list
            These are just the interfaces we are currently considering.

        """
        path_info = {
            'generated': self.generated,
            'status': status,
            'length': self.length,
            'ordermax': self.ordermax,
            'ordermin': self.ordermin,
            'weight': self.weight,
        }

        start, end, middle, _ = self.check_interfaces(interfaces)
        path_info['interface'] = (start, middle, end)

        return path_info

    def set_move(self, move):
        """Update the path move.

        The path move is a short string that represents how the path
        was generated. It should preferably match one of the moves
        defined in `_GENERATED`.

        Parameters
        ----------
        move : string
            A short description of the move.

        """
        if self.generated is None:
            self.generated = (move, 0, 0, 0)
        else:
            self.generated = (move, self.generated[1], self.generated[2],
                              self.generated[3])

    def get_move(self):
        """Return the move used to generate the path."""
        if self.generated is None:
            return None
        return self.generated[0]

    def success(self, target_interface):
        """Check if the path is successful.

        The check is based on the maximum order parameter and the value
        of `target_interface`. It is successful if the maximum order parameter
        is greater than `target_interface`.

        Parameters
        ----------
        target_interface : float
            The value for which the path is successful, i.e. the
            "target_interface" interface.

        """
        return self.ordermax[0] > target_interface

    def __iadd__(self, other):
        """Add path data to a path from another path, i.e. ``self += other``.

        This will simply append the phase points from `other`.

        Parameters
        ----------
        other : object of type `Path`
            The object to add path data from.

        Returns
        -------
        self : object of type `Path`
            The updated path object.

        """
        for phasepoint in other.phasepoints:
            app = self.append(phasepoint.copy())
            if not app:
                logger.warning(
                    'Truncated path at %d while adding paths', self.length
                )
                return self
        return self

    def copy(self):
        """Return a copy of the path."""
        new_path = self.empty_path()
        for phasepoint in self.phasepoints:
            new_path.append(phasepoint.copy())
        new_path.status = self.status
        new_path.time_origin = self.time_origin
        new_path.generated = self.generated
        new_path.maxlen = self.maxlen
        new_path.weight = self.weight
        new_path.path_number = self.path_number
        new_path.traj_v = self.traj_v
        return new_path

    @staticmethod
    def reverse_velocities(system):
        """Reverse the velocities in the phase points."""
        system.particles.reverse_velocities()

    def reverse(self, order_function=False, rev_v=True):
        """Reverse a path and return the reverse path as a new path.

        This will reverse a path and return the reversed path as
        a new object like :py:class:`.PathBase` object.

        Returns
        -------
        new_path : object like :py:class:`.PathBase`
            The time reversed path.
        order_function : object like :py:class:`.OrderParameter`, optional
            The method to use to re-calculate the order parameter,
            if it is velocity dependent.
        rev_v : boolean, optional
            If True, also the velocities are reversed, if False, the velocities
            for each frame are not altered.

        """
        new_path = self.empty_path()
        new_path.weight = self.weight
        new_path.maxlen = self.maxlen
        for phasepoint in reversed(self.phasepoints):
            new_point = phasepoint.copy()
            if rev_v:
                self.reverse_velocities(new_point)
            app = new_path.append(new_point)
            if not app:  # pragma: no cover
                msg = 'Could not reverse path'
                logger.error(msg)
                return None
        if order_function and order_function.velocity_dependent and rev_v:
            for phasepoint in new_path.phasepoints:
                phasepoint.order = order_function.calculate(phasepoint)
        return new_path

    def __str__(self):
        """Return a simple string representation of the Path."""
        msg = ['Path with length {} (max: {})'.format(self.length,
                                                      self.maxlen)]
        msg += ['Order parameter max: {}'.format(self.ordermax)]
        msg += ['Order parameter min: {}'.format(self.ordermin)]
        if self.length > 0:
            msg += ['Start {}'.format(self.phasepoints[0].order[0])]
            msg += ['End {}'.format(self.phasepoints[-1].order[0])]
        if self.status:
            msg += ['Status: {}'.format(_STATUS[self.status])]
        if self.generated:
            move = self.generated[0]
            txtmove = _GENERATED.get(move, 'unknown move')
            msg += ['Generated: {}'.format(txtmove)]
            msg += ['Weight: {}'.format(self.weight)]
        return '\n'.join(msg)

    @abstractmethod
    def restart_info(self):
        """Return a dictionary with restart information."""
        return

    @abstractmethod
    def load_restart_info(self):
        """Read a dictionary with restart information."""
        return

    @abstractmethod
    def empty_path(self, **kwargs):
        """Return an empty path of same class as the current one.

        This function is intended to spawn sibling paths that share some
        properties and also some characteristics of the current path.
        The idea here is that a path of a certain class should only be
        able to create paths of the same class.

        Returns
        -------
        out : object like :py:class:`.PathBase`
            A new empty path.

        """
        return

    def __eq__(self, other):
        """Check if two paths are equal."""
        if self.__class__ != other.__class__:
            logger.debug('%s and %s.__class__ differ', self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug('%s and %s.__dict__ differ', self, other)
            return False

        # Compare phasepoints:
        if not len(self.phasepoints) == len(other.phasepoints):
            return False
        for i, j in zip(self.phasepoints, other.phasepoints):
            if not i == j:
                return False
        if self.phasepoints:
            # Compare other attributes:
            for i in ('maxlen', 'time_origin', 'status', 'generated',
                      'length', 'ordermax', 'ordermin'):
                attr_self = hasattr(self, i)
                attr_other = hasattr(other, i)
                if attr_self ^ attr_other:  # pragma: no cover
                    logger.warning('Failed comparing path due to missing "%s"',
                                   i)
                    return False
                if not attr_self and not attr_other:
                    logger.warning(
                        'Skipping comparison of missing path attribute "%s"',
                        i)
                    continue
                if getattr(self, i) != getattr(other, i):
                    return False
        return True

    def __ne__(self, other):
        """Check if two paths are not equal."""
        return not self == other

    def delete(self, idx):
        """Remove a phase point from the path.

        Parameters
        ----------
        idx : integer
            The index of the frame to remove.

        """
        del self.phasepoints[idx]

    def sorting(self, key, reverse=False):
        """Re-order the phase points according to the given key.

        Parameters
        ----------
        key : string
            The attribute we will sort according to.
        reverse : boolean, optional
            If this is False, the sorting is from big to small.

        Yields
        ------
        out : object like :py:class:`.System`
            The ordered phase points from the path.

        """
        if key in ('ekin', 'vpot'):
            sort_after = [getattr(i.particles, key) for i in self.phasepoints]
        elif key == 'order':
            sort_after = [getattr(i, key)[0] for i in self.phasepoints]
        else:
            sort_after = [getattr(i, key) for i in self.phasepoints]
        idx = np.argsort(sort_after)
        if reverse:
            idx = idx[::-1]
        self.phasepoints = [self.phasepoints[i] for i in idx]

    def update_energies(self, ekin, vpot):
        """Update the energies for the phase points.

        This method is useful in cases where the energies are
        read from external engines and returned as a list of
        floats.

        Parameters
        ----------
        ekin : list of floats
            The kinetic energies to set.
        vpot : list of floats
            The potential energies to set.

        """
        if len(ekin) != len(vpot):
            logger.debug(
                'Kinetic and potential energies have different length.'
            )
        if len(ekin) != len(self.phasepoints):
            logger.debug(
                'Length of kinetic energy and phase points differ %d != %d.',
                len(ekin), len(self.phasepoints)
            )
        if len(vpot) != len(self.phasepoints):
            logger.debug(
                'Length of potential energy and phase points differ %d != %d.',
                len(vpot), len(self.phasepoints)
            )
        for i, phasepoint in enumerate(self.phasepoints):
            try:
                vpoti = vpot[i]
            except IndexError:
                logger.warning(
                    'Ran out of potential energies, setting to None.'
                )
                vpoti = None
            try:
                ekini = ekin[i]
            except IndexError:
                logger.warning(
                    'Ran out of kinetic energies, setting to None.'
                )
                ekini = None
            phasepoint.particles.vpot = vpoti
            phasepoint.particles.ekin = ekini


class Path(PathBase):
    """A path where the full trajectory is stored in memory.

    This class represents a path. A path consists of a series of
    consecutive snapshots (the trajectory) with the corresponding
    order parameter. Here we store all information for all phase points
    on the path.

    """

    def get_shooting_point(self, criteria='rnd', interfaces=None):
        """Return a shooting point from the path.

        This will simply draw a shooting point from the path at
        random. All points can be selected with equal probability with
        the exception of the end points which are not considered.

        Parameters
        ----------
        criteria : string, optional
            The criteria to select the shooting point:
            'rnd': random, except the first and last point, standard sh.
            'exp': selection towards low density region.
        list/tuple of floats, optional
          These are the interface positions of the form
          ``[left, middle, right]``.

        Returns
        -------
        out[0] : object like :py:class:`.System`
            The phase point we selected.
        out[1] : int
            The shooting point index.

        """
        keep_list = []
        rnd_idx = self.rgen.random_integers(1, self.length - 2)

        # The method is not done to be working on the 0^- ensemble.
        if criteria == 'exp' and interfaces[0] != float('-inf'):
            n_slabs = 42
            hyst = [0]*n_slabs
            for p_p in self.phasepoints:
                idx = abs(int((p_p.order[0] - interfaces[0]) /
                              (interfaces[-1] - interfaces[0])*n_slabs))
                hyst[idx] += 1

            # Exclude the extremis, no much interesting and always low value
            for i in [0, 1, 2, -3, -2, -1]:
                hyst[i] = max(hyst)

            # find the not zero minimum
            h_min = hyst.index(min(hyst))
            while hyst[h_min] == 0:
                hyst[h_min] = max(hyst)
                h_min = hyst.index(min(hyst))

            for idx, p_p in enumerate(self.phasepoints):
                i_slab = abs(int((p_p.order[0] - interfaces[0]) /
                                 (interfaces[-1] - interfaces[0])*n_slabs))
                if i_slab == h_min:
                    keep_list.append(idx)

        idx = rnd_idx if len(keep_list) < 4 else keep_list[
            self.rgen.random_integers(1, len(keep_list) - 2)]

        logger.debug("Selected point with orderp %s",
                     self.phasepoints[idx].order[0])
        return self.phasepoints[idx], idx

    def empty_path(self, **kwargs):
        """Return an empty path of same class as the current one.

        Returns
        -------
        out : object like :py:class:`.PathBase`
            A new empty path.

        """
        maxlen = kwargs.get('maxlen', None)
        time_origin = kwargs.get('time_origin', 0)
        return self.__class__(self.rgen, maxlen=maxlen,
                              time_origin=time_origin)

    def restart_info(self):
        """Return a dictionary with restart information."""
        info = {
            'rgen': self.rgen.get_state(),
            'generated': self.generated,
            'maxlen': self.maxlen,
            'time_origin': self.time_origin,
            'status': self.status,
            'weight': self.weight,
            'phasepoints': [i.restart_info() for i in self.phasepoints]
        }
        return info

    def load_restart_info(self, info):
        """Set up the path using restart information."""
        for key, val in info.items():
            # For phasepoints, create new System objects
            # and load the information for these.
            # The snaps still need to forcefield to be re-initiated.
            if key == 'phasepoints':
                for point in val:
                    system = System()
                    system.load_restart_info(point)
                    self.append(system)
            elif key == 'rgen':
                self.rgen = create_random_generator(info['rgen'])
            else:
                if hasattr(self, key):
                    setattr(self, key, val)

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

def box_matrix_to_list(matrix, full=False):
    """Return a list representation of the box matrix.

    This method ensures correct ordering of the elements for PyRETIS:
    ``xx, yy, zz, xy, xz, yx, yz, zx, zy``.

    Parameters
    ----------
    matrix : numpy.array
        A matrix (2D) representing the box.
    full : boolean, optional
        Return a full set of parameters (9) if set to True. If False,
        and we need 3 or fewer parameters (i.e. the other 6 are zero)
        we will only return the 3 non-zero ones.

    Returns
    -------
    out : list
        A list with the box-parametres.

    """
    if matrix is None:
        return None
    if np.count_nonzero(matrix) <= 3 and not full:
        return [matrix[0, 0], matrix[1, 1], matrix[2, 2]]
    return [matrix[0, 0], matrix[1, 1], matrix[2, 2],
            matrix[0, 1], matrix[0, 2], matrix[1, 0],
            matrix[1, 2], matrix[2, 0], matrix[2, 1]]

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

def counter():
    """Return how many times this function is called."""
    counter.count = 0 if not hasattr(counter, 'count') else counter.count + 1
    return counter.count

class OutputBase(metaclass=ABCMeta):
    """A generic class for handling output.

    Attributes
    ----------
    formatter : object like py:class:`.OutputFormatter`
        The object responsible for formatting output.
    target : string
        Determines where the target for the output, for
        instance "screen" or "file".
    first_write : boolean
        Determines if we have written something yet, or
        if this is the first write.

    """

    target = None

    def __init__(self, formatter):
        """Create the object and attach a formatter."""
        self.formatter = formatter
        self.first_write = True

    def output(self, step, data):
        """Use the formatter to write data to the file.

        Parameters
        ----------
        step : int
            The current step number.
        data : list
            The data we are going to output.

        """
        if self.first_write and self.formatter.print_header:
            self.first_write = False
            self.write(self.formatter.header)
        for line in self.formatter.format(step, data):
            self.write(line)

    @abstractmethod
    def write(self, towrite, end='\n'):
        """Write a string to the output defined by this class.

        Parameters
        ----------
        towrite : string
            The string to write.
        end : string, optional
            A "terminator" for the given string.

        Returns
        -------
        status : boolean
            True if we managed to write, False otherwise.

        """
        return

    def formatter_info(self):
        """Return a string with info about the formatter."""
        if self.formatter is not None:
            return self.formatter.__class__
        return None

    def __str__(self):
        """Return basic info."""
        return f'{self.__class__.__name__}\n\t* Formatter: {self.formatter}'

class FileIO(OutputBase):
    """A generic class for handling IO with files.

    This class defines how PyRETIS stores and reads data.
    Formatting is handled by an object like :py:class:`.OutputFormatter`

    Attributes
    ----------
    filename : string
        Name (e.g. path) to the file to read or write.
    file_mode : string
        Specifies the mode in which the file is opened.
    backup : boolean
        Determines the behavior if we want to write to a file
        that is already existing.
    fileh : object like :py:class:`io.IOBase`
        The file handle we are interacting with.
    last_flush : object like :py:class:`datetime.datetime`
        The previous time for flushing to the file.
    FILE_FLUSH : integer
        The interval for flushing to the file. That is, we will
        flush if the time since the last flush is larger than this
        value. Note that this is only checked in relation to writing.

    """

    target = 'file'
    FILE_FLUSH = 1  # Interval for flushing files in seconds.

    def __init__(self, filename, file_mode, formatter, backup=True):
        """Set up the file object.

        Parameters
        ----------
        filename : string
            The path to the file to open or read.
        file_mode : string
            Specifies the mode for opening the file.
        formatter : object like py:class:`.OutputFormatter`
            The object responsible for formatting output.
        backup : boolean, optional
            Defines how we handle cases where we write to a
            file which is already existing.

        """
        super().__init__(formatter)
        self.filename = filename
        self.file_mode = file_mode
        if backup not in (True, False):
            logger.info('Setting backup to default: True')
            self.backup = True
        else:
            self.backup = backup
        self.fileh = None
        if self.file_mode.startswith('a') and self.formatter is not None:
            self.formatter.print_header = False
        self.last_flush = None

    def open_file_read(self):
        """Open a file for reading."""
        if not self.file_mode.startswith('r'):
            raise ValueError(
                ('Inconsistent file mode "{}" '
                 'for reading').format(self.file_mode)
            )
        try:
            self.fileh = open(self.filename, self.file_mode)
        except (OSError, IOError) as error:
            logger.critical(
                'Could not open file "%s" for reading', self.filename
            )
            logger.critical(
                'I/O error ({%d}): {%s}', error.errno, error.strerror
            )
        return self.fileh

    def open_file_write(self):
        """Open a file for writing.

        In this method, we also handle the possible backup settings.
        """
        if not self.file_mode[0] in ('a', 'w'):
            raise ValueError(
                ('Inconsistent file mode "{}" '
                 'for writing').format(self.file_mode)
            )
        msg = []
        try:
            if os.path.isfile(self.filename):
                msg = ''
                if self.file_mode.startswith('a'):
                    logger.info(
                        'Appending to existing file "%s"', self.filename
                    )
                else:
                    if self.backup:
                        msg = create_backup(self.filename)
                        logger.debug(msg)
                    else:
                        logger.debug(
                            'Overwriting existing file "%s"', self.filename
                        )
            self.fileh = open(self.filename, self.file_mode)
        except (OSError, IOError) as error:  # pragma: no cover
            logger.critical(
                'Could not open file "%s" for writing', self.filename
            )
            logger.critical(
                'I/O error (%d): %d', error.errno, error.strerror
            )
        return self.fileh

    def open(self):
        """Open a file for reading or writing."""
        if self.fileh is not None:
            logger.debug(
                '%s asked to open file, but it has already opened a file.',
                self.__class__.__name__
            )
            return self.fileh
        if self.file_mode[0] in ('r',):
            return self.open_file_read()
        if self.file_mode[0] in ('a', 'w'):
            return self.open_file_write()
        raise ValueError('Unknown file mode "{}"'.format(self.file_mode))

    def load(self):
        """Read blocks or lines from the file."""
        return self.formatter.load(self.filename)

    def write(self, towrite, end='\n'):
        """Write a string to the file.

        Parameters
        ----------
        towrite : string
            The string to output to the file.
        end : string, optional
            Appended to `towrite` when writing, can be used to print a
            new line after the input `towrite`.

        Returns
        -------
        status : boolean
            True if we managed to write, False otherwise.

        """
        status = False
        if towrite is None:
            return status
        if self.fileh is not None and not self.fileh.closed:
            try:
                if end is not None:
                    self.fileh.write('{}{}'.format(towrite, end))
                    status = True
                else:
                    self.fileh.write(towrite)
                    status = True
            except (OSError, IOError) as error:  # pragma: no cover
                msg = 'Write I/O error ({}): {}'.format(error.errno,
                                                        error.strerror)
                logger.critical(msg)
            if self.last_flush is None:
                self.flush()
                self.last_flush = datetime.now()
            delta = datetime.now() - self.last_flush
            if delta.total_seconds() > self.FILE_FLUSH:  # pragma: no cover
                self.flush()
                self.last_flush = datetime.now()
            return status
        if self.fileh is not None and self.fileh.closed:
            logger.warning('Ignored writing to closed file %s', self.filename)
        if self.fileh is None:
            logger.critical(
                'Attempting to write to empty file handle for file %s',
                self.filename
            )
        return status

    def close(self):
        """Close the file."""
        if self.fileh is not None and not self.fileh.closed:
            try:
                self.flush()
            finally:
                self.fileh.close()

    def flush(self):
        """Flush file buffers to file."""
        if self.fileh is not None and not self.fileh.closed:
            self.fileh.flush()
            os.fsync(self.fileh.fileno())

    def output(self, step, data):
        """Open file before first write."""
        if self.first_write:
            self.open()
        return super().output(step, data)

    def __del__(self):
        """Close the file in case the object is deleted."""
        self.close()

    def __enter__(self):
        """Context manager for opening the file."""
        self.open()
        return self

    def __exit__(self, *args):
        """Context manager for closing the file."""
        self.close()

    def __iter__(self):
        """Make it possible to iterate over lines in the file."""
        return self

    def __next__(self):
        """Let the file object handle the iteration."""
        if self.fileh is None:
            raise StopIteration
        if self.fileh.closed:
            raise StopIteration
        return next(self.fileh)

    def __str__(self):
        """Return basic info."""
        msg = ['FileIO (file: "{}")'.format(self.filename)]
        if self.fileh is not None and not self.fileh.closed:
            msg += ['\t* File is open']
            msg += ['\t* Mode: {}'.format(self.fileh.mode)]
        msg += ['\t* Formatter: {}'.format(self.formatter)]
        return '\n'.join(msg)


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
