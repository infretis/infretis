import numpy as np
import logging
import struct
import os
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
TRR_DATA_ITEMS = ('box_size', 'vir_size', 'pres_size',
                  'x_size', 'v_size', 'f_size')

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

