import math
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PERIODIC_TABLE = {
    "H": 1.007975,
    "He": 4.002602,
    "Li": 6.9675,
    "Be": 9.0121831,
    "B": 10.8135,
    "C": 12.0106,
    "N": 14.006855,
    "O": 15.9994,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.3055,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.0675,
    "Cl": 35.4515,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.045,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.3835,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
}

_XYZ_BIG_FMT = "{:5s}" + 3 * " {:15.9f}"
_XYZ_BIG_VEL_FMT = _XYZ_BIG_FMT + 3 * " {:15.9f}"


def _cos(angle):
    """Return cosine of an angle.

    We also check if the angle is close to 90.0 and if so, we return
    a zero.

    Parameters
    ----------
    angle : float
        The angle in degrees.

    Returns
    -------
    out : float
        The cosine of the angle.

    """
    if math.isclose(angle, 90.0):
        return 0.0
    return math.cos(math.radians(angle))


def box_vector_angles(length, alpha, beta, gamma):
    """Obtain the box matrix from given lengths and angles.

    Parameters
    ----------
    length : numpy.array
        1D array, the box-lengths on form ``[a, b, c]``.
    alpha : float
        The alpha angle, in degrees.
    beta : float
        The beta angle, in degrees.
    gamma : float
        The gamma angle, in degrees.

    Returns
    -------
    out : numpy.array, 2D
        The (upper triangular) box matrix.

    """
    box_matrix = np.zeros((3, 3))
    cos_alpha = _cos(alpha)
    cos_beta = _cos(beta)
    cos_gamma = _cos(gamma)
    box_matrix[0, 0] = length[0]
    box_matrix[0, 1] = length[1] * cos_gamma
    box_matrix[0, 2] = length[2] * cos_beta
    box_matrix[1, 1] = math.sqrt(length[1] ** 2 - box_matrix[0, 1] ** 2)
    box_matrix[1, 2] = (
        length[1] * length[2] * cos_alpha
        - box_matrix[0, 1] * box_matrix[0, 2]
    ) / box_matrix[1, 1]
    box_matrix[2, 2] = math.sqrt(
        length[2] ** 2 - box_matrix[0, 2] ** 2 - box_matrix[1, 2] ** 2
    )
    return box_matrix


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
    return [
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
    ]


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
    if low.find("box:") != -1:
        txt = low.split("box:")[1].strip()
        return np.array([float(i) for i in txt.split()])
    return None


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
        data_keys = ("atomname", "x", "y", "z", "vx", "vy", "vz")
    read_header = False
    with open(filename, "r", encoding="utf8") as fileh:
        for lines in fileh:
            if read_header:
                snapshot = {"header": lines.strip()}
                snapshot["box"] = get_box_from_header(snapshot["header"])
                read_header = False
                continue
            if lines_to_read == 0:  # new snapshot
                if snapshot is not None:
                    yield snapshot
                try:
                    lines_to_read = int(lines.strip())
                except ValueError:
                    logger.error("Error in the input file %s", filename)
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
    xyz_keys = ("atomname", "x", "y", "z", "vx", "vy", "vz")
    for snapshot in read_txt_snapshots(filename, data_keys=xyz_keys):
        yield snapshot


def write_xyz_trajectory(
    filename, pos, vel, names, box, step=None, append=True
):
    """Write XYZ snapshot to a trajectory.

    This is intended as a lightweight alternative for just
    dumping snapshots to a trajectory file.

    Parameters
    ----------
    filename : string
        The file name to dump to.
    pos : numpy.array
        The positions we are to write.
    vel : numpy.array
        The velocities we are to write.
    names : list of strings
        Atom names to write.
    box : numpy.array
        The box dimensions/vectors
    step : integer, optional
        If the ``step`` is given, then the step number is
        written to the header.
    append : boolean, optional
        Determines if we append or overwrite an existing file.

    Note
    ----
    We will here append to the file.

    """
    npart = len(pos)

    filemode = "a" if append else "w"
    with open(filename, filemode, encoding="utf-8") as output_file:
        output_file.write(f"{npart}\n")
        header = ["#"]
        if step is not None:
            header.append(f"Step: {step}")
        if box is not None:
            header.append(f'Box: {" ".join([f"{i:9.4f}" for i in box])}')
        header.append("\n")
        header_str = " ".join(header)
        output_file.write(header_str)
        for i in range(npart):
            line = _XYZ_BIG_VEL_FMT.format(
                names[i],
                pos[i, 0],
                pos[i, 1],
                pos[i, 2],
                vel[i, 0],
                vel[i, 1],
                vel[i, 2],
            )
            output_file.write(f"{line}\n")


def convert_snapshot(snapshot):
    """Convert a XYZ snapshot to numpy arrays.

    Parameters
    ----------
    snapshot : dict
        The dict containing a snapshot read from a XYZ-file.

    Returns
    -------
    box : numpy.array, 1D
        The box dimensions if we manage to read it.
    xyz : numpy.array
        The positions.
    vel : numpy.array
        The velocities.
    names : list of strings
        The atom names found in the file.

    """
    names = snapshot["atomname"]
    box = snapshot.get("box", None)
    natom = len(names)
    xyz = np.zeros((natom, 3))
    vel = np.zeros_like(xyz)
    for i, dim in enumerate(("x", "y", "z")):
        xyz[:, i] = snapshot[dim]
        key = f"v{dim}"
        if key in snapshot:
            vel[:, i] = snapshot[key]
    return box, xyz, vel, names


def look_for_input_files(input_path, required_files, extra_files=None):
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
        msg = f"Input path folder {input_path} not existing"
        raise ValueError(msg)

    # Get the list of files in the input_path folder
    files_in_input_path = [
        i.name for i in os.scandir(input_path) if i.is_file()
    ]

    input_files = {}
    # Check if the required files are present
    for file_type, file_to_check in required_files.items():
        req_ext = os.path.splitext(file_to_check)[1][1:].lower()
        if file_to_check in files_in_input_path:
            input_files[file_type] = os.path.join(input_path, file_to_check)
            logger.debug("%s input: %s", file_type, input_files[file_type])
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
                input_files[file_type] = os.path.join(
                    input_path, selected_file
                )
                logger.warning(
                    f"using {input_files[file_type]} "
                    + f'as "{file_type}" file'
                )
            else:
                msg = f'Missing input file "{file_to_check}" '
                if file_counter > 1:
                    msg += f'and multiple files have extension ".{req_ext}"'
                raise ValueError(msg)

    # Check if the extra files are present
    if extra_files:
        input_files["extra_files"] = []
        for file_to_check in extra_files:
            if file_to_check in files_in_input_path:
                input_files["extra_files"].append(file_to_check)
            else:
                msg = (
                    f"Extra file {file_to_check} not present in {input_path}"
                )
                logger.info(msg)

    return input_files
