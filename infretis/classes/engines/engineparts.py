"""Helper methods for MD engines."""
import logging
import math
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO, Any

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


def _cos(angle: float) -> float:
    """Return cosine of an angle in degrees.

    Note:
        If the angle is close to 90.0 we return 0.0.

    Args:
        angle: The angle in degrees.

    Returns:
        The cosine of the angle.
    """
    if math.isclose(angle, 90.0):
        return 0.0
    return math.cos(math.radians(angle))


def box_vector_angles(
    length: np.ndarray, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Return the box matrix from given lengths and angles.

    Args:
        length: The box lengths as a 1D array.
        alpha: The alpha angle, in degrees.
        beta: The beta angle, in degrees.
        gamma: The gamma angle, in degrees.

    Returns:
        The (upper triangular) box matrix.

    Note:
        The angles and box lengths follow the convention from
        LAMMPS (https://docs.lammps.org/Howto_triclinic.html).
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
        length[1] * length[2] * cos_alpha - box_matrix[0, 1] * box_matrix[0, 2]
    ) / box_matrix[1, 1]
    box_matrix[2, 2] = math.sqrt(
        length[2] ** 2 - box_matrix[0, 2] ** 2 - box_matrix[1, 2] ** 2
    )
    return box_matrix


def box_matrix_to_list(
    matrix: np.ndarray, full: bool = False
) -> np.ndarray | None:
    """Flatten a box matrix to a list.

    This method flattens and orders a box matrix to the following order:
    `xx, yy, zz, xy, xz, yx, yz, zx, zy`.

    Args:
        matrix: A matrix (2D) representing the box.
        full: If True, this method returns the full set of 9 parameters.
            If False, only the diagonal elements will be returned.

    Returns:
        A list with the box parameters.
    """
    if matrix is None:
        return None
    if np.count_nonzero(matrix) <= 3 and not full:
        return np.array([matrix[0, 0], matrix[1, 1], matrix[2, 2]])
    return np.array(
        [
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
    )


def get_box_from_header(header: str) -> np.ndarray | None:
    """Get box lengths from a text header.

    Args:
        header: Text from which we will extract the box information.

    Returns:
        The box lengths.
    """
    low = header.lower()
    if low.find("box:") != -1:
        txt = low.split("box:")[1].strip()
        return np.array([float(i) for i in txt.split()])
    return None


def read_txt_snapshots(
    filename: str | Path, data_keys: tuple[str, ...] | None = None
) -> Iterator[dict[str, Any]]:
    """Read snapshots from a text file.

    Args:
        filename: Path to the file to read snapshots from.
        data_keys: A tuple representing the data we are to read,
            typically on the form: ('atomname', 'x', 'y', 'z', ...)`.

    Yields:
        A dictionary with the snapshot. The keys are the given
            `data_keys` and the values contain the corresponding data.
    """
    lines_to_read = 0
    snapshot: dict[str, Any] = {}
    if data_keys is None:
        data_keys = ("atomname", "x", "y", "z", "vx", "vy", "vz")
    read_header = False
    with open(filename, encoding="utf8") as fileh:
        for lines in fileh:
            if read_header:
                snapshot = {"header": lines.strip()}
                snapshot["box"] = get_box_from_header(snapshot["header"])
                read_header = False
                continue
            if lines_to_read == 0:  # new snapshot
                if snapshot:
                    yield snapshot
                try:
                    lines_to_read = int(lines.strip())
                except ValueError:
                    logger.error("Error in the input file %s", filename)
                    raise
                read_header = True
                snapshot = {}
            else:
                lines_to_read -= 1
                data = lines.strip().split()
                for i, (val, key) in enumerate(zip(data, data_keys)):
                    value = val.strip() if i == 0 else float(val)
                    try:
                        snapshot[key].append(value)
                    except KeyError:
                        snapshot[key] = [value]
    if snapshot:
        yield snapshot


def read_xyz_file(filename: str | Path) -> Iterator[dict[str, Any]]:
    """Read files in XYZ format.

    This method will read a XYZ file and yield the different snapshots
    found in the file.

    Args:
        filename: Path to the file to open.

    Yields:
        A dictionary containing the snapshot.
    """
    xyz_keys = ("atomname", "x", "y", "z", "vx", "vy", "vz")
    yield from read_txt_snapshots(filename, data_keys=xyz_keys)


def write_xyz_trajectory(
    filename: str,
    pos: np.ndarray,
    vel: np.ndarray,
    names: list[str] | None,
    box: np.ndarray | None,
    step: int | None = None,
    append: bool = True,
) -> None:
    """Write a XYZ snapshot to a trajectory file.

    This is intended as a lightweight alternative for just
    dumping snapshots to a trajectory file.

    Args:
        filename: Path to the file to write to.
        pos: The positions to write.
        vel: The velocities to write.
        names: Atom names to write.
        box: The box dimensions/vectors
        step: If the `step` is given, then the step number is
            written to the header.
        append: Determines if we append (if True) or overwrite (if False)
            if the `filename` file exists.
    """
    npart = len(pos)
    if names is None:
        names = ["X"] * npart
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


def convert_snapshot(
    snapshot: dict[str, Any]
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, list[str]]:
    """Convert a XYZ snapshot to numpy arrays.

    Args:
        snapshot: The dict containing a snapshot read from a XYZ-file.

    Returns:
        A tuple containing:
            - The box dimensions if we manage to read it.
            - The positions.
            - The velocities.
            - The atom names found in the file.
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


def look_for_input_files(
    input_path: str | Path,
    required_files: dict[str, str] | dict[str, Path],
    extra_files: dict[str, str] | dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Check that required files for a MD engines are present.

    It will first search for the default files. If not present,
    it will search for the files with the same extension.
    In this search, if there are no files or multiple files for
    a required extension, the function will raise an Error.
    There might also be optional files which are not required, but
    might be passed in here. If these are not present we will
    not fail, but delete the reference to this file.

    Args:
        input_path: Path to the directory where the input files are
            stored.
        required_files: These are the file names (and types as given by
            their extensions) of the required files.
        extra_files: These are the file names of the extra files.

    Returns
    -------
    out : dict
        The paths to the required and extra files we found.

    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        msg = f"Input path folder {str(input_path)} not existing"
        raise ValueError(msg)

    # Get the list of files in the input_path folder
    files_in_input_path = [i.name for i in input_path.iterdir() if i.is_file()]

    input_files: dict[str, Any] = {}
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
                input_files[file_type] = input_path / selected_file
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
        for file_type, file_to_check in extra_files.items():
            if file_to_check in files_in_input_path:
                input_files[file_type] = os.path.join(
                    input_path, file_to_check
                )
            else:
                msg = f"Extra file {file_to_check} not present in {input_path}"
                logger.info(msg)
    return input_files


class ReadAndProcessOnTheFly:
    """Read and process from an open fileobject on the fly.

    This method will read from an open fileobject on the fly,
    and do some processing on new data that is written to the file.
    Files should be opened using a 'with open' statement to be sure
    that they are closed.

    To do
    use with open in here. Point at current pos and read N finished blocks. Put
    pointer at that position and return traj. If only some frames ready, point
    at last whole ready block read and return [] or the ready frames.
    """

    def __init__(
        self,
        file_path: str | Path,
        processing_function: Callable[..., Any],
        read_mode: str = "r",
    ):
        """Create the reader object."""
        self.file_path = file_path
        self.processing_function = processing_function
        self.current_position = 0
        self.file_object: IO[Any] | None = None
        self.read_mode = read_mode

    def read_and_process_content(self) -> Any:
        """Read and process content from a file."""
        # we may open at a time where the file
        # is currently not open for reading
        try:
            with open(self.file_path, self.read_mode) as self.file_object:
                self.file_object.seek(self.current_position)
                self.previous_position = self.current_position
                return self.processing_function(self)
        except FileNotFoundError:
            return []


def xyz_reader(reader_class: ReadAndProcessOnTheFly) -> list[np.ndarray]:
    """Read XYZ-files on the fly."""
    # trajectory of ready frames to be returned
    trajectory: list[np.ndarray] = []
    # holder for storing frame coordinates
    frame_coordinates: list[list[float]] = []
    block_size = 0
    N_atoms = 0
    if reader_class.file_object is None:
        return trajectory
    for i, line in enumerate(iter(reader_class.file_object.readline, "")):
        spl = line.split()
        if i == 0 and spl:
            N_atoms = int(spl[0])
            block_size = N_atoms + 2  # 2 header lines
        # if we are not in the atom nr or header block
        if i % block_size > 1:
            # if there aren't enough values to iterate through
            # return the (possibly empty) ready trajectory frames
            if len(spl) != 4:
                reader_class.current_position = reader_class.previous_position
                return trajectory
            else:
                frame_coordinates.append([float(spl[i]) for i in range(1, 4)])
        # if we are done with one block
        # update the file object pointer to the new position
        if i % block_size == N_atoms + 1 and i > 0:
            trajectory.append(np.array(frame_coordinates))
            reader_class.current_position = reader_class.file_object.tell()
            frame_coordinates = []

    return trajectory


def lammpstrj_reader(
    reader_class: ReadAndProcessOnTheFly,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Return the coordinates, velocities and box bounds from a trajectory.

    For large trajectories, the whole thing is returned as one list
    and might shred through your ram.

    Note that the precision defaults to the numpy array precision.

    lammps format should be `dump custom id type x y z vx vy vz`
    which gives
    ITEM: TIMESTEP
    t
    ITEM: NUMBER OF ATOMS
    n
    ITEM: BOX BOUNDS xy xz yz pp pp pp
    xlo xhi (?)
    ylo yhi (?)
    zlo zhi (?)
    ITEM: ATOMS id type x y z vx vy vz
    n_atoms
    """
    # the ready frames to be returned
    # which will be a list of np.arrays
    trajectory: list[np.ndarray] = []
    # the corresponding box dimensions to be returned
    box: list[np.ndarray] = []
    box_snapshot = np.zeros(1)
    # the number of lines each snapshot takes in the trajectory
    # It is a placeholder for now
    block_size = 4
    N_atoms = 0
    coordinate_snapshot = np.zeros(1)
    if reader_class.file_object is None:
        return trajectory, box
    for i, line in enumerate(iter(reader_class.file_object.readline, "")):
        spl = line.split()
        if i == 3 and spl:
            N_atoms = int(spl[0])
            coordinate_snapshot = np.zeros((N_atoms, 6))
            box_snapshot = np.zeros((3, 3))
            # Natoms + timestep_block
            # + N_atoms_block + box_block + atoms_header
            block_size = N_atoms + 2 + 2 + 4 + 1
        # the line number if a single frame was extracted
        line_nr = i % block_size
        # if we are in the box bound block
        if line_nr >= 5 and line_nr <= 7:
            # frame is not ready
            n_box_cols = len(spl)
            if n_box_cols not in [2, 3]:
                reader_class.current_position = reader_class.previous_position
                return trajectory, box
            else:
                # we may have either 2 or 3 columns in box output
                box_snapshot[line_nr - 5] = spl + [0] * (3 - n_box_cols)
        # we are in the atoms block
        elif line_nr >= 9:
            # frame is not ready
            if len(spl) != 8:
                reader_class.current_position = reader_class.previous_position
                return trajectory, box
            else:
                # the atom number, which are not sorted by default in lammps
                atom = int(spl[0]) - 1
                coordinate_snapshot[atom, :] = spl[2:8]
        # if we are done with one block
        # update the file object pointer to the new position
        # and append the box and trajectory
        if i % block_size == block_size - 1 and i > 0:
            trajectory.append(coordinate_snapshot)
            box.append(np.array(box_snapshot))
            reader_class.current_position = reader_class.file_object.tell()
            # allocate memory for next frame
            coordinate_snapshot = np.zeros((N_atoms, 6))
            box_snapshot = np.zeros((3, 3))
    return trajectory, box
