"""Test some utils from the lammps engine."""

import pathlib

import numpy as np

from infretis.classes.engines.engineparts import (
    ReadAndProcessOnTheFly,
    lammpstrj_reader,
)
from infretis.classes.engines.lammps import (
    get_atom_masses,
    read_energies,
    read_lammpstrj,
    shift_boxbounds,
    write_lammpstrj,
)

HERE = pathlib.Path(__file__).resolve().parent

ATOM_MASSES = np.array(
    [12.011, 35.45, 1.008, 1.008, 1.008, 15.9994, 1.008, 1.008, 35.45]
)
ATOM_TYPES = np.array([1, 2, 3, 3, 3, 4, 5, 5, 6])
ENERGIES = np.array(
    [
        840.63779,
        849.66006,
        869.09075,
        818.6732,
        835.96235,
        848.68126,
        811.74666,
        769.69304,
        780.0135,
        760.957,
        748.75368,
        734.62429,
    ]
)
BOX = np.array([12.65, 12.65, 12.65])
BOX_FRAMES = np.array(
    [
        [[0, 12.65], [0, 12.65], [0, 12.65]],
        [[2, 14.65], [1, 13.65], [3, 15.65]],
    ]
)
LAST_FRAME_POS = np.array([[5, 9, 4, 7, 3, 6, 1, 8, 2]]).repeat(3, axis=0)
LAST_FRAME_VEL = -LAST_FRAME_POS


def test_read_masses():
    """Test that we can read unsorted masses from a lammps.data file."""
    masses = get_atom_masses(
        HERE / "data/lammps_files/unsorted_atom_style_full.data",
        atom_style="full",
    )
    assert np.all(masses.flatten() == ATOM_MASSES)


def test_read_energies():
    """Test reading kinetic energies from a log.lammps file."""
    energies = read_energies(HERE / "data/lammps_files/log.lammps")
    assert np.all(energies["KinEng"] == ENERGIES)


def test_lammpstrj_reader():
    """Test that we read a trajectory with the on-the-fly-reader correctly."""
    parser = ReadAndProcessOnTheFly(
        HERE / "data/lammps_files/traj.lammpstrj",
        lammpstrj_reader,
    )
    posvel, box = parser.read_and_process_content()
    assert np.all(posvel[-1][:, :3].T == LAST_FRAME_POS)
    assert np.all(posvel[-1][:, -3:].T == LAST_FRAME_VEL)


def test_read_lammpstrj():
    """Test that we can read a single frame from a lammps trajectory."""
    natoms = 9
    for frame in range(2):
        id_type, pos, vel, box = read_lammpstrj(
            HERE / "data/lammps_files/traj.lammpstrj", frame, natoms
        )
    assert np.all(id_type[:, 1] == ATOM_TYPES)
    assert np.all(box == BOX_FRAMES[frame])


def test_write_lammpstrj(tmp_path):
    """Test that writing a lammpstrj is identical to the originl read."""
    natoms = 9
    id_type0, pos0, vel0, box0 = read_lammpstrj(
        HERE / "data/lammps_files/traj.lammpstrj", 0, natoms
    )
    write_lammpstrj(tmp_path / "test_write.lammpstrj", id_type0, pos0, vel0, box0)
    id_type, pos, vel, box = read_lammpstrj(
        tmp_path / "test_write.lammpstrj", 0, natoms
    )
    assert np.all(id_type == id_type0)
    assert np.all(pos == pos0)
    assert np.all(vel == vel0)
    assert np.all(box == box0)


def test_shift_boxbounds():
    """Test that we can shift the box bounds."""
    natoms = 9
    # a shift from box [0, 12.65] should return same positions
    id_type, pos, vel, box = read_lammpstrj(
        HERE / "data/lammps_files/traj.lammpstrj", 0, natoms
    )
    new_pos, new_box = shift_boxbounds(pos, box)
    assert np.all(new_pos == pos)
    assert np.all(new_box == BOX)
    id_type, pos, vel, box = read_lammpstrj(
        HERE / "data/lammps_files/traj.lammpstrj", 1, natoms
    )
    new_pos, new_box = shift_boxbounds(pos, box)
    assert np.all(new_box == BOX)
    new_pos[:, 0] += BOX_FRAMES[1][0, 0]
    new_pos[:, 1] += BOX_FRAMES[1][1, 0]
    new_pos[:, 2] += BOX_FRAMES[1][2, 0]
    assert np.all(new_pos.T == LAST_FRAME_POS)
