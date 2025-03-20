"""Test some utils from the lammps engine."""

import pathlib

import numpy as np
import pytest

from infretis.classes.engines.engineparts import (
    ReadAndProcessOnTheFly,
    lammpstrj_reader,
)
from infretis.classes.engines.lammps import (
    LAMMPSEngine,
    check_lammps_input,
    get_atom_masses,
    read_energies,
    read_lammpstrj,
    shift_boxbounds,
    write_lammpstrj,
    LammpsBox,
)

HERE = pathlib.Path(__file__).resolve().parent

ATOM_MASSES = np.array(
    [
        12.011,
        35.45,
        1.008,
        1.008,
        1.008,
        15.9994,
        1.008,
        1.008,
        35.45,
        99.9,
        99.9,
        99.9,
    ]
)
ATOM_TYPES = np.array([1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7])
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
BOX1 = np.array([20.0, 20.0, 20.0])
BOX_FRAMES = np.array(
    [
        [[0, 12.65], [0, 12.65], [0, 12.65]],
        [[2, 14.65], [1, 13.65], [3, 15.65]],
    ]
)
LAST_FRAME_POS = np.array([[5, 9, 4, 7, 3, 6, 1, 8, 2, 10, 11, 12]]).repeat(
    3, axis=0
)
LAST_FRAME_VEL = -LAST_FRAME_POS

def return_lammps_engine():
    """Set up a lammps engine for the H2 system."""
    lammps_input_path = HERE / "../../examples/lammps/H2/lammps_input"
    engine = LAMMPSEngine("lmp_mpi", lammps_input_path.resolve(), 0, 0, 300)
    engine.rgen = np.random.default_rng()
    engine.vel_settings = {
        "zero_momentum": False,
    }
    return engine


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
    natoms = 12
    for frame in range(2):
        id_type, pos, vel, box = read_lammpstrj(
            HERE / "data/lammps_files/traj.lammpstrj", frame, natoms
        )
    assert np.all(id_type[:, 1] == ATOM_TYPES)
    assert np.all(box == BOX_FRAMES[frame])


def test_write_lammpstrj(tmp_path):
    """Test that writing a lammpstrj is identical to the originl read."""
    natoms = 12
    id_type0, pos0, vel0, box0 = read_lammpstrj(
        HERE / "data/lammps_files/traj.lammpstrj", 0, natoms
    )
    write_lammpstrj(
        tmp_path / "test_write.lammpstrj", id_type0, pos0, vel0, box0
    )
    id_type, pos, vel, box = read_lammpstrj(
        tmp_path / "test_write.lammpstrj", 0, natoms
    )
    assert np.all(id_type == id_type0)
    assert np.all(pos == pos0)
    assert np.all(vel == vel0)
    assert np.all(box == box0)


def test_reverse_velocities(tmp_path):
    """Test that reversing the velocities only changes the velocities."""
    infile = HERE / "data/lammps_files/traj.lammpstrj"
    outfile = tmp_path / "test_reverse.lammpstrj"
    engine = return_lammps_engine()
    natoms = 12
    engine.n_atoms = natoms
    engine._reverse_velocities(infile, outfile)
    id_type0, pos0, vel0, box0 = read_lammpstrj(infile, 0, natoms)
    shifted_pos1, vel1, shifted_box1, _ = engine._read_configuration(outfile)
    assert np.all(vel0 == -vel1)


def test_shift_boxbounds():
    """Test that we can shift the box bounds."""
    natoms = 12
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


class PartialWriter:
    """A class to write a given file to a new file N chars at a time.

    Note that it reads the whole file into memory at once.

    """

    def __init__(self, read_file, write_file):

        with open(read_file) as rfile:
            self.content = rfile.read()

        self.read_file = read_file
        self.write_file = write_file
        self.n_chars_in_write_file = 0

    def add_chars_to_file(self, n):
        if n > 0:
            with open(self.write_file, "a") as wfile:
                start = self.n_chars_in_write_file
                wfile.write(self.content[start : start + n])
                self.n_chars_in_write_file += n


def test_check_lammps_input(tmp_path):
    """Test reading of a good and bad lammps.input file."""
    lmp_inp = HERE / "../../examples/lammps/H2/lammps_input/lammps.input"
    assert check_lammps_input(lmp_inp) is None

    bad_lmp_inp = tmp_path / "lammps.input"
    with open(bad_lmp_inp, "w") as wfile:
        with open(lmp_inp) as rfile:
            for line in rfile:
                if "dump" not in line:
                    wfile.write(line)
    with pytest.raises(ValueError):
        check_lammps_input(bad_lmp_inp)


def test_on_the_fly_read(tmp_path):
    """Test reading a trajectory that is written one character at a time."""
    partial_traj = tmp_path / "test.lammpstrj"

    writer = PartialWriter(
        HERE / "data/lammps_files/traj.lammpstrj", partial_traj
    )

    parser = ReadAndProcessOnTheFly(partial_traj, lammpstrj_reader)
    # no file exists, this returns []
    out = parser.read_and_process_content()
    assert out == []

    frames_correctly_read = 0
    for i in range(len(writer.content)):
        writer.add_chars_to_file(1)
        traj, box = parser.read_and_process_content()
        if traj != [] and box != []:
            assert np.all(box[0][:, :2] == BOX_FRAMES[frames_correctly_read])
            frames_correctly_read += 1

def test_LammpsCell():
    """Test the LammpsCell class"""
    # Regular orthogonal cell
    frame = 0
    natoms = 12
    id_type, pos, vel, box = read_lammpstrj(
        HERE / "data/lammps_files/traj.lammpstrj", frame, natoms
    )
    cell = LammpsBox(box)
    assert np.allclose(cell.lengths(), BOX)

    frame = 0
    natoms = 22
    id_type, pos, vel, box = read_lammpstrj(
        HERE / "data/lammps_files/traj1.lammpstrj", frame, natoms
    )
    cell = LammpsBox(box)
    assert np.allclose(cell.lengths(), BOX1)
