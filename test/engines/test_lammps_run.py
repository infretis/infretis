import pathlib
import pytest
import tomli
import shutil
from distutils.dir_util import copy_tree
import os
import filecmp

import numpy as np

import tomli
import tomli_w
from check_installed_engine import detect_program

ENGINE = detect_program(["lmp", "lmp_mpi", "lmp_serial"])
HERE = pathlib.Path(__file__).resolve().parent

@pytest.mark.local
@pytest.mark.skipif(len(ENGINE) == 0, reason="lammps not installed")
def test_lammps_sim(tmp_path: pathlib.PosixPath) -> None:
    """Check that we get the same results between
    compressed and uncompressed lammps."""

    h2_stuff = HERE / "../../examples/lammps/H2/"
    h2_init = HERE / "../../infretis/tools/generate_H2_loadpaths.py"
    folder = tmp_path / "lmp"
    folder.mkdir()
    copy_tree(h2_stuff, str(folder) +'/')
    shutil.copy(h2_init, str(folder) +'/')
    os.chdir(folder)

    # read .toml and change som options
    with open("infretis.toml", "rb") as rfile:
        config = tomli.load(rfile)
    config["engine"]["lmp"] = ENGINE

    # for generate_H2_loadpaths
    with open("infretis.toml", "wb") as f:
        tomli_w.dump(config, f)

    config["engine"]["compressed"] = False
    config["runner"]["workers"] = 1
    config["simulation"]["steps"] = 10
    config["simulation"]["load_dir"] = "load_un"

    with open("infretis_un.toml", "wb") as f:
        tomli_w.dump(config, f)

    config["engine"]["compressed"] = True
    config["simulation"]["load_dir"] = "load_co"
    with open("infretis_co.toml", "wb") as f:
        tomli_w.dump(config, f)

    # get initial paths
    python = detect_program(["python", "python3"])
    success = os.system(f"{python} generate_H2_loadpaths.py")
    assert success == 0

    # import infretis.tools.generate_H2_loadpaths

    copy_tree("temporary_load", "load_un")
    copy_tree("temporary_load", "load_co")

    success = os.system("infretisrun -i infretis_un.toml >| out.txt")
    assert success == 0
    success = os.system("sed -i 's/lammpstrj/lammpstrj.gz/g' lammps_input/lammps.input")
    assert success == 0
    success = os.system("infretisrun -i infretis_co.toml >| out.txt")
    assert success == 0

    assert filecmp.cmp("infretis_data_1.txt", "infretis_data_2.txt")

    for pn in os.listdir("load_un"):
        if int(pn) < 10:
            continue
        assert filecmp.cmp(f"load_un/{pn}/order.txt", f"load_co/{pn}/order.txt")
        assert filecmp.cmp(f"load_un/{pn}/energy.txt", f"load_co/{pn}/energy.txt")

        # co trajectories have additional ".gz" endings
        data_un = np.loadtxt(f"load_un/{pn}/traj.txt", dtype="str")
        data_co = np.loadtxt(f"load_co/{pn}/traj.txt", dtype="str")
        assert (data_un[:, 0] == data_co[:, 0]).all()
        assert (data_un[:, 2] == data_co[:, 2]).all()
        assert (data_un[:, 3] == data_co[:, 3]).all()

        # add .gz
        data_un_gz = np.array([i + ".gz" for i in data_un[:, 1]])
        assert (data_un_gz == data_co[:, 1]).all()

        # check if files are there
        for file in set(data_un[:, 1]):
            os.path.isfile(f"load_un/{pn}/accepted/{file}")
        for file in set(data_co[:, 1]):
            os.path.isfile(f"load_co/{pn}/accepted/{file}")
