"""Test methods for doing TIS."""
import filecmp
import os
import shutil
from pathlib import PosixPath
from subprocess import STDOUT, check_output

import pytest
import tomli
import tomli_w


def read_simlogw(inp, workers=4, restart=False):
    winfo = [None for i in range(workers)]
    restartcnt = 0
    with open(inp) as f:
        for line in f:
            if "shooting" in line:
                rip = line.rstrip().split()
                if len(rip) == 12:
                    worker, ens_path = int(rip[-1]), [rip[5], rip[8]]
                else:
                    worker, ens_path = int(rip[-1]), [rip[5], rip[9]]
                winfo[worker] = ens_path
            if restart:
                if f"submit worker {workers-1} END" in line:
                    restartcnt += 1
                if restartcnt == 2:
                    break
    return sorted([f"{i[0]}-{i[1]}" for i in winfo])


def change_toml_steps(inp, steps):
    with open(inp, mode="rb") as f:
        config = tomli.load(f)
        config["simulation"]["steps"] = steps
    with open(inp, "wb") as f:
        tomli_w.dump(config, f)


def rm_restarted_from(inp):
    with open(inp, mode="rb") as f:
        config = tomli.load(f)
        config["current"].pop("restarted_from")
    with open(inp, "wb") as f:
        tomli_w.dump(config, f)


@pytest.mark.heavy
def test_run_airetis_wf(tmp_path: PosixPath) -> None:
    folder = tmp_path / "temp"
    folder.mkdir()
    basepath = os.path.dirname(__file__)
    load_dir = os.path.join(
        basepath, "../../examples/turtlemd/double_well/load_copy"
    )
    toml_dir = os.path.join(basepath, "data/wf.toml")
    # copy files from template folder
    shutil.copytree(load_dir, str(folder) + "/load")
    shutil.copy(toml_dir, str(folder) + "/infretis.toml")
    os.chdir(folder)

    success = os.system("infretisrun -i infretis.toml >| out.txt")
    assert success == 0

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/10steps_wf/{item}")

    change_toml_steps("restart.toml", 20)
    success = os.system("infretisrun -i restart.toml >> out.txt")
    assert success == 0
    rm_restarted_from("restart.toml")

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/20steps_wf/{item}")

    change_toml_steps("restart.toml", 30)
    success = os.system("infretisrun -i restart.toml >> out.txt")
    assert success == 0
    rm_restarted_from("restart.toml")

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/30steps_wf/{item}")

    # check the delete_old_all setting,
    # nb: technically num_files == 24, but due to restarts pn_olds get reset.
    num_files = len(os.listdir("load"))
    assert num_files < 40


@pytest.mark.heavy
def test_restart_multiple_w(tmp_path: PosixPath) -> None:
    """Check that restarted workers continue the same tasks pre-restart."""
    folder = tmp_path / "temp"
    folder.mkdir()
    basepath = os.path.dirname(__file__)
    load_dir = os.path.join(
        basepath, "../../examples/turtlemd/double_well/load_copy"
    )
    toml_dir = os.path.join(basepath, "data/wf.toml")
    # copy files from template folder
    shutil.copytree(load_dir, str(folder) + "/load")
    shutil.copy(toml_dir, str(folder) + "/infretis.toml")
    os.chdir(folder)

    workers = 4
    with open("infretis.toml", mode="rb") as f:
        config = tomli.load(f)
        config["simulation"]["steps"] = 1000000
        config["dask"]["workers"] = workers
    with open("infretis.toml", "wb") as f:
        tomli_w.dump(config, f)

    # WIP: can possibly decrease timeout s. if github pipeline allows it.
    # start 4w 5s simulation
    try:
        check_output(
            ["infretisrun", "-i", "infretis.toml"], stderr=STDOUT, timeout=3
        )
    except Exception:
        pass
    winfo1 = read_simlogw("sim.log", workers)

    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
        cstep = config["current"]["cstep"]
    # arbitrary small number
    assert cstep > 10

    # restart 4w 5s simulation
    try:
        check_output(
            ["infretisrun", "-i", "restart.toml"], stderr=STDOUT, timeout=3
        )
    except Exception:
        pass

    winfo2 = read_simlogw("sim.log", workers, restart=True)
    for work1, work2 in zip(winfo1, winfo2):
        assert work1 == work2
