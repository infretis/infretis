"""Test methods for doing TIS."""
import difflib
import filecmp
import os
import shutil
from pathlib import PosixPath
from subprocess import STDOUT, check_output

import pytest
import tomli
import tomli_w

from infretis.bin import internalrun

def get_diff_data(inp1, inp2):
    "Check the difference between two infretis_data.txt files in a simple way."
    diffps, diffms = [], []
    with open(inp1) as left, open(inp2) as right:
        diffs = difflib.unified_diff(left.readlines(), right.readlines(), n=0)
        for diff in diffs:  # difference is empty if no differences
            if "+\t" in diff[:2]:
                diffps.append(sum([int(i) for i in diff if i.isdigit()]))
            if "-\t" in diff[:2]:
                diffms.append(sum([int(i) for i in diff if i.isdigit()]))
    diffnum = 0
    for diffp, diffm in zip(diffps, diffms):
        diffnum += abs(diffp - diffm)
    return diffnum


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
    basepath = PosixPath(__file__).parent
    load_dir = (
        basepath / "../../examples/turtlemd/double_well/load_copy"
    ).resolve()
    toml_dir = basepath / "data/wf.toml"
    # copy files from template folder
    shutil.copytree(str(load_dir), str(folder) + "/load")
    shutil.copy(str(load_dir / "../orderp.py"), str(folder))
    shutil.copy(str(toml_dir), str(folder) + "/infretis.toml")
    os.chdir(folder)

    isnone = internalrun("infretis.toml")
    assert isnone is None

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/10steps_wf/{item}")

    change_toml_steps("restart.toml", 20)
    isnone = internalrun("restart.toml")
    assert isnone is None
    rm_restarted_from("restart.toml")

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/20steps_wf/{item}")

    change_toml_steps("restart.toml", 30)
    isnone = internalrun("restart.toml")
    assert isnone is None
    rm_restarted_from("restart.toml")

    # compare
    items = ["infretis_data.txt", "restart.toml"]
    for item in items:
        assert filecmp.cmp(f"./{item}", f"{basepath}/data/30steps_wf/{item}")

    # check the delete_old_all setting,
    # nb: technically num_files == 24, but due to restarts pn_olds get reset.
    num_files = len(os.listdir("load"))
    assert num_files < 40

    # run 30 steps without restart
    with open("infretis.toml", mode="rb") as f:
        config = tomli.load(f)
        config["simulation"]["steps"] = 30
        config["output"]["delete_old_all"] = False
    with open("infretis.toml", "wb") as f:
        tomli_w.dump(config, f)

    # remove restart.toml
    restart_file = PosixPath("restart.toml")
    restart_file.unlink()

    internalrun("infretis.toml")
    assert (
        get_diff_data(
            "infretis_data_1.txt",
            f"{basepath}/data/30steps_wf/infretis_data.txt",
        )
        < 5
    )
    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
        config["output"]["data_file"] = "./infretis_data.txt"
        config["output"]["tar_file"] = "./infretis_data.tar"
        config["output"]["delete_old_all"] = True
    with open("restart.toml", "wb") as f:
        tomli_w.dump(config, f)
    assert filecmp.cmp(
        "./restart.toml", f"{basepath}/data/30steps_wf/restart.toml"
    )


@pytest.mark.heavy
def test_restart_multiple_w(tmp_path: PosixPath) -> None:
    """Check that restarted workers continue the same tasks pre-restart."""
    folder = tmp_path / "temp"
    folder.mkdir()
    basepath = PosixPath(__file__).parent
    load_dir = (
        basepath / "../../examples/turtlemd/double_well/load_copy"
    ).resolve()
    toml_dir = basepath / "data/wf.toml"
    # copy files from template folder
    shutil.copytree(str(load_dir), str(folder) + "/load")
    shutil.copy(str(load_dir / "../orderp.py"), str(folder))
    shutil.copy(str(toml_dir), str(folder) + "/infretis.toml")
    os.chdir(folder)

    workers = 4
    with open("infretis.toml", mode="rb") as f:
        config = tomli.load(f)
        config["simulation"]["steps"] = 1000000
        config["runner"]["workers"] = workers
    with open("infretis.toml", "wb") as f:
        tomli_w.dump(config, f)

    # WIP: can possibly decrease timeout s. if github pipeline allows it.
    # start 4w 5s simulation
    try:
        check_output(
            ["infretisrun", "-i", "infretis.toml"], stderr=STDOUT, timeout=5
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

    # check that we have 4 worker.logs
    dirs = os.listdir(".")
    for i in range(workers):
        assert f"worker{i}.log" in dirs
