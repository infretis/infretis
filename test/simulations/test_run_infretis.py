"""Test methods for doing TIS."""
import os
import tomli, tomli_w

import numpy as np
import shutil
import filecmp
from pathlib import PosixPath

from infretis.classes.engines.engineparts import read_xyz_file
from infretis.classes.engines.factory import create_engine
from infretis.classes.engines.turtlemd import TurtleMDEngine
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import Path
from infretis.core.tis import prepare_shooting_point, shoot, wire_fencing, compute_weight


def change_toml_steps(inp, steps):
    with open(inp, mode="rb") as f:
        config = tomli.load(f)
        config['simulation']['steps'] = steps
    with open(inp, "wb") as f:
        tomli_w.dump(config, f)

def rm_restarted_from(inp):
    with open(inp, mode="rb") as f:
        config = tomli.load(f)
        config["current"].pop("restarted_from")
    with open(inp, "wb") as f:
        tomli_w.dump(config, f)

def test_run_airetis_wf(tmp_path: PosixPath) -> None:
    folder = tmp_path / "temp"
    folder.mkdir()
    basepath = os.path.dirname(__file__)
    load_dir = os.path.join(basepath,
                            "../../examples/turtlemd/double_well/load")
    toml_dir = os.path.join(basepath,
                            "data/wf.toml")
    # copy files from template folder
    shutil.copytree(load_dir, str(folder) + '/load')
    shutil.copy(toml_dir, str(folder) + '/infretis.toml')
    os.chdir(folder)

    success = os.system("infretisrun -i infretis.toml >| out.txt")
    assert success == 0

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/10steps_wf/{item}')

    change_toml_steps("restart.toml", 20)
    success = os.system("infretisrun -i restart.toml >> out.txt")
    assert success == 0
    rm_restarted_from("restart.toml")

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/20steps_wf/{item}')

    change_toml_steps("restart.toml", 30)
    success = os.system("infretisrun -i restart.toml >> out.txt")
    assert success == 0
    rm_restarted_from("restart.toml")

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/30steps_wf/{item}')
