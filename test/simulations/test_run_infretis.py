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
from infretis.classes.path import Path, restart_path
from infretis.classes.rgen import MockRandomGenerator
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
    print(folder, dir(folder))
    # copy files from template folder
    shutil.copytree(load_dir, str(folder) + '/load')
    shutil.copy(toml_dir, str(folder) + '/infretis.toml')
    os.chdir(folder)

    # print('whereami', os.getcwd())
    # change_toml_steps("infretis.toml", 30)
    ape = os.system("infretisrun -i infretis.toml >| out.txt")
    # print('bape', ape)
    # #### need to check whetehr the command ran succ or not

    # shutil.copy(str(folder) + '/worker0.log', basepath + '/worker0.log')
    # shutil.copy(str(folder) + '/sim.log', basepath + '/sim.log')
    # shutil.copy(str(folder) + '/restart.toml', basepath + '/restart.toml')
    # shutil.copy(str(folder) + '/infretis_data.txt', basepath + '/infretis_data.txt')
    # shutil.copy(str(folder) + '/out.txt', basepath + '/out.txt')
    # exit('a')

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/10steps_wf/{item}')

    change_toml_steps("restart.toml", 20)
    os.system("infretisrun -i restart.toml >> out.txt")
    rm_restarted_from("restart.toml")

    # # shutil.copy(str(folder) + '/load/23/energy.txt', basepath + '/energy.txt')
    # shutil.copy(str(folder) + '/sim.log', basepath + '/sim.log')
    # shutil.copy(str(folder) + '/worker0.log', basepath + '/worker0.log')
    # shutil.copy(str(folder) + '/restart.toml', basepath + '/restart.toml')
    # shutil.copy(str(folder) + '/infretis_data.txt', basepath + '/infretis_data.txt')
    # shutil.copy(str(folder) + '/out.txt', basepath + '/out.txt')
    # # shutil.copytree(str(folder) + '/load', basepath + '/load')

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/20steps_wf/{item}')

    change_toml_steps("restart.toml", 30)
    os.system("infretisrun -i restart.toml >> out.txt")
    rm_restarted_from("restart.toml")

    # # shutil.copy(str(folder) + '/load/23/energy.txt', basepath + '/energy.txt')
    # shutil.copy(str(folder) + '/sim.log', basepath + '/sim.log')
    # shutil.copy(str(folder) + '/restart.toml', basepath + '/restart.toml')
    # shutil.copy(str(folder) + '/infretis_data.txt', basepath + '/infretis_data.txt')
    # shutil.copy(str(folder) + '/out.txt', basepath + '/out.txt')
    # shutil.copytree(str(folder) + '/load', basepath + '/load')

    # compare
    items = ['infretis_data.txt', 'restart.toml']
    for item in items:
        assert filecmp.cmp(f'./{item}', f'{basepath}/data/30steps_wf/{item}')
