import os
from pathlib import PosixPath
import tomli

from infretis.classes.repex import REPEX_state, spawn_rng


def test_rgen_io(tmp_path: PosixPath) -> None:
    """Test repex rgen and rgen spawn reproducability."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0, "restarted_from": -1},
            "runner": {"workers": 1},
            "simulation": {"seed": 0, "steps":10},
        }
    )
    folder = tmp_path / "temp"
    folder.mkdir()
    os.chdir(folder)

    # save initial state for restart
    state.write_toml()

    # generate numbers
    save_rng = []
    save_rng_child = []
    for i in range(5):
        save_rng.append(state.rgen.random())
        child = spawn_rng(state.rgen)
        save_rng_child.append(child.random())

    # restart with the "restarted_from" keyword
    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
        config["current"]["restarted_from"] = -1
    state = REPEX_state(config)

    # test that the numbers are the same
    for rng, child_rng in zip(save_rng, save_rng_child):
        assert state.rgen.random() == rng
        child = spawn_rng(state.rgen)
        assert child.random() == child_rng
