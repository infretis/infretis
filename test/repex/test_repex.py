import os
from pathlib import PosixPath

import tomli

from infretis.classes.repex import REPEX_state


def test_rgen_io(tmp_path: PosixPath) -> None:
    """Test repex rgen and rgen spawn reproducability."""
    state = REPEX_state(
        {
            "current": {"size": 1, "cstep": 0},
            "dask": {"workers": 1},
            "simulation": {},
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
        child = state.rgen.spawn(1)[0]
        save_rng_child.append(child.random())

    # restart with the "restarted_from" keyword
    with open("restart.toml", mode="rb") as f:
        config = tomli.load(f)
        config["current"]["restarted_from"] = {}
    state = REPEX_state(config)

    # test that the numbers are the same
    for rng, child_rng in zip(save_rng, save_rng_child):
        assert state.rgen.random() == rng
        child = state.rgen.spawn(1)[0]
        assert child.random() == child_rng
