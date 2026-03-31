import copy
import os
from pathlib import Path, PosixPath

import pytest
import tomli_w

from infretis.setup import (
    TOMLConfigError,
    check_config,
    setup_config,
    write_header,
)

HERE = Path(__file__).resolve().parent


def test_write_header(tmp_path: PosixPath) -> None:
    """Test that we create new data file if datafile already present."""
    f1 = tmp_path / "temp"
    f1.mkdir()
    os.chdir(f1)
    config: dict = {"current": {"size": 10}, "output": {"data_dir": "./"}}

    # write the first infretis_data.txt file
    write_header(config)
    assert os.path.isfile("./infretis_data.txt")
    for i in range(1, 6):
        # create new infretis_data.txt files
        write_header(config)
        isfile = f"./infretis_data_{i}.txt"
        assert os.path.isfile(isfile)
        assert config["output"]["data_file"] == isfile


def set_nested_value(d, keys, value):
    """Set a value in a nested dictionary by following the list of keys,
    creating keys if they don't exist."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}  # Create a new dict if the key doesn't exist
        d = d[key]
    d[keys[-1]] = value


def test_check_config():
    toml_path = (
        Path(__file__).parent / "../../examples/gromacs/H2/infretis.toml"
    )
    original_config = setup_config(toml_path)
    test_cases = [
        (["runner", "workers"], 100),
        (["simulation", "tis_set", "interface_cap"], 100),
        (["simulation", "interfaces"], [0.0, 0.5, 0.2, 1.0]),
        (["simulation", "interfaces"], [0.0, 0.2, 0.2, 1.0]),
        (["simulation", "interfaces"], []),
    ]
    for keys, invalid_value in test_cases:
        config = copy.deepcopy(original_config)
        set_nested_value(config, keys, invalid_value)
        print("Testing:", keys, invalid_value)
        with pytest.raises(TOMLConfigError):
            check_config(config)


def test_multi_engine_config():
    toml_path = (
        Path(__file__).parent / "../../examples/gromacs/H2/infretis.toml"
    )
    original_config = setup_config(toml_path)
    original_config["simulation"]["tis_set"]["multi_engine"] = True
    original_config["engine0"] = original_config["engine"].copy()
    original_config["engine1"] = original_config["engine"].copy()
    original_config["engine2"] = original_config.pop("engine")
    test_cases = [
        (["engine2", "timestep"], 10.0),
    ]
    for keys, invalid_value in test_cases:
        config = copy.deepcopy(original_config)
        set_nested_value(config, keys, invalid_value)
        print("Testing:", keys, invalid_value)
        with pytest.raises(TOMLConfigError):
            check_config(config)


# ---------------------------------------------------------------------------
# setup_config restart/input handling tests
# ---------------------------------------------------------------------------

def _minimal_fresh_config(steps=100):
    """Build a minimal fresh config (no [current]) that passes check_config."""
    return {
        "runner": {"workers": 1},
        "simulation": {
            "interfaces": [0.0, 0.3, 0.6],
            "steps": steps,
            "seed": 0,
            "shooting_moves": ["sh", "wf", "wf"],
            "tis_set": {
                "n_jumps": 2,
                "maxlength": 100,
                "allowmaxlength": False,
                "zero_momentum": False,
            },
        },
        "engine": {
            "class": "turtlemd",
            "timestep": 0.002,
            "temperature": 0.07,
        },
        "output": {"data_dir": "./", "screen": 0},
    }


def _minimal_restart_config(cstep=50, steps=100, restarted_from=None):
    """Build a minimal restart config (with [current])."""
    cfg = _minimal_fresh_config(steps)
    cfg["current"] = {
        "traj_num": 3,
        "cstep": cstep,
        "active": [0, 1, 2],
        "locked": [],
        "size": 3,
        "frac": {},
        "wsubcycles": [0],
        "tsubcycles": 0,
    }
    if restarted_from is not None:
        cfg["current"]["restarted_from"] = restarted_from
    return cfg


def _write_toml(path, cfg):
    """Write a config dict to a TOML file."""
    with open(path, "wb") as f:
        tomli_w.dump(cfg, f)


def test_setup_config_fresh_start(tmp_path, monkeypatch):
    """Fresh config with no restart.toml -> normal fresh run."""
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path / "infretis.toml", _minimal_fresh_config(steps=100))

    result = setup_config(str(tmp_path / "infretis.toml"))
    assert result is not None
    assert result["current"]["cstep"] == 0
    assert "restarted_from" not in result["current"]


def test_setup_config_fresh_with_restart_redirects(tmp_path, monkeypatch):
    """Fresh config + restart.toml exists -> auto-redirect, merge steps."""
    monkeypatch.chdir(tmp_path)

    # Fresh config with steps=200
    _write_toml(tmp_path / "infretis.toml", _minimal_fresh_config(steps=200))

    # Restart file with cstep=50, steps=100 (old value)
    restart_cfg = _minimal_restart_config(cstep=50, steps=100)
    # Create fake traj dirs so active path check passes
    for act in restart_cfg["current"]["active"]:
        traj_dir = tmp_path / "trajs" / str(act)
        traj_dir.mkdir(parents=True)
        (traj_dir / "traj.txt").touch()
    restart_cfg["simulation"]["load_dir"] = str(tmp_path / "trajs")
    _write_toml(tmp_path / "restart.toml", restart_cfg)

    result = setup_config(str(tmp_path / "infretis.toml"))
    assert result is not None
    # State comes from restart.toml
    assert result["current"]["cstep"] == 50
    # steps merged from fresh config
    assert result["simulation"]["steps"] == 200
    # restarted_from was set
    assert result["current"]["restarted_from"] == 50


def test_setup_config_restart_continues(tmp_path, monkeypatch):
    """restart.toml with cstep < steps -> continues."""
    monkeypatch.chdir(tmp_path)

    restart_cfg = _minimal_restart_config(cstep=50, steps=100)
    for act in restart_cfg["current"]["active"]:
        traj_dir = tmp_path / "trajs" / str(act)
        traj_dir.mkdir(parents=True)
        (traj_dir / "traj.txt").touch()
    restart_cfg["simulation"]["load_dir"] = str(tmp_path / "trajs")
    _write_toml(tmp_path / "restart.toml", restart_cfg)

    result = setup_config(str(tmp_path / "restart.toml"))
    assert result is not None
    assert result["current"]["cstep"] == 50
    assert result["current"]["restarted_from"] == 50


def test_setup_config_complete_returns_none(tmp_path, monkeypatch):
    """restart.toml with cstep >= steps -> returns None (complete)."""
    monkeypatch.chdir(tmp_path)

    restart_cfg = _minimal_restart_config(cstep=100, steps=100)
    _write_toml(tmp_path / "restart.toml", restart_cfg)

    result = setup_config(str(tmp_path / "restart.toml"))
    assert result is None


def test_setup_config_restarted_from_equal_cstep_continues(
    tmp_path, monkeypatch
):
    """cstep == restarted_from but cstep < steps -> must continue (Bug B)."""
    monkeypatch.chdir(tmp_path)

    # This is the exact scenario that was broken: a completed run wrote
    # restarted_from = cstep, user bumped steps, relaunched.
    restart_cfg = _minimal_restart_config(
        cstep=100, steps=200, restarted_from=100
    )
    for act in restart_cfg["current"]["active"]:
        traj_dir = tmp_path / "trajs" / str(act)
        traj_dir.mkdir(parents=True)
        (traj_dir / "traj.txt").touch()
    restart_cfg["simulation"]["load_dir"] = str(tmp_path / "trajs")
    _write_toml(tmp_path / "restart.toml", restart_cfg)

    result = setup_config(str(tmp_path / "restart.toml"))
    assert result is not None, (
        "setup_config must not block continuation when cstep == restarted_from "
        "but cstep < steps"
    )
    assert result["current"]["cstep"] == 100


def test_setup_config_steps_merge_from_fresh(tmp_path, monkeypatch):
    """Fresh config steps=500, restart has steps=100 -> merged to 500."""
    monkeypatch.chdir(tmp_path)

    _write_toml(tmp_path / "infretis.toml", _minimal_fresh_config(steps=500))

    restart_cfg = _minimal_restart_config(cstep=100, steps=100)
    for act in restart_cfg["current"]["active"]:
        traj_dir = tmp_path / "trajs" / str(act)
        traj_dir.mkdir(parents=True)
        (traj_dir / "traj.txt").touch()
    restart_cfg["simulation"]["load_dir"] = str(tmp_path / "trajs")
    _write_toml(tmp_path / "restart.toml", restart_cfg)

    result = setup_config(str(tmp_path / "infretis.toml"))
    assert result is not None
    assert result["simulation"]["steps"] == 500, (
        "simulation.steps must come from the fresh config, not restart.toml"
    )
