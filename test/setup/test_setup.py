import copy
import os
from pathlib import Path, PosixPath

import pytest

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
    config = {"current": {"size": 10}, "output": {"data_dir": "./"}}

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
    original_config["simulation"]["multi_engine"] = True
    original_config["engine0"] = original_config["engine"]
    original_config["engine1"] = original_config["engine"]
    original_config["engine2"] = original_config.pop("engine")
    test_cases = [
        (["engine2", "timestep"], [10.0]),
    ]
    for keys, invalid_value in test_cases:
        copy.deepcopy(original_config)
