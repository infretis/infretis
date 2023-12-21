import os
from pathlib import PosixPath

from infretis.setup import write_header

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
