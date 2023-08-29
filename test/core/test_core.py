"""Test methods from infretis.core.core"""
import pathlib

import pytest

from infretis.core.core import make_dirs

THIS_FILE = pathlib.Path(__file__).resolve()


def test_make_dirs(tmp_path):
    """Test that we can create directories."""
    # Test that we fail if we try to create a directory
    # with the same name as a file that already exists:
    with pytest.raises(OSError):
        msg = make_dirs(THIS_FILE)
        assert "is a file. Will abort!" in msg
    # Test that we can create an empty directory:
    test_dir = tmp_path / "this_is_a_new_directory"
    msg = make_dirs(test_dir)
    assert "Created directory" in msg
    assert f"{test_dir}" in msg
    # Test that we fail, but without an error if the
    # directory already exists:
    msg = make_dirs(test_dir)
    assert "already exist" in msg
    assert f"{test_dir}" in msg
