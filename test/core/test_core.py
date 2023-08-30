"""Test methods from infretis.core.core"""
import errno
import logging
import os  # Used to simulate errors for make_dirs
import pathlib

import pytest

from infretis.core.core import (
    generic_factory,
    make_dirs,
)

THIS_FILE = pathlib.Path(__file__).resolve()


def mock_mkdir(path, mode=0o777):
    if path is not None:
        raise PermissionError(errno.EPERM, "Permission denied")
    return path, mode


def test_make_dirs(tmp_path, monkeypatch):
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
    # Test that we fail with an error if we get another type of
    # exception. To do this, we here monky patch the make directories
    # method to just raise an error:
    with monkeypatch.context() as monkey:
        monkey.setattr(os, "mkdir", mock_mkdir)
        with pytest.raises(PermissionError) as errorinfo:
            msg = make_dirs(tmp_path / "some_directory_to_create")
            assert msg is None
            assert errorinfo.value.errno == errno.EPERM


class KlassForTesting:
    """A class for testing."""

    def __init__(self):
        self.variable = 10

    def method1(self):
        return self.variable

    def multiply(self, number):
        self.variable *= number


def test_generic_factory(caplog):
    """Test that we can create classes with the generic factory."""
    factory_map = {"my_new_class": {"cls": KlassForTesting}}

    # Check that we can create the class:
    settings = {"class": "my_new_class"}
    cls = generic_factory(settings, factory_map, name="Testing")
    assert isinstance(cls, KlassForTesting)

    # Check that we raises an error when no class is given
    # (this is case sensitive):
    settings = {"clAsS": "my_new_class"}
    with caplog.at_level(logging.CRITICAL):
        cls = generic_factory(settings, factory_map, name="Testing")
        assert "No class given for" in caplog.text
        assert cls is None
    caplog.clear()
    # Check that we fail when we request a class that is
    # not in the mapping:
    settings = {"class": "A class not in the mapping"}
    with caplog.at_level(logging.CRITICAL):
        cls = generic_factory(settings, factory_map, name="Testing")
        assert "Could not create unknown class" in caplog.text
        assert cls is None
