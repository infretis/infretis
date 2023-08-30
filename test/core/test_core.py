"""Test methods from infretis.core.core"""
import errno
import logging
import os  # Used to simulate errors for make_dirs
import pathlib

import pytest

from infretis.core.core import generic_factory, inspect_function, make_dirs

THIS_FILE = pathlib.Path(__file__).resolve()


def mock_mkdir(path, mode=0o777):
    """A version of mkdir that just makes an error."""
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


# Define some functions for testing the inspect method.
# These functions are just various permulations of using arguments
# and keyword arguments:


def function1(arg1, arg2, arg3, arg4):
    """To test positional arguments."""
    return arg1, arg2, arg3, arg4


def function2(arg1, arg2, arg3, arg4=10):
    """To test positional and keyword arguments."""
    return arg1, arg2, arg3, arg4


# pylint: disable=unused-argument
def function3(arg1, arg2, arg3, arg4=100, arg5=10):
    """To test positional and keyword arguments."""
    return arg1, arg2, arg3, arg4, arg5


def function4(*args, **kwargs):
    """To test positional and keyword arguments."""
    return args, kwargs


def function5(arg1, arg2, *args, arg3=100, arg4=100):
    """To test positional and keyword arguments."""
    return arg1, arg2, args, arg3, arg4


def function6(arg1, arg2, arg3=100, *, arg4=10):
    """To test positional and keyword arguments."""
    return arg1, arg2, arg3, arg4


def function7(arg1, arg2, arg3=100, *args, arg4, arg5=10):
    """To test positional and keyword arguments."""
    return arg1, arg2, arg3, arg4, arg5, *args


def function8(arg1, arg2=100, self="something"):
    """To test redifinition of __init__."""
    return arg1, arg2, self


functions_for_testing = [
    function1,
    function2,
    function3,
    function4,
    function5,
    function6,
    function7,
]
correctly_read_functions = [
    {
        "args": ["arg1", "arg2", "arg3", "arg4"],
        "varargs": [],
        "kwargs": [],
        "keywords": [],
    },
    {
        "args": ["arg1", "arg2", "arg3"],
        "varargs": [],
        "kwargs": ["arg4"],
        "keywords": [],
    },
    {
        "args": ["arg1", "arg2", "arg3"],
        "varargs": [],
        "kwargs": ["arg4", "arg5"],
        "keywords": [],
    },
    {"args": [], "varargs": ["args"], "kwargs": [], "keywords": ["kwargs"]},
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4"],
        "varargs": ["args"],
        "keywords": [],
    },
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4"],
        "varargs": [],
        "keywords": [],
    },
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4", "arg5"],
        "varargs": ["args"],
        "keywords": [],
    },
]


@pytest.mark.parametrize(
    "func, expected", zip(functions_for_testing, correctly_read_functions)
)
def test_inspect(func, expected):
    """Test that we can inspect methods."""
    args = inspect_function(func)
    assert args == expected


def test_arg_kind():
    """Test a function with only positional arguments."""
    # To get into the "arg.kind == arg.POSITIONAL_ONLY", just
    # use the equal method of range:
    args = inspect_function(range.__eq__)
    assert not args["keywords"]
    assert not args["varargs"]
    assert not args["kwargs"]
    assert "self" in args["args"]
    assert "value" in args["args"]
    assert len(args["args"]) == 2
