"""Test methods from infretis.core.core"""
import errno
import inspect
import logging
import os  # Used to simulate errors for make_dirs
import pathlib
import sys  # Used to make a local import
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.random import RandomState

from infretis.core.core import (
    _pick_out_arg_kwargs,
    generic_factory,
    import_from,
    initiate_instance,
    inspect_function,
    make_dirs,
)

THIS_FILE = pathlib.Path(__file__).resolve()
LOCAL_DIR = THIS_FILE.parent.resolve()


def mock_mkdir(path, mode=0o777):
    """A version of mkdir that just makes an error."""
    if path is not None:
        raise PermissionError(errno.EPERM, "Permission denied")
    return path, mode


@contextmanager
def change_dir(dest):
    """Context manger for chdir"""
    cwd = os.getcwd()
    os.chdir(dest)
    try:
        yield
    finally:
        os.chdir(cwd)


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


class ClassForTesting:
    """A class for testing."""

    def __init__(self):
        self.variable = 10

    def method1(self):
        return self.variable

    def multiply(self, number):
        self.variable *= number


def test_generic_factory(caplog):
    """Test that we can create classes with the generic factory."""
    factory_map = {"my_new_class": {"cls": ClassForTesting}}

    # Check that we can create the class:
    settings = {"class": "my_new_class"}
    cls = generic_factory(settings, factory_map, name="Testing")
    assert isinstance(cls, ClassForTesting)

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
    },
    {
        "args": ["arg1", "arg2", "arg3"],
        "kwargs": ["arg4"],
    },
    {
        "args": ["arg1", "arg2", "arg3"],
        "kwargs": ["arg4", "arg5"],
    },
    {"varargs": ["args"], "keywords": ["kwargs"]},
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4"],
        "varargs": ["args"],
    },
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4"],
    },
    {
        "args": ["arg1", "arg2"],
        "kwargs": ["arg3", "arg4", "arg5"],
        "varargs": ["args"],
    },
]

for item in correctly_read_functions:
    for key in ("args", "kwargs", "varargs", "keywords"):
        if key not in item:
            item[key] = []


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


def test_pick_out_kwargs():
    """Test pick out of "self" for kwargs."""
    # In some cases, "self" might be given as a keyword argument,
    # this will check that we pick it out correctly so that
    # it is not passed along
    settings = {"arg1": 10, "arg2": 100, "self": "text"}

    class Abomination:
        """Just to allow redefinition of __init__."""

    abo = Abomination()
    abo.__init__ = function8

    args, kwargs = _pick_out_arg_kwargs(abo, settings)
    assert "self" not in args
    assert "self" not in kwargs


class ClassForTestingNoArg:
    def __init__(self):
        """Without any arguments."""


class ClassForTestingArg:
    def __init__(self, variable):
        """With argument."""
        self.variable = variable


class ClassForTestingKwarg:
    def __init__(self, variable=51):
        """With argument."""
        self.variable = variable


class ClassForTestingArgKwarg:
    def __init__(self, variable1, variable2=123):
        self.variable1 = variable1
        self.variable2 = variable2


def test_initiate_instance(caplog):
    """Test that we can initiate classes."""
    # First, a class without any arguments:
    with caplog.at_level(logging.DEBUG):
        cls = initiate_instance(ClassForTestingNoArg, {})
        assert isinstance(cls, ClassForTestingNoArg)
        assert "without arguments." in caplog.text
    caplog.clear()
    # A class with only keyword arguments:
    with caplog.at_level(logging.DEBUG):
        cls = initiate_instance(ClassForTestingKwarg, {})
        assert isinstance(cls, ClassForTestingKwarg)
        assert "without arguments." in caplog.text
        assert cls.variable == 51

        cls = initiate_instance(ClassForTestingKwarg, {"variable": 1})
        assert isinstance(cls, ClassForTestingKwarg)
        assert "with keyword arguments." in caplog.text
        assert cls.variable == 1
    caplog.clear()
    # A class with only arguments:
    with caplog.at_level(logging.DEBUG):
        cls = initiate_instance(ClassForTestingArg, {"variable": 1})
        assert isinstance(cls, ClassForTestingArg)
        assert "with positional arguments." in caplog.text
        assert cls.variable == 1
    caplog.clear()
    # A class with arguments and keyword arguments:
    with caplog.at_level(logging.DEBUG):
        cls = initiate_instance(ClassForTestingArgKwarg, {"variable1": 1})
        assert isinstance(cls, ClassForTestingArgKwarg)
        assert "with positional arguments." in caplog.text
        assert cls.variable1 == 1
        assert cls.variable2 == 123

        cls = initiate_instance(
            ClassForTestingArgKwarg, {"variable1": 1, "variable2": 2}
        )
        assert isinstance(cls, ClassForTestingArgKwarg)
        assert "with positional and keyword arguments." in caplog.text
        assert cls.variable1 == 1
        assert cls.variable2 == 2
    # Test that we fail when we miss an argument:
    with pytest.raises(ValueError):
        initiate_instance(ClassForTestingArg, {})


def test_import_from():
    """Test that we can import dynamically."""
    module = LOCAL_DIR / "foobarbaz.py"
    klass = "Foo"
    imp = import_from(module, klass)
    sys.path.insert(0, str(LOCAL_DIR))
    from foobarbaz import Foo

    del sys.path[0]
    assert inspect.getsource(imp) == inspect.getsource(Foo)


def test_import_from_errors(caplog):
    """Test that we handle the errors for import_from."""
    module = LOCAL_DIR / "foobarbaz.py"
    klass = "DoesNotExist"
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError):
            import_from(module, klass)
        assert f'Could not import "{klass}" from' in caplog.text
    caplog.clear()

    module = LOCAL_DIR / "this-file-should-not-exist.py"
    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(ValueError):
            import_from(module, klass)
        assert "Could not import module" in caplog.text

### def write_repex_restart(tmp_path):
# def test_write_ensemble_restart(tmp_path):
#     ensemble = {"rgen": RandomState(1234)}
#     config = {"simulation": {"load_dir": "this-is-a-folder"}}
#     test_dir = tmp_path / config["simulation"]["load_dir"] / "test"
#     make_dirs(test_dir)
#     with change_dir(tmp_path):
#         write_ensemble_restart(ensemble, config, "test")
#         info = read_restart_file(test_dir / "ensemble.restart")
#         state = ensemble["rgen"].get_state()
#         assert len(info["rgen"]) == len(state)
#         for i, (vali, valj) in enumerate(zip(state, info["rgen"])):
#             if i == 1:
#                 assert np.array_equal(vali, valj)
#             else:
#                 assert vali == valj
