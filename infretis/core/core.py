from __future__ import annotations

import errno
import inspect
import logging
import os
import sys
from importlib import util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from inspect import Parameter

    from infretis.classes.engines.enginebase import EngineBase
    from infretis.classes.orderparameter import OrderParameter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generic_factory(
    settings: dict[str, Any],
    object_map: dict[str, dict[str, Any]],
    name: str = "generic",
) -> EngineBase | OrderParameter | None:
    """Create instances of classes based on settings.

    This method is intended as a semi-generic factory for creating
    instances of different objects based on simulation input settings.
    The input settings define what classes should be created and
    the object_map defines a mapping between settings and the
    class.

    Parameters
    ----------
    settings : dict
        This defines how we set up and select the order parameter.
    object_map : dict
        Definitions on how to initiate the different classes.
    name : string, optional
        Short name for the object type. Only used for error messages.

    Returns
    -------
    out : instance of a class
        The created object, in case we were successful. Otherwise we
        return none.

    """
    try:
        klass = settings["class"].lower()
    except KeyError:
        msg = "No class given for %s -- could not create object!"
        logger.critical(msg, name)
        return None
    if klass not in object_map:
        logger.critical(
            'Could not create unknown class "%s" for %s',
            settings["class"],
            name,
        )
        return None
    cls = object_map[klass]["class"]
    return initiate_instance(cls, settings)


def initiate_instance(
    klass: type[Any], settings: dict[str, Any]
) -> EngineBase | OrderParameter:
    """Initialise a class with optional arguments.

    Parameters
    ----------
    klass : class
        The class to initiate.
    settings : dict
        Positional and keyword arguments to pass to `klass.__init__()`.

    Returns
    -------
    out : instance of `klass`
        Here, we just return the initiated instance of the given class.

    """
    args, kwargs = _pick_out_arg_kwargs(klass, settings)
    # Ready to initiate:
    msg = 'Initiated "%s" from "%s" %s'
    name = klass.__name__
    mod = klass.__module__
    if not args:
        if not kwargs:
            logger.debug(msg, name, mod, "without arguments.")
            return klass()
        logger.debug(msg, name, mod, "with keyword arguments.")
        return klass(**kwargs)
    if not kwargs:
        logger.debug(msg, name, mod, "with positional arguments.")
        return klass(*args)
    logger.debug(msg, name, mod, "with positional and keyword arguments.")
    return klass(*args, **kwargs)


def _pick_out_arg_kwargs(
    klass: Any, settings: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Pick out arguments for a class from settings.

    Parameters
    ----------
    klass : class
        The class to initiate.
    settings : dict
        Positional and keyword arguments to pass to `klass.__init__()`.

    Returns
    -------
    out[0] : list
        A list of the positional arguments.
    out[1] : dict
        The keyword arguments.

    """
    info = inspect_function(klass.__init__)
    used, args, kwargs = set(), [], {}
    for arg in info["args"]:
        if arg == "self":
            continue
        try:
            args.append(settings[arg])
            used.add(arg)
        except KeyError as exc:
            msg = f'Required argument "{arg}" for "{klass}" not found!'
            raise ValueError(msg) from exc
    for arg in info["kwargs"]:
        if arg == "self":
            continue
        if arg in settings:
            kwargs[arg] = settings[arg]
    return args, kwargs


def inspect_function(function: Callable) -> dict[str, list[Any]]:
    """Return arguments/kwargs of a given function.

    This method is intended for use where we are checking that we can
    call a certain function. This method will return arguments and
    keyword arguments a function expects. This method may be fragile -
    we assume here that we are not really interested in args and
    kwargs and we do not look for more information about these here.

    Parameters
    ----------
    function : callable
        The function to inspect.

    Returns
    -------
    out : dict
        A dict with the arguments, the following keys are defined:

        * `args` : list of the positional arguments
        * `kwargs` : list of keyword arguments
        * `varargs` : list of arguments
        * `keywords` : list of keyword arguments

    """
    out = {
        "args": [],
        "kwargs": [],
        "varargs": [],
        "keywords": [],
    }  # type: dict[str, list[Any]]
    arguments = inspect.signature(function)  # pylint: disable=no-member
    for arg in arguments.parameters.values():
        kind = _arg_kind(arg)
        if kind is not None:
            out[kind].append(arg.name)
        else:  # pragma: no cover
            logger.critical(
                'Unknown variable kind "%s" for "%s"', arg.kind, arg.name
            )
    return out


def _arg_kind(arg: Parameter) -> str | None:
    """Determine kind for a given argument.

    This method will help :py:func:`.inspect_function` to determine
    the correct kind for arguments.

    Parameters
    ----------
    arg : object like :py:class:`inspect.Parameter`
        The argument we will determine the type of.

    Returns
    -------
    out : string
        A string we use for determine the kind.

    """
    kind = None
    if arg.kind == arg.POSITIONAL_OR_KEYWORD:
        if arg.default is arg.empty:
            kind = "args"
        else:
            kind = "kwargs"
    elif arg.kind == arg.POSITIONAL_ONLY:
        kind = "args"
    elif arg.kind == arg.VAR_POSITIONAL:
        kind = "varargs"
    elif arg.kind == arg.VAR_KEYWORD:
        kind = "keywords"
    elif arg.kind == arg.KEYWORD_ONLY:
        # We treat these as keyword arguments:
        kind = "kwargs"
    return kind


def create_external(settings, key, required_methods, key_settings=None):
    """Create external objects from settings.

    This method will handle the creation of objects from settings. The
    requested objects can be PyRETIS internals or defined in external
    modules.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.
    key : string
        The setting we are creating for.
    factory : callable
        A method to call that can handle the creation of internal
        objects for us.
    required_methods : list of strings
        The methods we need to have if creating an object from external
        files.
    key_settings : dict, optional
        This dictionary contains the settings for the specific key we
        are processing. If this is not given, we will try to obtain
        these settings by `settings[key]`. The reason why we make it
        possible to pass these as settings is in case we are processing
        a key which does not give a simple setting, but a list of settings.
        It that case `settings[key]` will give a list to process. That list
        is iterated somewhere else and `key_settings` can then be used to
        process these elements.

    Returns
    -------
    out : object
        This object represents the class we are requesting here.

    """
    klass = settings.get("class", None)
    module = settings.get("module", None)
    if key_settings is None:
        try:
            key_settings = settings[key]
        except KeyError:
            logger.debug('No "%s" setting found. Skipping set-up', key)
            return None
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is ok or if we should include
    # the 'exe_path' from settings.
    # 1) Check if we can find the module:
    if os.path.isfile(module):
        obj = import_from(module, klass)
    else:
        if "exe_path" in settings["simulation"]:
            module = os.path.join(settings["simulation"]["exe_path"], module)
            obj = import_from(module, klass)
        else:
            msg = f'Could not find module "{module}" for {key}!'
            raise ValueError(msg)
    # run some checks:
    for function in required_methods:
        objfunc = getattr(obj, function, None)
        if not objfunc:
            msg = f"Could not find method {klass}.{function}"
            logger.critical(msg)
            raise ValueError(msg)
        else:
            if not callable(objfunc):
                msg = f"Method {klass}.{function} is not callable!"
                logger.critical(msg)
                raise ValueError(msg)
    return initiate_instance(obj, settings)


def import_from(module_path: str, function_name: str) -> Any:
    """Import a method/class from a module.

    This method will dynamically import a specified method/object
    from a module and return it. If the module can not be imported or
    if we can't find the method/class in the module we will raise
    exceptions.

    Parameters
    ----------
    module_path : string
        The path/filename to load from.
    function_name : string
        The name of the method/class to load.

    Returns
    -------
    out : object
        The thing we managed to import.

    """
    msg = f"Could not import module: {module_path}"
    try:
        module_name = os.path.basename(module_path)
        module_name = os.path.splitext(module_name)[0]
        spec = util.spec_from_file_location(module_name, module_path)
        module = util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        sys.modules[module_name] = module
        logger.debug("Imported module: %s", module)
        return getattr(module, function_name)
    except (OSError, ImportError):
        msg = f"Could not import module: {module_path}"
        logger.critical(msg)
    except AttributeError:
        msg = f'Could not import "{function_name}" from "{module_path}"'
        logger.critical(msg)
    raise ValueError(msg)


def make_dirs(dirname: str) -> str:
    """Create directories for path simulations.

    This function will create a folder using a specified path.
    If the path already exists and if it's a directory, we will do
    nothing. If the path exists and is a file we will raise an
    `OSError` exception here.

    Parameters
    ----------
    dirname : string
        This is the directory to create.

    Returns
    -------
    out : string
        A string with some info on what this function did. Intended for
        output.

    """
    msg = f'Directory "{dirname}" was not created.'
    try:
        os.makedirs(dirname)
        msg = f'Created directory: "{dirname}"'
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise err
        if os.path.isfile(dirname):
            msg = f'"{dirname}" is a file. Will abort!'
            raise OSError(errno.EEXIST, msg) from err
        if os.path.isdir(dirname):
            msg = f'Directory "{dirname}" already exist.'
    return msg
