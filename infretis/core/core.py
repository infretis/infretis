"""Methods for XYZ."""
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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generic_factory(
    settings: dict[str, Any],
    object_map: dict[str, Any],
    name: str = "generic",
) -> Any | None:
    """Create instances of classes based on settings.

    This method is intended as a semi-generic factory for creating
    instances of different objects based on the given settings.
    The settings define what classes should be created and
    the object_map defines a mapping between settings and the
    class.

    Args:
        settings: Settings for initiating the class.
        object_map: A mapping with the known classes.
        name: Short name for the object type. Only used for error messages.

    Returns:
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


def initiate_instance(klass: type[Any], settings: dict[str, Any]) -> Any:
    """Initialise a class with optional arguments.

    Args;
        klass: The class to initiate.
        settings: Positional and keyword arguments to pass
            to `klass.__init__()`.

    Returns:
        The initiated instance of the given `klass`.
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
    """Extract arguments required by class initialization.

    Extracts arguments and keyword arguments from a settings dictionary
    that are required to initialize an instance of the specified class.

    Args:
        klass: The class for which to find initialization arguments.
        settings: Positional and keyword arguments to pass
            to `klass.__init__()`.

    Returns:
        out[0]: A list of the positional arguments.
        out[1]: The keyword arguments.

    Raises:
        ValueError: If a required argument for the class initialization
                    is not found in the given settings.

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
    """Extract arguments/kwargs of the given function.

    This method is intended for use where we are checking that we can
    call a certain function. This method will return arguments and
    keyword arguments a function expects, so that other methods
    can check if these arguments are available, e.g., in a
    dictionary of simulation settings.

    Args:
        function: The function to inspect.

    Returns:
        A dictionary containing four keys:
            - `args`: A list of names of standard positional arguments.
            - `kwargs`: A list of names of keyword arguments with
                default values.
            - `varargs`: A list containing the name of the *args
                parameter, if present.
            - `keywords`: A list containing the name of the **kwargs
                parameter, if present.

    Note:
        This method does not check the *args and **kwargs in detail,
        it will just extract their name, if present.
    """
    out = {
        "args": [],
        "kwargs": [],
        "varargs": [],
        "keywords": [],
    }  # type: dict[str, list[Any]]
    arguments = inspect.signature(function)
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
    """Determine kind (positional, keyword, etc.) for a given parameter.

    This helper method will support :py:func:`.inspect_function` by
    categorizing each parameter of a function into specific types such
    as positional, keyword with default values, variable positional,
    and variable keyword.

    Args:
        arg: The parameter we will determine the kind of.

    Returns:
        A string representation of the kind ("args", "kwargs",
        "varargs", or "keywords"). Returns None if the argument
        kind does not match any of these.

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


def create_external(
    settings: dict[str, Any], key: str, required_methods: list[str]
) -> Any:
    """Create external objects from settings.

    This method will handle the creation of objects from settings. The
    requested objects can be internals or defined in external modules.

    Args:
        settings: This dictionary contains the settings we use to create
            the object. Specifically, we look up the "class" and
            "module" names. If these are not given, we use the
            "exe_path" setting from the "simulation" section to
            find the absolute path to the module.
        key: The setting we are creating the object for. This argument
            is only used in an error message.
        required_methods: A list of method names that the created
            object must implement.

    Returns:
        This object represents the class we are requesting here.

    """
    klass = settings.get("class", "")
    module = settings.get("module", "")
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is OK or if we should include
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
    """Import a method/class from a specified module.

    This method will dynamically import a specified method/class
    given the module's file path and the name of the method/class.
    Exceptions are raised if the module can not be imported or if
    the method/class is not found in the module.

    Args:
        module_path: The path/filename to load from.
        function_name: The name of the method/class to import from
        the module.

    Returns:
        A referemce to the imported method or class.

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
    """Create a directory at a specified path.

    This function attempts to create a directory using the provided
    path. If the directory already exists, no action is taken, and a
    message indicating this is returned. If the path exists but is a
    file, an `OSError` is raised.

    Args:
        dirname: This path of the directory to create.

    Returns:
        A message indicating the outcome of the operation.

    Raises:
        OSError: If the path exists and is a file.

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
