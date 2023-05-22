import inspect
import os
import importlib
import errno
import sys
import logging
import pickle
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def generic_factory(settings, object_map, name='generic'):
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
        klass = settings['class'].lower()
    except KeyError:
        msg = 'No class given for %s -- could not create object!'
        logger.critical(msg, name)
        return None
    if klass not in object_map:
        print('zoopa')
        logger.critical('Could not create unknown class "%s" for %s',
                        settings['class'], name)
        return None
    cls = object_map[klass]['cls']
    return initiate_instance(cls, settings)

def initiate_instance(klass, settings):
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
    print('hell', klass, settings, args, kwargs)
    # Ready to initiate:
    msg = 'Initiated "%s" from "%s" %s'
    name = klass.__name__
    mod = klass.__module__
    if not args:
        if not kwargs:
            logger.debug(msg, name, mod, 'without arguments.')
            return klass()
        logger.debug(msg, name, mod, 'with keyword arguments.')
        return klass(**kwargs)
    if not kwargs:
        logger.debug(msg, name, mod, 'with positional arguments.')
        return klass(*args)
    logger.debug(msg, name, mod,
                 'with positional and keyword arguments.')
    print('bimbo a')
    return klass(*args, **kwargs)

def _pick_out_arg_kwargs(klass, settings):
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
    for arg in info['args']:
        if arg == 'self':
            continue
        try:
            args.append(settings[arg])
            used.add(arg)
        except KeyError:
            msg = f'Required argument "{arg}" for "{klass}" not found!'
            raise ValueError(msg)
    for arg in info['kwargs']:
        if arg == 'self':
            continue
        if arg in settings:
            kwargs[arg] = settings[arg]
    return args, kwargs

def inspect_function(function):
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
    out = {'args': [], 'kwargs': [],
           'varargs': [], 'keywords': []}
    arguments = inspect.signature(function)  # pylint: disable=no-member
    for arg in arguments.parameters.values():
        kind = _arg_kind(arg)
        if kind is not None:
            out[kind].append(arg.name)
        else:  # pragma: no cover
            logger.critical('Unknown variable kind "%s" for "%s"',
                            arg.kind, arg.name)
    return out

def _arg_kind(arg):
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
            kind = 'args'
        else:
            kind = 'kwargs'
    elif arg.kind == arg.POSITIONAL_ONLY:
        kind = 'args'
    elif arg.kind == arg.VAR_POSITIONAL:
        kind = 'varargs'
    elif arg.kind == arg.VAR_KEYWORD:
        kind = 'keywords'
    elif arg.kind == arg.KEYWORD_ONLY:
        # We treat these as keyword arguments:
        kind = 'kwargs'
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
    klass = settings.get('class', None)
    module = settings.get('module', None)
    if key_settings is None:
        try:
            key_settings = settings[key]
        except KeyError:
            logger.debug('No "%s" setting found. Skipping set-up', key)
            return None
    # module = key_settings.get('module', None)
    # klass = None
    # try:
    #     klass = key_settings['class']
    # except KeyError:
    #     logger.debug('No "class" setting for "%s" specified. Skipping set-up',
    #                  key)
    #     return None
    # print('clown 3')
    # if module is None:
    #     print('clown 4')
    #     return factory(key_settings)
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is ok or if we should include
    # the 'exe_path' from settings.
    # 1) Check if we can find the module:
    # print('clown 5')
    if os.path.isfile(module):
        # print('clown 6')
        obj = import_from(module, klass)
    else:
        # print('clown 7')
        if 'exe_path' in settings['simulation']:
            module = os.path.join(settings['simulation']['exe_path'],
                                  module)
            obj = import_from(module, klass)
        else:
            msg = 'Could not find module "{}" for {}!'.format(module, key)
            raise ValueError(msg)
    # run some checks:
    # print('clown 8')
    for function in required_methods:
        objfunc = getattr(obj, function, None)
        if not objfunc:
            msg = 'Could not find method {}.{}'.format(klass,
                                                       function)
            logger.critical(msg)
            raise ValueError(msg)
        else:
            if not callable(objfunc):
                msg = 'Method {}.{} is not callable!'.format(klass,
                                                             function)
                logger.critical(msg)
                raise ValueError(msg)
    # return initiate_instance(obj, key_settings)
    return initiate_instance(obj, settings)

def import_from(module_path, function_name):
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
    try:
        module_name = os.path.basename(module_path)
        module_name = os.path.splitext(module_name)[0]
        spec = importlib.util.spec_from_file_location(module_name,
                                                      module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
        logger.debug('Imported module: %s', module)
        return getattr(module, function_name)
    except (ImportError, IOError):
        msg = f'Could not import module: {module_path}'
        logger.critical(msg)
    except AttributeError:
        msg = f'Could not import "{function_name}" from "{module_path}"'
        logger.critical(msg)
    raise ValueError(msg)

def make_dirs(dirname):
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
    try:
        os.makedirs(dirname)
        msg = f'Created directory: "{dirname}"'
    except OSError as err:
        if err.errno != errno.EEXIST:  # pragma: no cover
            raise err
        if os.path.isfile(dirname):
            msg = f'"{dirname}" is a file. Will abort!'
            raise OSError(errno.EEXIST, msg) from err
        if os.path.isdir(dirname):
            msg = f'Directory "{dirname}" already exist.'
    return msg

def write_ensemble_restart(ensemble, config, save):
    """Write a restart file for a path ensemble.

    Parameters
    ----------
    ensemble : dict
        it contains:

        * `path_ensemble` : object like :py:class:`.PathEnsemble`
          The path ensemble we are writing restart info for.
        * ` system` : object like :py:class:`.System`
          System is used here since we need access to the temperature
          and to the particle list.
        * `order_function` : object like :py:class:`.OrderParameter`
          The class used for calculating the order parameter(s).
        * `engine` : object like :py:class:`.EngineBase`
          The engine to use for propagating a path.

    settings_ens : dict
        A dictionary with the ensemble settings.

    """
    info = {}
    info['rgen'] = ensemble.rgen.get_state()

    filename = os.path.join(
        os.getcwd(),
        config['simulation']['load_dir'],
        save,
        'ensemble.restart')

    with open(filename, 'wb') as outfile:
        toprint = os.path.join(save, 'ensemble.restart')
        pickle.dump(info, outfile)
