from infretis.newf.core import initiate_instance, generic_factory

from abc import ABCMeta, abstractmethod
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class OrderParameter:
    """Base class for order parameters.

    This class represents an order parameter and other collective
    variables. The order parameter is assumed to be a function
    that can uniquely be determined by the system object and its
    attributes.

    Attributes
    ----------
    description : string
        This is a short description of the order parameter.
    velocity_dependent : boolean
        This flag indicates whether or not the order parameter
        depends on the velocity direction. If so, we need to
        recalculate the order parameter when reversing trajectories.

    """

    def __init__(self, description='Generic order parameter', velocity=False):
        """Initialise the OrderParameter object.

        Parameters
        ----------
        description : string
            Short description of the order parameter.

        """
        self.description = description
        self.velocity_dependent = velocity
        if self.velocity_dependent:
            logger.debug(
                'Order parameter "%s" was marked as velocity dependent.',
                self.description
            )

    @abstractmethod
    def calculate(self, system):
        """Calculate the main order parameter and return it.

        All order parameters should implement this method as
        this ensures that the order parameter can be calculated.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This object contains the information needed to calculate
            the order parameter.

        Returns
        -------
        out : list of floats
            The order parameter(s). The first order parameter returned
            is used as the progress coordinate in path sampling
            simulations!

        """
        return

    def __str__(self):
        """Return a simple string representation of the order parameter."""
        msg = [
            f'Order parameter: "{self.__class__.__name__}"',
            f'{self.description}'
        ]
        if self.velocity_dependent:
            msg.append('This order parameter is velocity dependent.')
        return '\n'.join(msg)

    def load_restart_info(self, info):
        """Load the orderparameter restart info."""

    def restart_info(self):
        """Save any mutatable parameters for the restart."""

class Distancevel(OrderParameter):
    """A rate of change of the distance order parameter.

    This class defines a very simple order parameter which is just
    the time derivative of the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index, periodic=True):
        """Initialise the order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = 'Periodic' if periodic else 'Non-periodic'
        txt = (f'{pbc} rate-of-change-distance, particles {index[0]} and '
               f'{index[1]}')
        super().__init__(description=txt, velocity=True)
        self.periodic = periodic
        self.index = index

    def calculate(self, system):
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The rate-of-change of the distance order parameter.

        """
        particles = system.particles
        delta = particles.pos[self.index[1]] - particles.pos[self.index[0]]
        if self.periodic:
            delta = system.box.pbc_dist_coordinate(delta)
        lamb = np.sqrt(np.dot(delta, delta))
        # Add the velocity as an additional collective variable:
        delta_v = particles.vel[self.index[1]] - particles.vel[self.index[0]]
        cv1 = np.dot(delta, delta_v) / lamb
        return [cv1]

class Position(OrderParameter):
    """A positional order parameter.

    This class defines a very simple order parameter which is just
    the position of a given particle.

    Attributes
    ----------
    index : integer
        This is the index of the atom which will be used, i.e.
        ``system.particles.pos[index]`` will be used.
    dim : integer
        This is the dimension of the coordinate to use.
        0, 1 or 2 for 'x', 'y' or 'z'.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the position or not.

    """

    def __init__(self, index, dim='x', periodic=False, description=None):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the position of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        if description is None:
            description = f'Position of particle {index} (dim: {dim})'
        super().__init__(description=description, velocity=False)
        self.periodic = periodic
        self.index = index
        self.dim = {'x': 0, 'y': 1, 'z': 2}.get(dim, None)
        if self.dim is None:
            msg = f'Unknown dimension {dim} requested'
            logger.critical(msg)
            raise ValueError(msg)

    def calculate(self, system):
        """Calculate the position order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions.

        Returns
        -------
        out : list of floats
            The position order parameter.

        """
        particles = system.particles
        pos = particles.pos[self.index]
        lamb = pos[self.dim]
        if self.periodic:
            lamb = system.box.pbc_coordinate_dim(lamb, self.dim)
        return [lamb]

class Distance(OrderParameter):
    """A distance order parameter.

    This class defines a very simple order parameter which is just
    the scalar distance between two particles.

    Attributes
    ----------
    index : tuple of integers
        These are the indices used for the two particles.
        `system.particles.pos[index[0]]` and
        `system.particles.pos[index[1]]` will be used.
    periodic : boolean
        This determines if periodic boundaries should be applied to
        the distance or not.

    """

    def __init__(self, index, periodic=True):
        """Initialise order parameter.

        Parameters
        ----------
        index : tuple of ints
            This is the indices of the atom we will use the position of.
        periodic : boolean, optional
            This determines if periodic boundary conditions should be
            applied to the position.

        """
        _verify_pair(index)
        pbc = 'Periodic' if periodic else 'Non-periodic'
        txt = f'{pbc} distance, particles {index[0]} and {index[1]}'
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        self.index = index

    def calculate(self, system):
        """Calculate the order parameter.

        Here, the order parameter is just the distance between two
        particles.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the positions and box used for the
            calculation.

        Returns
        -------
        out : list of floats
            The distance order parameter.

        """
        particles = system.particles
        delta = particles.pos[self.index[1]] - particles.pos[self.index[0]]
        if self.periodic:
            delta = system.box.pbc_dist_coordinate(delta)
        lamb = np.sqrt(np.dot(delta, delta))
        return [lamb]

class Velocity(OrderParameter):
    """Initialise the order parameter.

    This class defines a very simple order parameter which is just
    the velocity of a given particle.

    Attributes
    ----------
    index : integer
        This is the index of the atom which will be used, i.e.
        ``system.particles.vel[index]`` will be used.
    dim : integer
        This is the dimension of the coordinate to use.
        0, 1 or 2 for 'x', 'y' or 'z'.

    """

    def __init__(self, index, dim='x'):
        """Initialise the order parameter.

        Parameters
        ----------
        index : int
            This is the index of the atom we will use the velocity of.
        dim : string
            This select what dimension we should consider,
            it should equal 'x', 'y' or 'z'.

        """
        txt = f'Velocity of particle {index} (dim: {dim})'
        super().__init__(description=txt, velocity=True)
        self.index = index
        self.dim = {'x': 0, 'y': 1, 'z': 2}.get(dim, None)
        if self.dim is None:
            logger.critical('Unknown dimension %s requested', dim)
            raise ValueError

    def calculate(self, system):
        """Calculate the velocity order parameter.

        Parameters
        ----------
        system : object like :py:class:`.System`
            The object containing the velocities.

        Returns
        -------
        out : list of floats
            The velocity order parameter.

        """
        return [system.particles.vel[self.index][self.dim]]


def create_orderparameters(engines, settings):
    for engine in engines.keys():
        engines[engine].order_function = create_orderparameter(settings)

def create_orderparameter(settings):
    """Create order parameters from settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    out : object like :py:class:`.OrderParameter`
        This object represents the order parameter.

    """
    main_order = create_external(
        settings,
        'orderparameter',
        order_factory,
        ['calculate'],
    )
    if main_order is None:
        logger.info('No order parameter created')
        return None
    logger.info('Created main order parameter:\n%s', main_order)

    extra_cv = []
    order_settings = settings.get('collective-variable', [])
    for order_setting in order_settings:
        order = create_external(
            settings,
            'collective-variable',
            order_factory,
            ['calculate'],
            key_settings=order_setting
        )
        logger.info('Created additional collective variable:\n%s', order)
        extra_cv.append(order)
    if not extra_cv:
        return main_order
    all_order = [main_order] + extra_cv
    order = CompositeOrderParameter(order_parameters=all_order)
    logger.info('Composite order parameter:\n%s', order)
    return order

def create_external(settings, key, factory, required_methods,
                    key_settings=None):
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
    if key_settings is None:
        try:
            key_settings = settings[key]
        except KeyError:
            logger.debug('No "%s" setting found. Skipping set-up', key)
            return None
    module = key_settings.get('module', None)
    klass = None
    try:
        klass = key_settings['class']
    except KeyError:
        logger.debug('No "class" setting for "%s" specified. Skipping set-up',
                     key)
        return None
    if module is None:
        return factory(key_settings)
    # Here we assume we are to load from a file. Before we import
    # we need to check that the path is ok or if we should include
    # the 'exe_path' from settings.
    # 1) Check if we can find the module:
    if os.path.isfile(module):
        obj = import_from(module, klass)
    else:
        if 'exe_path' in settings['simulation']:
            module = os.path.join(settings['simulation']['exe_path'],
                                  module)
            obj = import_from(module, klass)
        else:
            msg = 'Could not find module "{}" for {}!'.format(module, key)
            raise ValueError(msg)
    # run some checks:
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
    return initiate_instance(obj, key_settings)

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

def order_factory(settings):
    """Create order parameters according to the given settings.

    This function is included as a convenient way of setting up and
    selecting the order parameter.

    Parameters
    ----------
    settings : dict
        This defines how we set up and select the order parameter.

    Returns
    -------
    out : object like :py:class:`.OrderParameter`
        An object representing the order parameter.

    """
    factory_map = {
        'orderparameter': {
            'cls': OrderParameter
        },
        'position': {
            'cls': Position
        },
        'velocity': {
            'cls': Velocity
        },
        'distance': {
            'cls': Distance
        },
        'distancevel': {
            'cls': Distancevel
        },
    }
    return generic_factory(settings, factory_map, name='orderparameter')
