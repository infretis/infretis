"""random generator class."""
import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.random import RandomState

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


class RandomGeneratorBase(metaclass=ABCMeta):
    """A base class for random number generators.

    This is a base class for random number generators. It does not
    actually implement a generator.

    Attributes
    ----------
    seed : int
        A seed for the generator

    """

    def __init__(self, seed=0):
        """Initialise the random number generator.

        Parameters
        ----------
        seed : int, optional
            An integer used for seeding the generator if needed.

        """
        self.seed = seed

    @abstractmethod
    def rand(self, shape=1):
        """Draw random numbers in [0, 1).

        Parameters
        ----------
        shape : int, optional
            The number of numbers to draw

        Returns
        -------
        out : float
            Pseudo-random number in [0, 1)

        """
        return

    @abstractmethod
    def get_state(self):
        """Return info about the current state."""
        return

    @abstractmethod
    def set_state(self, state):
        """Set state for random generator."""
        return

    @abstractmethod
    def random_integers(self, low, high):
        """Draw random integers in [low, high].

        Parameters
        ----------
        low : int
            This is the lower limit
        high : int
            This is the upper limit

        Returns
        -------
        out : int
            The pseudo-random integers in [low, high].

        """
        return

    @abstractmethod
    def normal(self, loc=0.0, scale=1.0, size=None):
        """Return values from a normal distribution.

        Parameters
        ----------
        loc : float, optional
            The mean of the distribution
        scale : float, optional
            The standard deviation of the distribution
        size : int, tuple of ints, optional
            Output shape, i.e. how many values to generate. Default is
            None which is just a single value.

        Returns
        -------
        out : float, numpy.array of floats
            The random numbers generated.

        """
        return

    @abstractmethod
    def multivariate_normal(self, mean, cov, cho=None, size=1):
        """Draw numbers from a multi-variate distribution.

        Parameters
        ----------
        mean : numpy array (1D, 2)
            Mean of the N-dimensional array
        cov : numpy array (2D, (2, 2))
            Covariance matrix of the distribution.
        cho : numpy.array (2D, (2, 2)), optional
            Cholesky factorization of the covariance. If not given,
            it will be calculated here.
        size : int, optional
            The number of samples to do.

        Returns
        -------
        out : float or numpy.array of floats size
            The random numbers that are drawn here.

        """
        return

    def draw_maxwellian_velocities(self, vel, mass, beta, sigma_v=None):
        """Draw numbers from a Gaussian distribution.

        Parameters
        ----------
        system : object like :py:class:`.System`
            This is used to determine the shape (number of particles
            and dimensionality) and requires veloctities.
        engine : object like :py:class:`.Engine`
            This is used to determine the temperature parameter(s)
        sigma_v : numpy.array, optional
            The standard deviation in velocity, one for each particle.
            If it's not given it will be estimated.

        """
        if not sigma_v or sigma_v < 0.0:
            kbt = 1.0 / beta
            sigma_v = np.sqrt(kbt * (1 / mass))

        npart, dim = vel.shape
        vel = self.normal(loc=0.0, scale=sigma_v, size=(npart, dim))
        return vel, sigma_v


class RandomGenerator(RandomGeneratorBase):
    """A random number generator from numpy.

    This class that defines a random number generator. It will use
    `numpy.random.RandomState` for the actual generation and we refer
    to the numpy documentation [1]_.

    Attributes
    ----------
    seed : int
        A seed for the pseudo-random generator.
    rgen : object like :py:class:`numpy.random.RandomState`
        This is a container for the Mersenne Twister pseudo-random
        number generator as implemented in numpy [#]_.

    References
    ----------
    .. [#] The NumPy documentation on RandomState,
       http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html

    """

    def __init__(self, seed=0):
        """Initialise the random number generator.

        If a seed is given, the random number generator will be seeded.

        Parameters
        ----------
        seed : int, optional
            An integer used for seeding the generator if needed.

        """
        super().__init__(seed=seed)
        self.rgen = RandomState(seed=seed)

    def choice(self, a, size=None, replace=True, p=None):
        """Chooses random samples.

        Parameters
        ----------
        a : iterable, int
            The group to choose from, an int is converted to range(a).
        size: int, optional
            The number of samples to choose.
        replace: bool, optional
            If to replace samples between draws, default True.
        p : iterable, optional
            The probabilities for every option in a,
            default is an uniform dirstribution.

        Returns
        -------
        choice : array-like
            The picked choices.

        """
        return self.rgen.choice(a, size, replace, p)

    def rand(self, shape=1):
        """Draw random numbers in [0, 1).

        Parameters
        ----------
        shape : int, optional
            The number of numbers to draw

        Returns
        -------
        out : float
            Pseudo-random number in [0, 1)

        Note
        ----
        Here, we will just draw a list of numbers and not for
        an arbitrary shape.

        """
        return self.rgen.rand(shape)

    def get_state(self):
        """Return current state."""
        state = {
            "seed": self.seed,
            "state": self.rgen.get_state(),
            "rgen": "rgen",
        }
        return state

    def set_state(self, state):
        """Set state for random generator."""
        return self.rgen.set_state(state["state"])

    def random_integers(self, low, high):
        """Draw random integers in [low, high].

        Parameters
        ----------
        low : int
            This is the lower limit
        high : int
            This is the upper limit

        Returns
        -------
        out : int
            The pseudo-random integers in [low, high].

        Note
        ----
        np.random.randint(low, high) is defined as drawing
        from `low` (inclusive) to `high` (exclusive).

        """
        return self.rgen.randint(low, high + 1)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Return values from a normal distribution.

        Parameters
        ----------
        loc : float, optional
            The mean of the distribution
        scale : float, optional
            The standard deviation of the distribution
        size : int, tuple of ints, optional
            Output shape, i.e. how many values to generate. Default is
            None which is just a single value.

        Returns
        -------
        out : float, numpy.array of floats
            The random numbers generated.

        """
        return self.rgen.normal(loc=loc, scale=scale, size=size)

    def multivariate_normal(self, mean, cov, cho=None, size=1):
        """Draw numbers from a multi-variate distribution.

        This is an attempt on speeding up the call of
        `RandomState.multivariate_normal` if we need to call it over and
        over again. Such repeated calling will do an SVD repeatedly,
        which is wasteful. In this function, this transform can be
        supplied and it is only estimated if it's not explicitly given.

        Parameters
        ----------
        mean : numpy array (1D, 2)
            Mean of the N-dimensional array
        cov : numpy array (2D, (2, 2))
            Covariance matrix of the distribution.
        cho : numpy.array (2D, (2, 2)), optional
            Cholesky factorization of the covariance. If not given,
            it will be calculated here.
        size : int, optional
            The number of samples to do.

        Returns
        -------
        out : float or numpy.array of floats size
            The random numbers that are drawn here.

        See also
        --------
        numpy.random.multivariate_normal

        """
        if cho is None:
            cho = np.linalg.cholesky(cov)
        norm = self.normal(loc=0.0, scale=1.0, size=2 * size)
        norm = norm.reshape(size, 2)
        meanm = np.array(
            [
                mean,
            ]
            * size
        )
        return meanm + np.dot(norm, cho.T)


class MockRandomGenerator(RandomGeneratorBase):
    """A **mock** random generator, useful **only for testing**.

    This class represents a random generator that can be used for
    testing algorithms. It will simply return numbers from a
    small list of pseudo-random numbers. This class is only useful
    for testing algorithms on different systems. It should *NEVER*
    be used for actual production runs!
    """

    def __init__(self, seed=0, norm_shift=False):
        """Initialise the mock random number generator.

        Here, we set up some predefined random number which we will
        use as a pool for the generation.

        Parameters
        ----------
        seed : int, optional
            An integer used for seeding the generator if needed.
        norm_shift: boolean
            If True is will ensure that the fake 'normal distribution'
            is shifted to get the right mean.

        """
        super().__init__(seed=seed)
        self.rgen = [
            0.78008018,
            0.04459916,
            0.76596775,
            0.97676713,
            0.53799598,
            0.98657116,
            0.36343553,
            0.55356511,
            0.03172585,
            0.48984682,
            0.73416687,
            0.98453452,
            0.55129902,
            0.40598753,
            0.59448394,
            0.26823255,
            0.31168372,
            0.05072849,
            0.44876368,
            0.94301709,
        ]
        self.length = len(self.rgen)
        self.randint = self.seed
        self.norm_shift = norm_shift
        logger.critical('You are using a "mock" random generator!\n')
        if norm_shift:
            logger.critical("Fake-normal is shifted.\n")
            logger.critical("Comparison with TISMOL might fail\n")
        else:
            logger.critical("Fake-normal is not shifted.\n")
            logger.critical("Random numbers not centered around 0.\n")

    def rand(self, shape=1):
        """Draw random numbers in [0, 1).

        Parameters
        ----------
        shape : int
            The number of numbers to draw.

        Returns
        -------
        out : float
            Pseudo-random number in [0, 1).

        """
        numbers = []
        for _ in range(shape):
            if self.seed >= self.length:
                self.seed = 0
            numbers.append(self.rgen[self.seed])
            self.seed += 1
        return np.array(numbers)

    def get_state(self):
        """Return current state."""
        state = {"seed": self.seed, "state": self.seed, "rgen": "mock"}
        return state

    def set_state(self, state):
        """Set current state."""
        self.seed = state["state"]

    def random_integers(self, low, high):
        """Return random integers in [low, high].

        Parameters
        ----------
        low : int
            This is the lower limit
        high : int
            This is the upper limit

        Returns
        -------
        out : int
            This is a pseudo-random integer in [low, high].

        """
        idx = self.rand() * (high - low + 1)
        return int(idx[0]) + low

    def normal(self, loc=0.0, scale=1.0, size=None):
        """Return values from a normal distribution.

        Parameters
        ----------
        loc : float, optional
            The mean of the distribution
        scale : float, optional
            The standard deviation of the distribution
        size : int, tuple of ints, optional
            Output shape, i.e. how many values to generate. Default is
            None which is just a single value.

        Returns
        -------
        out : float, numpy.array of floats
            The random numbers generated.

        Note
        ----
        This is part of the Mock-random number generator. Hence, it
        won't provide a true normal distribution, though the mean is set to
        the value of loc.

        """
        if self.norm_shift:
            shift = loc - 0.5
        else:
            shift = 0.0
        if size is None:
            return self.rand(shape=1) + shift
        numbers = np.zeros(size)
        for i in np.nditer(numbers, op_flags=["readwrite"]):
            i[...] = self.rand(shape=1)[0] + shift
        return numbers

    def multivariate_normal(self, mean, cov, cho=None, size=1):
        """Draw numbers from a multi-variate distribution.

        This is an attempt on speeding up the call of
        `RandomState.multivariate_normal` if we need to call it over and
        over again. Such repeated calling will do an SVD repeatedly,
        which is wasteful. In this function, this transform can be
        supplied and it is only estimated if it's not explicitly given.

        Parameters
        ----------
        mean : numpy array (1D, 2)
            Mean of the N-dimensional array
        cov : numpy array (2D, (2, 2))
            Covariance matrix of the distribution.
        cho : numpy.array (2D, (2, 2)), optional
            Cholesky factorization of the covariance. If not given,
            it will be calculated here.
        size : int, optional
            The number of samples to do.

        Returns
        -------
        out : float or numpy.array of floats size
            The random numbers that are drawn here.

        See also
        --------
        numpy.random.multivariate_normal

        """
        norm = self.normal(loc=0.0, scale=1.0, size=2 * size)
        norm = norm.reshape(size, 2)
        meanm = np.array(
            [
                mean,
            ]
            * size
        )
        return 0.01 * (meanm + norm)

    def choice(self, a, size=None, replace=True, p=None):
        """Choose random samples.

        Parameters
        ----------
        a : iterable
            The group to choose from.
        size: int, optional
            The number of samples to choose.
        replace: bool, optional
            If to replace samples between draws, default True.
        p : iterable, optional
            The probabilities for every option in a,
            default is an uniform dirstribution.

        Returns
        -------
        choice : array-like
            The picked choices.

        """
        if isinstance(a, int):
            a = list(range(a))
        if p is None:
            p = [1 / len(a) for i in a]
        if size is None:
            size = 1
        out = []
        for _ in range(size):
            r = self.rand()
            p0 = 0
            # Make sure p sums to 1 even after popping
            p_sum = sum(p)
            p = [i / p_sum for i in p]
            for i, pi in zip(a, p):
                p0 += pi
                if r < p0:
                    out.append(i)
                    if not replace:
                        idx = a.index(i)
                        _ = a.pop(idx)
                        _ = p.pop(idx)
                    break
        return out


class Borg:
    """A class for sharing states of objects."""

    class_state = None
    number_of_borgs = 0

    @classmethod
    def update_state(cls, state):
        """Update the class state and enable sharing of it."""
        if cls.class_state is None:
            logger.debug("Setting state to the shared.")
            cls.class_state = state
        else:
            logger.debug("Reusing shared state.")
        return cls.class_state

    @classmethod
    def reset_state(cls):
        """Remove memory of the shared state."""
        cls.class_state = None

    @classmethod
    def make_new_swarm(cls):
        """Make new swarm (new class)."""
        name = cls.__name__ + str(cls.number_of_borgs)
        cls.number_of_borgs += 1
        new_borg = type(name, cls.__bases__, dict(cls.__dict__))
        new_borg.reset_state()  # pylint: disable=no-member
        return new_borg

    def __init__(self, *args, **kwargs):
        """Enable share of the state.

        This will update the Borg.class_state and
        update self.__dict__ to this object, which enables
        sharing of the state.

        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self.update_state(self.__dict__)


class RandomGeneratorBorg(Borg, RandomGenerator):
    """A class for sharing the state between RandomGenerator objects."""


class MockRandomGeneratorBorg(Borg, MockRandomGenerator):
    """A class for sharing the state between MockRandomGenerator objects."""


def create_random_generator(settings=None):
    """Create a random generator from given settings.

    Parameters
    ----------
    settings : dict, optional
        This is the dict used for creating the random generator.
        Currently, we will actually just look for a seed value.

    Returns
    -------
    out : object like :py:class:`.RandomGenerator`
        The random generator created.

    """
    if settings is None:
        settings = {}

    rgen = settings.get("rgen", "rgen")
    seed = settings.get("seed", 0)
    logger.debug("Seed for random generator: %s %d", rgen, seed)

    class_map = {
        "rgen": RandomGenerator,
        "rgen-borg": RandomGeneratorBorg,
        "mock": MockRandomGenerator,
        "mock-borg": MockRandomGeneratorBorg,
    }
    rgen_class = class_map.get(rgen, RandomGenerator)
    rgen = rgen_class(seed=seed)

    if "state" in settings:
        rgen.set_state(settings)
        rgen.status = "restarted"
    else:
        rgen.status = "new rgen"

    return rgen
