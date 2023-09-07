import numpy as np

class MockRandomGenerator():
    """A **mock** random generator, useful **only for testing**.

    This class represents a random generator that can be used for
    testing algorithms. It will simply return numbers from a
    small list of pseudo-random numbers. This class is only useful
    for testing algorithms on different systems. It should *NEVER*
    be used for actual production runs!
    """

    def __init__(self, seed=0):
        """Initialise the mock random number generator.

        Here, we set up some predefined random number which we will
        use as a pool for the generation.

        Parameters
        ----------
        seed : int, optional
            An integer used for seeding the generator if needed.
        """
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
        self.seed = seed
        self.length = len(self.rgen)

    def random(self, shape=1):
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
        if shape == 1:
            return numbers[0]
        return np.array(numbers)

    def integers(self, low, high):
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
        idx = self.random() * (high - low + 1)
        return int(idx) + low

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
        if size is None:
            return self.random(shape=1)
        numbers = np.zeros(size)
        for i in np.nditer(numbers, op_flags=["readwrite"]):
            i[...] = self.random(shape=1)
        return numbers

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
            r = self.random()
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
ape =  MockRandomGenerator(2)
print(ape.random())
