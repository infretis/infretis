"""An ase calculator for the H2 system."""
from ase.calculators.lj import LennardJones

class LennardJonesCalc(LennardJones):
    """Class for the LJ calculator."""

    def __init__(self, sigma, epsilon, rc, smooth):
        """Initialize LJ calculator."""
        super().__init__(sigma = sigma, epsilon = epsilon, rc = rc, smooth = False)

