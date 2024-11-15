from ase.calculators.lj import LennardJones

class LennardJonesCalc(LennardJones):
    def __init__(self, sigma, epsilon, rc, smooth):
        super().__init__(sigma = sigma, epsilon = epsilon, rc = rc, smooth = False)

