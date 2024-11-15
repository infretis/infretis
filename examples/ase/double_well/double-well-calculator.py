import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def double_well_potential(r, a, b, c):
    return a * r**4 - b * (r - c) ** 2


def double_well_force(r, a, b, c):
    return -4 * a * r**3 + 2 * b * (r - c)


def forces_and_energy_DW(pos, N, a, b, c):
    pe = 0.0
    frc = np.zeros((N, 3))
    for i in range(N):
        x = pos[i, 0]
        pe += double_well_potential(x, a, b, c)
        frc[i, 0] = double_well_force(x, a, b, c)

    return pe, frc


class DWCalc(Calculator):
    def __init__(self, a, b, c):
        super().__init__()
        self.implemented_properties = ["energy", "forces", "stress"]
        self.a = a
        self.b = b
        self.c = c
        print(a, b, c)

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        N = int(atoms.positions.shape[0])

        e, f = forces_and_energy_DW(atoms.positions, N, self.a, self.b, self.c)

        self.results = {"energy": e, "forces": f, "stress": np.zeros(6)}
