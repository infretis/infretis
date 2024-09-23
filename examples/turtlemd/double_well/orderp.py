from infcore.core.orderparameter import OrderParameter


class PositionX(OrderParameter):
    """Position order parameter.


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

    def __init__(self, index: tuple[int, int], periodic: bool = True):
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
        pbc = "Periodic" if periodic else "Non-periodic"
        txt = f"{pbc} distance, particles {index[0]} and {index[1]}"
        super().__init__(description=txt, velocity=False)
        self.periodic = periodic
        if periodic:
            raise NotImplementedError("Can't use pbc for position order yet")
        self.index = index

    def calculate(self, system) -> list[float]:
        """Calculate the order parameter.

        Here, the order parameter is just the position
        of a particle.

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
        pos = system.pos[self.index[0], self.index[1]]
        return [pos]


def _verify_pair(index: tuple[int, int]):
    """Check that the given index contains a pair."""
    try:
        if len(index) != 2:
            msg = (
                "Wrong number of atoms for pair definition. "
                f"Expected 2 got {len(index)}"
            )
            raise ValueError(msg)
    except TypeError as err:
        msg = "Atom pair should be defined as a tuple/list of integers."
        raise TypeError(msg) from err
