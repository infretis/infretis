from copy import copy as copy0
import numpy as np

class Particles:
    """Base class for a collection of particles.

    This is a simple particle list. It stores the positions,
    velocities, forces, masses (and inverse masses) and type
    information for a set of particles. This class also defines a
    method for iterating over pairs of particles, which could be
    useful for implementing neighbour lists. In this particular
    class, this method will just define an all-pairs list.

    Attributes
    ----------
    npart : integer
        Number of particles.
    pos : numpy.array
        Positions of the particles.
    vel : numpy.array
        Velocities of the particles.
    force : numpy.array
        Forces on the particles.
    virial : numpy.array
        The current virial for the particles.
    mass : numpy.array
        Masses of the particles.
    imass : numpy.array
        Inverse masses, `1.0 / self.mass`.
    name : list of strings
        A name for the particle. This may be used as short text
        describing the particle.
    ptype : numpy.array of integers
        A type for the particle. Particles with identical `ptype` are
        of the same kind.
    dim : int
        This variable is the dimensionality of the particle list. This
        should be derived from the box. For some functions, it is
        convenient to be able to access the dimensionality directly
        from the particle list. It is therefore set as an attribute
        here.
    vpot : float
        The potential energy of the particles.
    ekin : float
        The kinetic energy of the particles.

    """

    particle_type = 'external'

    # Attributes to store when restarting/copying:
    _copy_attr = {'npart', 'name', 'ptype', 'dim',
                  'config', 'vel_rev'}
    # Attributes which are numpy arrays:
    _numpy_attr = {'pos', 'vel', 'force', 'virial', 'mass', 'imass',
                   'ptype', 'ekin', 'vpot'}

    def __init__(self, dim=1):
        """Initialise the Particle list.

        Here we just create an empty particle list.

        Parameters
        ----------
        dim : integer, optional
            The number of dimensions we are considering for positions,
            velocities and forces.

        """
        self.npart = 0
        self.pos = None
        self.vel = None
        self.vpot = None
        self.ekin = None
        self.force = None
        self.mass = None
        self.imass = None
        self.name = []
        self.ptype = None
        self.virial = None
        self.dim = dim
        self.config = (None, None)
        self.vel_rev = False

    def empty_list(self):
        """Reset the particle list.

        This will delete all particles in the list and set other
        variables to `None`.

        Note
        ----
        This is almost `self.__init__` repeated. The reason for this is
        simply that we want to define all attributes in `self.__init__`
        and not get any 'surprise attributes' defined elsewhere.
        Also, note that the dimensionality (`self.dim`) is not changed
        in this method.

        """
        self.npart = 0
        self.pos = None
        self.vpot = None
        self.ekin = None
        self.vel = None
        self.force = None
        self.mass = None
        self.imass = None
        self.name = []
        self.ptype = None
        self.virial = None
        self.config = (None, None)
        self.vel_rev = False

    def _copy_attribute(self, attr, copy_function):
        """Copy an attribute.

        Parameters
        ----------
        attr : string
            The attribute to copy.
        copy_function : callable
            The method to use for copying the attribute.

        Returns
        -------
        out : object
            A copy of the selected attribute.

        """
        val = getattr(self, attr, None)
        if val is None:
            return None
        return copy_function(val)

    def copy(self):
        """Return a copy of the particle state.

        Returns
        -------
        out : object like :py:class:`.Particles`
            A copy of the current state of the particles.

        """
        particles_copy = self.__class__(dim=self.dim)
        for attr in self._copy_attr:
            copy_attr = self._copy_attribute(attr, copy0)
            setattr(particles_copy, attr, copy_attr)
        for attr in self._numpy_attr:
            copy_attr = self._copy_attribute(attr, np.copy)
            setattr(particles_copy, attr, copy_attr)
        return particles_copy

    def __eq__(self, other):
        """Compare two particle states."""
        attrs = self._copy_attr.union(self._numpy_attr)
        return compare_objects(self, other, attrs,
                               numpy_attrs=self._numpy_attr)

    def set_pos(self, pos):
        """Set positions for the particles.

        This will copy the input positions, for this class, the
        input positions are assumed to be a file name with a
        corresponding integer which determines the index for the
        positions in the file for cases where the file contains
        several snapshots.

        Parameters
        ----------
        pos : tuple of (string, int)
            The positions to set, this represents the file name and the
            index for the frame in this file.

        """
        self.config = (pos[0], pos[1])

    def get_pos(self):
        """Just return the positions of the particles."""
        return self.config

    def set_vel(self, rev_vel):
        """Set velocities for the particles.

        Here we store information which tells if the
        velocities should be reversed or not.

        Parameters
        ----------
        rev_vel : boolean
            The velocities to set. If True, the velocities should
            be reversed before used.

        """
        self.vel_rev = rev_vel

    def get_vel(self):
        """Return info about the velocities."""
        return self.vel_rev

    def set_force(self, force):
        """Set the forces for the particles.

        Parameters
        ----------
        force : numpy.array
            The forces to set.

        """
        self.force = np.copy(force)

    def get_force(self):
        """Return (a copy of) the forces."""
        return np.copy(self.force)

    def add_particle(self, pos, vel, force, mass=1.0,
                     name='?', ptype=0):
        """Add a particle to the system.

        Parameters
        ----------
        pos : tuple
            Positions of new particle.
        vel : boolean
            Velocities of new particle.
        force : tuple
            Forces on the new particle.
        mass : float, optional
            The mass of the particle.
        name : string, optional
            The name of the particle.
        ptype : integer, optional
            The particle type.

        """
        self.name = [name]
        self.ptype = np.array(ptype, dtype=np.int16)
        self.pos = None
        self.set_pos(pos)
        self.vel = None
        self.set_vel(vel)
        self.force = force
        self.mass = np.zeros((1, 1))  # Column matrix.
        self.mass[0] = mass
        self.imass = 1.0 / self.mass
        self.npart = 1

    def get_selection(self, properties, selection=None):
        """Return selected properties for a selection of particles.

        Parameters
        ----------
        properties : list of strings
            The strings represent the properties to return.
        selection : list with indices to return, optional
            If a selection is not given, data for all particles
            are returned.

        Returns
        -------
        out : list
            A list with the properties in the order they were asked for
            in the ``properties`` argument.

        """
        sel_prop = []
        for prop in properties:
            if hasattr(self, prop):
                var = getattr(self, prop)
                if isinstance(var, list):
                    if selection is None:
                        sel_prop.append(var)
                    else:
                        sel_prop.append([var[i] for i in selection])
                else:
                    if selection is None:
                        sel_prop.append(var)
                    else:
                        sel_prop.append(var[selection])
        return sel_prop

    def __iter__(self):
        """Iterate over the particles.

        This function will yield the properties of the different
        particles.

        Yields
        ------
        out : dict
            The information in `self.pos`, `self.vel`, ... etc.

        """
        for i, pos in enumerate(self.pos):
            part = {'pos': pos, 'vel': self.vel[i], 'force': self.force[i],
                    'mass': self.mass[i], 'imass': self.imass[i],
                    'name': self.name[i], 'ptype': self.ptype[i]}
            yield part

    def pairs(self):
        """Iterate over all pairs of particles.

        Yields
        ------
        out[0] : integer
            The index for the first particle in the pair.
        out[1] : integer
            The index for the second particle in the pair.
        out[2] : integer
            The particle type of the first particle.
        out[3] : integer
            The particle type of the second particle.

        """
        for i, itype in enumerate(self.ptype[:-1]):
            for j, jtype in enumerate(self.ptype[i+1:]):
                yield (i, i+1+j, itype, jtype)

    def __str__(self):
        """Print out basic info about the particle list."""
        return 'Config: {}\nReverse velocities: {}'.format(
            self.config, self.vel_rev
        )

    def restart_info(self):
        """Generate information for saving a restart file."""
        info = {'type': self.particle_type}
        for copy_list in (self._copy_attr, self._numpy_attr):
            for attr in copy_list:
                try:
                    info[attr] = getattr(self, attr)
                except AttributeError:
                    logger.warning(('Missing attribute "%s" when creating'
                                    ' restart information.'), attr)
                    info[attr] = None
        return info

    def load_restart_info(self, info):
        """Load restart information.

        Parameters
        ----------
        info : dict
            Dictionary with the settings to load.

        """
        for attr in self._copy_attr.union(self._numpy_attr):
            if attr in info:
                setattr(self, attr, info[attr])
            else:
                msg = ('Could not set "{}" for particles '
                       'from restart info').format(attr)
                logger.error(msg)
                raise ValueError(msg)

    def reverse_velocities(self):
        """Reverse the velocities in the system."""
        self.vel = self.vel * -1

def particles_from_restart(restart):
    """Create particles from restart information.

    Parameters
    ----------
    restart : dict
        The restart settings.

    Returns
    -------
    particles : object like :py:class:`.Particles.`
        The object created from the restart information.

    """
    restart_particles = restart.get('particles', None)
    if restart_particles is None:
        logger.info('No particles were created from restart information.')
        return None
    particles = Particles(dim=restart_particles['dim'])
    particles.load_restart_info(restart_particles)
    return particles

