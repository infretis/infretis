# -*- coding: utf-8 -*-
# Copyright (c) 2022, PyRETIS Development Team.
"""A GROMACS external MD integrator interface. Used for testing."""
import logging
import os
import pickle
from numpy.random import RandomState

# from pyretis.engines.gromacs import GromacsEngine
# from pyretis.engines.gromacs2 import GromacsEngine2

from infretis.newc.rgen import RandomGeneratorBorg
from infretis.newc.engines.gromacs import GromacsEngine

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


RND = RandomGeneratorBorg()
# Here, we set a specific random number state, by
# loading it from a file, if that file is present:
INPUTFILE = "pyretis_gmx_rnd.state"
# if os.path.isfile(INPUTFILE):
#     print('Loading state: {}'.format(INPUTFILE))
#     with open(INPUTFILE, 'rb') as inputf:
#         STATE = pickle.load(inputf)
#         RND.set_state(STATE)


def prepare_shooting_point(gro, input_file):
    """Create initial configuration for a shooting move.

    Parameters
    ----------
    gro : object like :py:class:`.GromacsEngine`
        The engine used for GROMACS.
    input_file : string
        The input configuration to generate velocities for.

    Returns
    -------
    output_file : string
        The name of the file created.
    energy : dict
        The energy terms read from the GROMACS .edr file.

    """
    gen_mdp = os.path.join(gro.exe_dir, "genvel.mdp")
    # Use specific seed:
    seed = 7644170
    settings = {
        "gen_vel": "yes",
        "gen_seed": seed,
        "nsteps": 0,
        "continuation": "no",
    }
    gro._modify_input(gro.input_files["input"], gen_mdp, settings, delim="=")
    # Run grompp for this input file:
    out_grompp = gro._execute_grompp(gen_mdp, input_file, "genvel")
    remove = [val for _, val in out_grompp.items()]
    # Run GROMACS for this tpr file:
    out_mdrun = gro._execute_mdrun(out_grompp["tpr"], "genvel")
    remove += [val for key, val in out_mdrun.items() if key != "conf"]
    confout = os.path.join(gro.exe_dir, out_mdrun["conf"])
    energy = gro.get_energies(out_mdrun["edr"])
    # remove run-files:
    logger.debug("Removing GROMACS output after velocity generation.")
    gro._remove_files(gro.exe_dir, remove)
    return confout, energy


class GromacsEngineR(GromacsEngine):
    """A class for interfacing GROMACS.

    This class uses a set of reproducible seeds for generation velocities.
    Otherwise, it is equal to :py:class:`.GromacsEngine`.

    """

    def __init__(
        self,
        gmx,
        mdrun,
        input_path,
        timestep,
        subcycles,
        maxwarn=0,
        gmx_format="g96",
        write_vel=True,
        write_force=False,
    ):
        """Set up the engine."""
        super().__init__(
            gmx,
            mdrun,
            input_path,
            timestep,
            subcycles,
            maxwarn=maxwarn,
            gmx_format=gmx_format,
            write_vel=write_vel,
            write_force=write_force,
        )

    def _prepare_shooting_point(self, input_file):
        """Create initial configuration for a shooting move."""
        return prepare_shooting_point(self, input_file)
