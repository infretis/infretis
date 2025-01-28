"""Engine factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, Dict

import importlib

if importlib.util.find_spec("scm") is not None:
    if importlib.util.find_spec("scm.plams") is not None:
        from infretis.classes.engines.ams import AMSEngine
from infretis.classes.engines.ase_engine import ASEEngine
from infretis.classes.engines.cp2k import CP2KEngine
from infretis.classes.engines.gromacs import GromacsEngine
from infretis.classes.engines.lammps import LAMMPSEngine
from infretis.classes.engines.turtlemdengine import TurtleMDEngine
from infretis.core.core import create_external, generic_factory

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.engines.enginebase import EngineBase


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_engine(
    settings: Dict[str, Any], eng_key: str = "engine"
) -> Optional[EngineBase]:
    """Create an engine from settings.

    Args:
        settings: Settings for the simulation. This method will
            use the `"engine"` section of the settings.
        eng_key: The key to the specific engine.

    Returns:
        The engine created here.
    """
    engine_map = {
        "gromacs": {"class": GromacsEngine},
        "cp2k": {"class": CP2KEngine},
        "turtlemd": {"class": TurtleMDEngine},
        "lammps": {"class": LAMMPSEngine},
        "ase": {"class": ASEEngine},
    }
    if importlib.util.find_spec("scm") is not None:
        if importlib.util.find_spec("scm.plams") is not None:
            engine_map["ams"] = {"class": AMSEngine}

    if settings[eng_key]["class"].lower() not in engine_map:
        return create_external(settings[eng_key], "engine", ["step"])
    engine = generic_factory(settings[eng_key], engine_map, name="engine")
    return engine


def create_engines(
    config: Dict[str, Any]
) -> Tuple[Dict[Any, EngineBase], Dict[Any, int]]:
    """Create the engines for a infretis simulation.

    We create min(n_engines_type_i, n_workers) engines in a dict
    'engines'. Each entry of engines['enginei']
    has elements:

        [   enginei0,    enginei1, enginei2]

    and each entry of 'engine_occ' has entries

         [workeri_pin, workerj_pin,       -1],

    where enginei0 is used by workeri, enginei1 by workerj, and
    enginei2 is free (-1 tells us the engine is not used).

    Args:
        config: The config dict that is setup from the .toml file.

    Returns:
        engines: A dictionary containing lists of engine instances
        engine_occ: A dictionary containing lists of worker pins

    """
    engine_count: dict = {}
    engines: dict = {}
    engine_occ: dict = {}
    # get all unique engines with number of occurences
    for engine in config["simulation"]["ensemble_engines"]:
        for engine_i in engine:
            engine_count[engine_i] = engine_count.get(engine_i, 0) + 1

    for engine, n_engine in engine_count.items():
        engines[engine] = []
        engine_occ[engine] = []
        n_create = min(n_engine, config["runner"]["workers"])
        for i in range(n_create):
            check_engine(config, eng_key=engine)
            engine_occ[engine].append(-1)
            engines[engine].append(create_engine(config, eng_key=engine))

    return engines, engine_occ


def check_engine(settings: Dict[str, Any], eng_key: str) -> bool:
    """Check the input settings for engine creation.

    Args:
        settings: The input settings to use for creating the engine.
        eng_key: The key to the specific engine.
    """
    msg = []
    if eng_key not in settings:
        msg += [f"The section [{eng_key}] is missing"]

    if "gmx" in settings[eng_key] and "gmx_format" not in settings[eng_key]:
        msg += ["File format is not specified for the engine"]

    elif (
        "cp2k" in settings[eng_key] and "cp2k_format" not in settings[eng_key]
    ):
        msg += ["File format is not specified for the engine"]

    if msg:
        msgtxt = "\n".join(msg)
        logger.critical(msgtxt)
        return False

    return True


def assign_engines(
    engine_occ: Dict[str, list], eng_names, pin
) -> Dict[Any, int]:
    """Assign non-occupied engine(s) to a worker based on the engine_occ dict.

    Args:
        engine_occ: The dict containing engine_occupations.
        eng_names: The engine names to get.
        pin: The worker pin.

    Returns:
        out: A dict containing a pointer to an engine instance of type
            eng_names[i].
    """
    # first free all engines that where occupied by worker i
    for eng_key in engine_occ.keys():
        for i, occupied_by in enumerate(engine_occ[eng_key]):
            if pin == occupied_by:
                engine_occ[eng_key][i] = -1

    # then get non-occupied engines of type 'eng_names'
    out = {}
    for eng_key in eng_names:
        for i, occupied_by in enumerate(engine_occ[eng_key]):
            if occupied_by == -1:
                engine_occ[eng_key][i] = pin
                out[eng_key] = i
                # exit inner loop when we find a non-occupied engine
                break

            if i == len(engine_occ[eng_key]):
                # we should never reach this:
                msg = (
                    f"All engines '{eng_key}' are occupied."
                    + "This should not happen!"
                )
                raise ValueError(msg)
    if out == {}:
        msg = "Did not find a free engine, this should not happen!"
        raise ValueError(msg)

    return out
