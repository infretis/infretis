"""Engine factory."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    settings: dict[str, Any], eng_key: str = "engine"
) -> EngineBase | None:
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
    }

    if settings[eng_key]["class"].lower() not in engine_map:
        return create_external(settings[eng_key], "engine", ["step"])
    engine = generic_factory(settings[eng_key], engine_map, name="engine")
    return engine


def create_engines(config: dict[str, Any]) -> dict[Any, EngineBase | None]:
    """Create the engines for a infretis simulation.

    We create min(n_engines_type_i, n_workers) engines in a dict
    'engines'. Each entry of engines['enginei']
    has elements:
        [
         [workeri_pin, workerj_pin,       -1],
         [   enginei0,    enginei1, enginei2]
        ]

    Where enginei0 is used by workeri, enginei1 by workerj, and
    enginei2 is free (-1 tells us the engine is not used).

    Args:
        config: The config dict that is setup from the .toml file.

    Returns:
        engines: A dictionary with a nested list of worker pins, and engines.

    """
    engine_count: dict = {}
    engines: dict = {}
    # get all unique engines with number of occurences
    for engine in config["simulation"]["ensemble_engines"]:
        for engine_i in engine:
            engine_count[engine_i] = engine_count.get(engine_i, 0) + 1

    for engine, n_engine in engine_count.items():
        engines[engine] = [[], []]
        n_create = min(n_engine, config["runner"]["workers"])
        for i in range(n_create):
            check_engine(config, eng_key=engine)
            engines[engine][0].append(-1)
            engines[engine][1].append(create_engine(config, eng_key=engine))

    return engines


def check_engine(settings: dict[str, Any], eng_key: str) -> bool:
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


def get_engines(engines: dict[str, list], eng_names, pin) -> list:
    """Get non-occupied engine(s) from the engine dict.

    Args:
        engines: The dict containing engine_occupations and engine instances.
        eng_names: The engine names to get.
        pin: The worker pin.

    Returns:
        out: A list of engine instances of type eng_names.
    """
    # first free all engines that where occupied by worker i
    for eng_key in engines.keys():
        for i, occupied_by in enumerate(engines[eng_key][0]):
            if pin == occupied_by:
                engines[eng_key][0][i] = -1

    # then get non-occupied engines of type 'eng_names'
    out = []
    for eng_key in eng_names:
        for i, (occupied_by, engine) in enumerate(zip(*engines[eng_key])):
            if occupied_by == -1:
                engines[eng_key][0][i] = pin
                out.append(engine)
                # exit inner loop when we find a non-occupied engine
                break

            if i == len(engines[eng_key][0]):
                # we should never reach this:
                msg = (
                    f"All engines '{eng_key}' are occupied."
                    + "This should not happen!"
                )
                raise ValueError(msg)

    return out
