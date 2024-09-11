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
    """Create the engines for a infretis simulation. The way we create these
    engines depends on the settings in config.

    For normal infretis, create config['runner']['workers'] identical engines.
    With quantis, we create one additional engine in the [0-] ensemble.

    Any other methods requiring multiple engines should set them here. The
    returned engines are stored as a global variable ENGINES, and which to use
    at what point may depend on the MC move. Therefore, new methods should use
    keys in the 'engines' dictionary that make sense for the method.

    Args:
        config: The config dict that is setup from the .toml file.

    Returns:
        engines: A dictionary of engines.

    """
    engines = {}
    eng_key = "engine"
    if config["simulation"]["tis_set"]["quantis"]:
        check_engine(config, eng_key="engine0")
        engine0 = create_engine(config, eng_key="engine0")
        logger.info(f"Created engine '{engine0}' from settings.")
        engines[-1] = engine0
        eng_key = "engine1"

    if not config["simulation"]["tis_set"]["multi_engine"]:
        check_engine(config, eng_key=eng_key)
        for i in range(config["runner"]["workers"]):
            engine = create_engine(config, eng_key)
            logger.info(f"Created engine '{engine}' from settings.")
            engines[i] = engine

    # as an example, we here create 1 engine in each ensemble
    # if multi_engine = true
    else:
        for i in range(len(config["simulation"]["interfaces"])):
            eng_key = f"engine{i}"
            check_engine(config, eng_key=eng_key)
            engines[i - 1] = create_engine(config, eng_key)
            logger.info(f"Created {engines[i-1]} in ensemble {i:03d}.")

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
