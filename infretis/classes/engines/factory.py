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


def create_engine(settings: dict[str, Any]) -> EngineBase | None:
    """Create an engine from settings.

    Args:
        settings: Settings for the simulation. This method will
            use the `"engine"` section of the settings.

    Returns:
        The engine created here.
    """
    engine_map = {
        "gromacs": {"class": GromacsEngine},
        "cp2k": {"class": CP2KEngine},
        "turtlemd": {"class": TurtleMDEngine},
        "lammps": {"class": LAMMPSEngine},
    }

    if settings["engine"]["class"].lower() not in engine_map:
        return create_external(settings["engine"], "engine", ["step"])
    engine = generic_factory(settings["engine"], engine_map, name="engine")
    return engine


def create_engines(config: dict[str, Any]) -> dict[Any, EngineBase | None]:
    """Create engines."""
    if config.get("engine", {}).get("obj", False):
        return config["engine"]["obj"]

    check_engine(config)
    engine = create_engine(config)
    logtxt = f'Created engine "{engine}" from settings.'
    logger.info(logtxt)
    # quantis stuff
    if config["simulation"]["tis_set"]["quantis"]:
        if "engine2" not in config.keys():
            msg = "quantis = true but engine2 was not specified in the .toml"
            raise ValueError(msg)
        engine2 = create_engine({"engine": config["engine2"]})
        return {
            config["engine"]["engine"]: engine,
            config["engine2"]["engine"] + "2": engine2,
        }

    return {config["engine"]["engine"]: engine}


def check_engine(settings: dict[str, Any]) -> bool:
    """Check the input settings for engine creation.

    Args:
        settings: The input settings to use for creating the engine.
    """
    msg = []
    if "engine" not in settings:
        msg += ["The section engine is missing"]
    if "input_path" not in settings["engine"]:
        msg += ["The section engine requires an input_path entry"]

    if "gmx" in settings["engine"] and "gmx_format" not in settings["engine"]:
        msg += ["File format is not specified for the engine"]
    elif (
        "cp2k" in settings["engine"]
        and "cp2k_format" not in settings["engine"]
    ):
        msg += ["File format is not specified for the engine"]

    if msg:
        msgtxt = "\n".join(msg)
        logger.critical(msgtxt)
        return False

    return True
