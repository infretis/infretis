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
    """Create N_workers identical engines."""
    engines = {}
    check_engine(config)
    for i in range(config["runner"]["workers"]):
        engine = create_engine(config)
        logtxt = f'Created engine "{engine}" from settings.'
        logger.info(logtxt)
        engines[i] = engine
    return engines


def create_multi_engines(
    config: dict[str, Any]
) -> dict[Any, EngineBase | None]:
    """Create engines for methods requiring different engines.

    For quantis, we create N_workers identical engines, and add to that a
    single engine for [0-].

    We here also show how to define a single engine for each ensemble.
    """

    tmp_config = config.copy()
    if config["simulation"]["tis_set"]["quantis"]:
        logger.info("Creating Quantis engines.")
        # [N+] engines
        tmp_config["engine"] = config["engine0"]
        engines = create_engines(tmp_config)
        # [0-] engines
        tmp_config["engine"] = config["engine-1"]
        check_engine(tmp_config)
        engine_minus = create_engine(tmp_config)
        logger.info(f"Created [0-] engine {engine_minus}.")
        engines[-1] = engine_minus

    else:
        engines = {}
        logger.info("Creating 1 engine in each ensemble.")
        for i in range(-1, len(config["simulation"]["interfaces"]) - 1):
            tmp_config["engine"] = config[f"engine{i}"]
            check_engine(tmp_config)
            engines[i] = create_engine(tmp_config)
            logger.info(f"Created {engines[i]} in ensemble {i+1:03d}.")

    return engines


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
