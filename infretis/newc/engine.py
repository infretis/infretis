from infretis.newc.engines.gromacs import GromacsEngine
from infretis.newc.orderparameter import create_orderparameter
from infretis.newf.core import generic_factory

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def create_engine(settings):
    """Create an engine from settings.

    Parameters
    ----------
    settings : dict
        This dictionary contains the settings for the simulation.

    Returns
    -------
    out : object like :py:class:`.EngineBase`
        This object represents the engine.

    """
    engine_map = {
        'gromacs': {'cls': GromacsEngine},
    }

    engine = generic_factory(settings['engine'], engine_map, name='engine')
    engine.order_function = create_orderparameter(settings)

    # raise ValueError('Could not create engine from settings!')
    return engine

def create_engines(config):
    if config.get('engine', {}).get('obj', False):
        return config['engine']['obj']

    check_engine(config)
    engine = create_engine(config)
    logtxt = f'Created engine "{engine}" from settings.'
    logger.info(logtxt)
    return {config['engine']['engine']: engine}

def check_engine(settings):
    """Check the engine settings.

    Checks that the input engine settings are correct, and
    automatically determine the 'internal' or 'external'
    engine setting.

    Parameters
    ----------
    settings : dict
        The current input settings.

    """
    msg = []
    if 'engine' not in settings:
        msg += ['The section engine is missing']

    elif settings['engine'].get('type') == 'external':

        if 'input_path' not in settings['engine']:
            msg += ['The section engine requires an input_path entry']

        if 'gmx' in settings['engine'] and \
                'gmx_format' not in settings['engine']:
            msg += ['File format is not specified for the engine']
        elif 'cp2k' in settings['engine'] and \
                'cp2k_format' not in settings['engine']:
            msg += ['File format is not specified for the engine']

    if msg:
        msgtxt = '\n'.join(msg)
        logger.critical(msgtxt)
        return False

    return True
