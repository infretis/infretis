import os
import numpy as np
import tomli
import logging
from infretis.classes.repex import REPEX_state
from infretis.classes.formats.formatter import get_log_formatter
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.ensemble import create_ensembles
from infretis.classes.engines.factory import create_engines
from infretis.classes.path import load_paths_from_disk
from dask.distributed import dask, Client, as_completed, get_worker
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
# Define a console logger. This will log to sys.stderr:
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(get_log_formatter(logging.WARNING))
logger.addHandler(console)

# def setup_internal(input_file):
def setup_internal(inp_file):

    # setup logger
    setup_logger()

    # setup toml config
    config = setup_config(inp_file)

    # setup repex 
    state = REPEX_state(config, minus=True)
    if 'restarted_from' in config['current']:
        state.set_rng()
    
    # setup ensembles
    state.ensembles = create_ensembles(config)

    # setup engines
    state.engines = create_engines(config)

    # setup engine orderparameter functions
    create_orderparameters(state.engines, config)
    
    # load paths from disk and add to repex
    paths = load_paths_from_disk(config)
    state.load_paths(paths)

    # create first md_items dict
    md_items = {'mc_moves': state.mc_moves,
                'interfaces': state.interfaces}

    # run pattern
    state.pattern()

    return md_items, state

def setup_dask(state):

    # isolate each worker
    dask.config.set({'distributed.scheduler.work-stealing': False})
    
    # setup client with state.workers workers
    client = Client(n_workers=state.workers)

    # in case external engine or o_parameter scripts are used
    for module in state.config['dask'].get('files', []):
        client.upload_file(module)

    # create future 
    futures = as_completed(None, with_results=True)

    # setup individual worker logs
    client.run(set_worker_logger)

    return client, futures

def setup_config(inp='infretis.toml', re_inp='restart.toml'):
    # sets up the dict from *toml file. 

    # load input:
    if os.path.isfile(inp):
        with open(inp, mode="rb") as f:
            config = tomli.load(f)
    else:
        logger.info(f'{inp} file not found, exit.')
        return
    
    # check if restart.toml exist:
    if inp != re_inp and os.path.isfile(re_inp):
        # load restart input:
        with open(re_inp, mode="rb") as f:
            re_config = tomli.load(f)

        # check if sim settings for the two are equal:
        equal = True
        for key in config.keys():
            if config[key] != re_config.get(key, {}):
                equal = False
                logger.info('We use {re_inp} instead.')
                break 
        config = re_config if equal else config

    # in case we restart, toml file has a 'current' subdict.
    if 'current' in config:
        curr = config['current']

        # if cstep and steps are equal, we stop here.
        if curr.get('cstep') == curr.get('restarted_from') != None:
            return 

        # set 'restarted_from'
        config['current']['restarted_from'] = config['current']['cstep']

        # check active paths:
        load_dir = config['simulation'].get('load_dir', 'trajs')
        for act in config['current']['active']:
            store_p = os.path.join(load_dir, str(act), 'traj.txt')
            if not os.path.isfile(store_p):
                return
    # no 'current' in toml, start from step 0.
    else:
        size = len(config['simulation']['interfaces'])
        config['current'] = {'traj_num': size, 'cstep': 0,
                             'active': list(range(size)),
                             'locked': [], 'size': size, 'frac': {}}
        # write/overwrite infretis_data.txt
        data_dir = config['output']['data_dir']
        data_file = os.path.join(data_dir, 'infretis_data.txt')
        config['output']['data_file'] = data_file
        with open(data_file, 'w') as fp:
            fp.write('# ' + '='*(34+8*size)+ '\n')
            ens_str = '\t'.join([f'{i:03.0f}' for i in range(size)])
            fp.write('# ' + f'\txxx\tlen\tmax OP\t\t{ens_str}\n')
            fp.write('# ' + '='*(34+8*size)+ '\n')

        # set pattern
        if config['output'].get('pattern', False):
            config['output']['pattern_file'] = os.path.join('pattern.txt')

    return config

def setup_logger(inp='sim.log'):
    fileh = logging.FileHandler(inp, mode='a')
    log_levl = getattr(logging, 'info'.upper(),
                       logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)

def set_worker_logger():
    # for each worker
    pin = get_worker().name
    log = logging.getLogger()
    fileh = logging.FileHandler(f"worker{pin}.log", mode='a')
    log_levl = getattr(logging, 'info'.upper(),
                       logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)
    logger.info(f'=============================')
    logger.info(f'Logging file for worker {pin}')
    logger.info(f'=============================\n')
