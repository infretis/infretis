import dask.distributed
from dask.distributed import Client, as_completed 
import numpy as np
import time
import os
import tomli
from infretis import calc_cv_vector
from help_func import run_md, print_path_info, set_shooting, treat_output
from help_func import setup_pyretis, setup_repex, print_end
dask.config.config['work-stealing'] = False

if __name__ == "__main__":

    # read config and setup pyretis sim
    with open("./infretis.toml", mode="rb") as f:
        config = tomli.load(f)

    sim = setup_pyretis(config)
    moves = sim.settings['tis']['shooting_moves']
    interfaces = sim.settings['simulation']['interfaces']

    if 'current' not in config:
        config['current'] = {}
        config['current']['step'] = 0
        config['current']['traj_num'] = 0
        config['current']['active'] = [i for i in range(config['dask']['workers'])]
        config['current']['locked'] = [i for i in range(config['dask']['workers'])]
        config['current']['interfaces'] = sim.settings['simulation']['interfaces']
        config['current']['size'] = len(interfaces)

    steps = config['simulation']['steps']
    traj_num_dic = {}

    # setup dask and repex
    n_workers = config['dask']['workers']
    client = Client(n_workers=n_workers)
    futures = as_completed(None, with_results=True)
    state = setup_repex(sim, config, traj_num_dic)
    state.config = config
    
    # start running
    print('stored ensemble paths:')
    print(' '.join([f'00{i}: {j},' for i, j in enumerate([sim0['path_ensemble'].last_path.path_number for sim0 in sim.ensembles])]))
    print('saved ensemble paths:', [last_path0.path_number for last_path0 in state._trajs[:-1]])
    print_path_info(state)
    print(' ')

    # initialization: submit the first batch of workers
    for worker in range(n_workers):
    
        print(f'------- submit worker {worker} START -------')
    
        # Pick ens and trajs
        ens, input_traj = state.pick()
        if len(ens) > 1 or ens[0] == -1:
            move = 'sh'
        else:
            move = moves[ens[0]+1]
    
        # Print & assign worker to pin
        set_shooting(sim, ens, input_traj, str(worker), move)
    
        # submit job
        fut = client.submit(run_md, ens, input_traj, sim.settings, sim.ensembles, worker, move, str(worker), pure=False)
        futures.add(fut)
    
        print(f'------- submit worker {worker} END -------')
        print()
    
    while config['current']['step'] < steps + n_workers:
        step = config['current']['step']

        # get output from finished worker
        output = next(futures)[1]
        print(f'------- infinity {step} START -------')
    
    
        # analyze & store output
        start_time = time.time()
        pin = treat_output(output, state, sim, traj_num_dic)
        print('time spent swapping and saving paths:', f'{time.time() - start_time:.02f}')
    
        # print analyzed output and write toml
        print_path_info(state)

        if step < steps:
            # chose ens and path for the next job
            ens, input_traj = state.pick()
            if len(ens) > 1:
                move = 'sh'
            else:
                move = moves[ens[0]+1]
    
            # print & assign worker to pin
            set_shooting(sim, ens, input_traj, pin, move)
    
            # submit job
            fut = client.submit(run_md, ens, input_traj, sim.settings, sim.ensembles, step, move, pin, pure=False)
            futures.add(fut)
    
        print(f'------- infinity {step} END -------')
        config['current']['step'] += 1
    
        print()

    live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
    print_end(live_trajs, config['current']['step'], traj_num_dic)
