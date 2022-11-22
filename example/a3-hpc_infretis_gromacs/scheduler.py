import dask.distributed
from dask.distributed import Client, as_completed 
import numpy as np
import time
import os
from pyretis.setup import create_simulation
from pyretis.inout.settings import parse_settings_file
from infretis import REPEX_state, calc_cv_vector
from help_func import run_md, print_path_info, set_shooting, treat_output
dask.config.config['work-stealing'] = False

if __name__ == "__main__":

    ## set up dask
    n_workers = 1
    client = Client(n_workers=n_workers)
    futures = as_completed(None, with_results=True)
    stopping = 100

    ## set up pyretis
    INP = './retis.rst'
    sim_settings = parse_settings_file(INP)
    sim = create_simulation(sim_settings)
    sim.set_up_output(sim_settings, progress=True)
    sim.initiate(sim_settings)
    interfaces = sim.settings['simulation']['interfaces']
    size=len(interfaces)
    ens_str = [f'{i:03.0f}' for i in range(len(sim.settings['simulation']['interfaces']))]
    traj_num_dic = {}            # store traj data
    traj_num = 0                 # numbering of accepted paths
    moves = sim.settings['tis']['shooting_moves']
    move_d = {i: move for i, move in zip(list(range(len(interfaces))), moves)}

    ## set up repex
    state = REPEX_state(size, minus=True)

    ## initiate by adding paths from retis sim to repex
    for i in range(size-1):
        # we add all the i+ paths.
        path = sim.ensembles[i+1]['path_ensemble'].last_path
        path.path_number = traj_num
        state.add_traj(ens=i, traj=path,
                       valid=calc_cv_vector(path, interfaces, move_d[i+1]),
                       count=False)
        traj_num_dic[traj_num] = {'weight': np.zeros(size+1),
                                  'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                 for kk in path.phasepoints),
                                  'ens_idx': traj_num + 1}
        traj_num += 1
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    path.path_number = traj_num
    state.add_traj(ens=-1, traj=path, valid=(1,), count=False)
    traj_num_dic[traj_num] = {'weight': np.zeros(size+1),
                              'adress':  set(kk.particles.config[0].split('salt')[-1]
                                             for kk in path.phasepoints),
                              'ens_idx': 0}
    traj_num += 1
    
    # Define an iterator that is going to be our results queue
    futures = as_completed(None, with_results=True)
    
    # Start running
    loop = 0
    print('stored ensemble paths:')
    print(' '.join([f'00{i}: {j},' for i, j in enumerate([sim0['path_ensemble'].last_path.path_number for sim0 in sim.ensembles])]))
    print('saved ensemble paths:', [last_path0.path_number for last_path0 in state._trajs[:-1]])
    print_path_info(state)
    print(' ')

    # initialization: submit the first batch of workers
    for worker in range(n_workers):
    
        print(f'------- infinity {loop} START -------')
    
        # Pick ens and trajs
        ens, input_traj = state.pick()
        if len(ens) > 1 or ens[0] == -1:
            move = 'sh'
        else:
            move = move_d[ens[0]+1]
    
        # Print & assign worker to pin
        set_shooting(sim, ens, input_traj, str(worker), move)
    
        # submit job
        fut = client.submit(run_md, ens, input_traj, sim.settings, sim.ensembles, loop, move, str(worker), pure=False)
        futures.add(fut)
    
        print(f'------- infinity {loop} END -------')
        loop += 1
        print()
    
    # while loop
    while loop < stopping:
    
        # get output from finished worker
        output = next(futures)[1]
        print(f'------- infinity {loop} START -------')
    
    
        # analyze & store output
        start_time = time.time()
        traj_num_dic, traj_num, pin = treat_output(output, state, sim,
                                                   traj_num_dic, traj_num, size)
        print('time spent swapping and saving paths:', f'{time.time() - start_time:.02f}')
    
        # print analyzed output
        print_path_info(state)
    
        # chose ens and path for the next job
        ens, input_traj = state.pick()
        if len(ens) > 1:
            move = 'sh'
        else:
            move = move_d[ens[0]+1]
    
        # print & assign worker to pin
        set_shooting(sim, ens, input_traj, pin, move)
    
        # submit job
        fut = client.submit(run_md, ens, input_traj, sim.settings, sim.ensembles, loop, move, pin, pure=False)
        futures.add(fut)
    
        print(f'------- infinity {loop} END -------')
        loop += 1
    
        print()
    
    # get the last futures out
    for output_l in futures:
        output = output_l[1]
        print(f'------- infinity {loop} START -------')
    
        traj_num_dic, traj_num, pin = treat_output(output, state, sim,
                                                   traj_num_dic, traj_num, size)
    
        print_path_info(state)
        print(f'------- infinity {loop} END -------')
        loop += 1
        print()


    live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
    
    print('--------------------------------------------------')
    print('live trajs:', live_trajs, f'after {stopping} cycles')
    print('==================================================')
    print('xxx | 000        001     002     003     004     |')
    print('--------------------------------------------------')
    for key, item in traj_num_dic.items():
        print(f'{key:03.0f}', "|" if key not in live_trajs else '*',
              '\t'.join([f'{item0:02.2f}' if item0 != 0.0 else '---' for item0 in item['weight'][:-1]])
             ,'\t', "|" if key not in live_trajs else '*')
