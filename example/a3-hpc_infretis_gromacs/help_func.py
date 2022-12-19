import os
import numpy as np
import time
import tomli_w
from dask.distributed import get_worker
from pyretis.core.tis import select_shoot
from pyretis.core.retis import retis_swap_zero # need to disable "add_path_data()"
from pyretis.setup import create_simulation
from pyretis.inout.settings import parse_settings_file
from infretis import calc_cv_vector, REPEX_state
from dask.distributed import Client, as_completed

# def run_md(ens_num, input_traj, sim, cycle, move, pin):
def run_md(md_items):
    ens_num = md_items['ens']
    cycle = md_items['cycle']
    input_traj = md_items['input_traj']
    settings = md_items['sim'].settings
    ensembles = md_items['sim'].ensembles
    interfaces = settings['simulation']['interfaces']
    start_time = time.time() 
    out = {'ensembles': [], 'accepted_trajs': [],                                        
           'traj_vectors': [], 'status': None,
           'pin': md_items['pin']}
    path_numbers_old = []

    for traj0 in input_traj:
        path_numbers_old.append(traj0.path_number)
    out['path_numbers_old'] = path_numbers_old
    
    if len(ens_num) == 1:
        start_cond = ensembles[ens_num[0]+1]['path_ensemble'].start_condition
        tis_settings = settings['ensemble'][ens_num[0]+1]['tis'] 

        accept, trial, status = select_shoot(ensembles[ens_num[0]+1],
                                             tis_settings,
                                             start_cond)
        if accept:                                                                  
            out_traj = trial                                                        
        else:                                                                       
            out_traj = input_traj[0]
        if ens_num[0] < 0:
            interfaces = interfaces[0:1]

        move = tis_settings.get('shooting_move', 'sh')
        cv_vector = calc_cv_vector(out_traj, interfaces, move)
                
        out['ensembles'] = list(ens_num)
        out['accepted_trajs'] = [out_traj]
        out['traj_vectors'] = [cv_vector]
        out['status'] = status
        out['move'] = move
    else:
        accept, trial, status = retis_swap_zero(ensembles, 
                                                settings,
                                                cycle)
        if accept:                                                                  
            out_traj = trial                                                        
        else:                                                                       
            out_traj = input_traj

        out['ensembles'] = list(ens_num)
        out['accepted_trajs'] = out_traj
        ifaces = [[interfaces[0:1]], interfaces]
        out['move'] = 'sh'
        for traj, iface in zip(out_traj, ifaces):
            out['traj_vectors'].append(calc_cv_vector(traj, iface, out['move']))     
        out['status'] = status
    curr_time = time.time() 
    out['time'] = curr_time - start_time
    return out


def print_path_info(state, ens_sel=(), input_traj=()):
    pwd_checker(state)
    state.print_state()
    write_toml(state, ens_sel=(), input_traj=())


def set_shooting(sim, ens, input_traj, pin):
    moves = sim.settings['tis']['shooting_moves']

    if len(ens) > 1 or ens[0] == -1:
        move = 'sh'
    else:
        move = moves[ens[0]+1]

    print('shooting', move, 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in ens]),
          'with paths:', ' '.join([str(trajj.path_number) for trajj in input_traj]),
          'and worker:', pin)

    for ens_num, traj_inp in zip(ens, input_traj):
        ens_num += 1
        sim.ensembles[ens_num]['path_ensemble'].last_path = traj_inp.copy()

    return move

def treat_output(output, state, sim, save=False):
    traj_num_dic = state.traj_num_dic
    traj_num = state.config['current']['traj_num']
    size = state.config['current']['size']

    ensembles = output['ensembles']
    out_trajs = output['accepted_trajs']
    traj_vectors = output['traj_vectors']
    status = output['status']
    path_numbers_old = output['path_numbers_old']
    pn_archive = []
    pin = output['pin']
    move = output['move']
    time_spent = output['time']
    
    # analyse and record worker data
    for ens_num, out_traj, traj_v, pn_old in zip(ensembles,
                                                 out_trajs,
                                                 traj_vectors,
                                                 path_numbers_old):
        # if path is new: number and save the path:
        if out_traj.path_number == None or status == 'ACC':
            out_traj.path_number = traj_num
            pn_archive.append(pn_old)
            ens_save_idx = traj_num_dic[pn_old]['ens_idx']
            traj_num_dic[traj_num] = {'weight': np.zeros(size+1),
                                      'adress': set(kk.particles.config[0].split('salt')[-1] 
                                                    for kk in out_traj.phasepoints),
                                      'ens_idx': ens_save_idx,
                                      'max_op': out_traj.ordermax,
                                      'length': out_traj.length}
            traj_num += 1
            sim.ensembles[ens_save_idx]['path_ensemble'].store_path(out_traj)
            
            cycle = {'step': traj_num -1 , 'endcycle': 10, 'startcycle': 0, 'stepno': 10, 'steps': 10}
            result = {f'status-{ens_num+1}': 'ACC', 'cycle': cycle, f'path-{ens_num+1}':  out_traj,
                      f'accept-{ens_num+1}': True, f'move-{ens_num+1}': 'sh', 
                      'all-2': {'ensemble_number': ens_num+1, 'mc-move': 'sh',
                                'status': 'ACC', 'trial': out_traj, 'accept': True},
                      f'pathensemble-{ens_num+1}': sim.ensembles[0]['path_ensemble']}
            
            # NB! Saving can take some time..

            if save:
                flipppa = time.time() 
                for task in sim.output_tasks:
                    task.output(result)
                print('saving path time:', time.time() - flipppa)
            
        state.add_traj(ens_num, out_traj, traj_v)
        
        # record weights 
        live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
        
        traj_lock = [traj0.path_number for traj0, lock0 in
                     zip(state._trajs[:-1], state._locks[:-1]) if lock0]
        w_start = 0
        for live in live_trajs:
            if live not in traj_lock:
                for weight in state._last_prob[w_start:-1]:
                    w_start += 1
                    if sum(weight) != 0:
                        traj_num_dic[live]['weight'] += weight
                        break
    
    # print information to screen
    print('shooted', move, 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in ensembles]),
          'with paths:', ' '.join([str(pn_old) for pn_old in path_numbers_old]), '->', 
          ' '.join([str(trajj.path_number) for trajj in out_trajs]),
          'with status:', status, 'and worker:', pin, f'total time: {time_spent:.2f}')

    state.config['current']['traj_num'] = traj_num

    # print analyzed output
    if'ACC' == status:
        write_to_pathens(state, pn_archive)

    return pin


def setup_pyretis(config):
    if 'current' not in config:
        config['current'] = {}
        config['current']['step'] = 0
        config['current']['active'] = []
        config['current']['locked'] = []
        config['current']['dic'] = []
        with open('infretis_data.txt', 'w') as fp:
            fp.write('# ' + '='*66 + '\n')
            fp.write('# ' + '\tpnum\t\tplen\t\tpmaxOP\t\t000     001     002     003     004     \n')
            fp.write('# ' + '-'*66 + '\n')
            pass

    ## load paths from traj/{0,1,2,3} etc...
    ## change that if config active  .. 

    inp = config['simulation']['pyretis_inp']
    sim_settings = parse_settings_file(inp)
    interfaces = sim_settings['simulation']['interfaces']
    size = len(interfaces)

    if not config['current']['active']:
        config['current']['active'] = list(range(size))

    active = config['current']['active']
    locks = config['current']['locked']

    config['current']['interfaces'] = interfaces
    config['current']['size'] = size
    sim_settings['current'] = {'size': size}
    sim_settings['current']['active'] = config['current']['active']

    # if active:
    # load from active ..

    sim = create_simulation(sim_settings)
    sim.set_up_output(sim_settings, progress=True)
    sim.initiate(sim_settings)

    return sim

def setup_repex(sim, config):

    ## if config['current']['locked']: then ....

    size = config['current']['size']
    interfaces = config['current']['interfaces']
    active = config['current']['active']
    state = REPEX_state(size, workers=config['dask']['workers'],
                        minus=True)
    traj_num_dic = state.traj_num_dic
    moves = sim.settings['tis']['shooting_moves']

    ## initiate by adding paths from retis sim to repex
    for i in range(size-1):
        # we add all the i+ paths.
        path = sim.ensembles[i+1]['path_ensemble'].last_path
        state.add_traj(ens=i, traj=path,
                       valid=calc_cv_vector(path, interfaces, moves[i+1]),
                       count=False)
        traj_num_dic[path.path_number] = {'weight': np.zeros(size+1),
                                          'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                         for kk in path.phasepoints),
                                          'ens_idx': i + 1,
                                          'max_op': path.ordermax,
                                          'length': path.length}
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    state.add_traj(ens=-1, traj=path, valid=(1,), count=False)
    traj_num_dic[path.path_number] = {'weight': np.zeros(size+1),
                                      'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                     for kk in path.phasepoints),
                                      'ens_idx': 0,
                                      'max_op': path.ordermax,
                                      'length': path.length}

    if 'traj_num' not in config['current'].keys():
        config['current']['traj_num'] = max(config['current']['active']) + 1
    state.config = config
    return state

# def print_end(live_trajs, stopping, traj_num_dic):
def print_end(state):
    live_trajs = state.live_paths()
    stopping = state.cstep
    traj_num_dic = state.traj_num_dic
    print('--------------------------------------------------')
    print('live trajs:', live_trajs, f'after {stopping-1} cycles')
    print('==================================================')
    print('xxx | 000        001     002     003     004     |')
    print('--------------------------------------------------')
    for key, item in traj_num_dic.items():
        print(f'{key:03.0f}', "|" if key not in live_trajs else '*',
              '\t'.join([f'{item0:02.2f}' if item0 != 0.0 else '---' for item0 in item['weight'][:-1]])
             ,'\t', "|" if key not in live_trajs else '*')


def write_to_pathens(state, pn_archive):
    traj_num_dic = state.traj_num_dic
    with open('infretis_data.txt', 'a') as fp:
        for pn in pn_archive:
            string = ''
            string += f'\t{pn:03.0f}\t\t'
            string += f"{traj_num_dic[pn]['length']:05.0f}" + '\t'
            string += f"{traj_num_dic[pn]['max_op'][0]:05.5f}" + '\t\t'
            weight = '\t'.join([f'{item0:02.2f}' if item0 != 0.0 else '----' for item0 in traj_num_dic[pn]['weight'][:-1]])
            fp.write(string + weight + '\t\n')


def setup_internal(config):

    # check if we restart or not 
    if 'current' not in config or config['current'].get('step') == 0:
        config['current'] = {}
        config['current']['step'] = 0
        config['current']['active'] = []
        config['current']['locked'] = []
        config['current']['dic'] = []
        with open('infretis_data.txt', 'w') as fp:
            fp.write('# ' + '='*66 + '\n')
            fp.write('# ' + '\txxx\t\tlen\t\tmax OP\t\t000     001     002     003     004     \n')
            fp.write('# ' + '-'*66 + '\n')
            pass

    # setup pyretis
    inp = config['simulation']['pyretis_inp']
    sim_settings = parse_settings_file(inp)

    interfaces = sim_settings['simulation']['interfaces']
    size = len(interfaces)
    config['current']['size'] = size
    if not config['current']['active']:
        config['current']['active'] = list(range(size))

    # give path to the active paths
    sim_settings['current'] = {'size': size}
    sim_settings['current']['active'] = config['current']['active']
    sim = create_simulation(sim_settings)
    sim.set_up_output(sim_settings)
    sim.initiate(sim_settings)

    # setup infretis
    state = REPEX_state(size, workers=config['dask']['workers'],
                        minus=True)
    state.steps = config['simulation']['steps']
    state.cstep = config['current']['step']
    traj_num_dic = state.traj_num_dic
    state.mc_moves = sim.settings['tis']['shooting_moves']

    ## initiate by adding paths from retis sim to repex
    for i in range(size-1):
        # we add all the i+ paths.
        path = sim.ensembles[i+1]['path_ensemble'].last_path
        state.add_traj(ens=i, traj=path,
                       valid=calc_cv_vector(path, interfaces, state.mc_moves[i+1]),
                       count=False)
        traj_num_dic[path.path_number] = {'weight': np.zeros(size+1),
                                          'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                         for kk in path.phasepoints),
                                          'ens_idx': i + 1,
                                          'max_op': path.ordermax,
                                          'length': path.length}
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    state.add_traj(ens=-1, traj=path, valid=(1,), count=False)
    traj_num_dic[path.path_number] = {'weight': np.zeros(size+1),
                                      'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                     for kk in path.phasepoints),
                                      'ens_idx': 0,
                                      'max_op': path.ordermax,
                                      'length': path.length}

    if 'traj_num' not in config['current'].keys():
        config['current']['traj_num'] = max(config['current']['active']) + 1
    state.config = config
    return {'sim': sim}, state


def setup_dask(workers):
    client = Client(n_workers=workers)
    futures = as_completed(None, with_results=True)
    return client, futures


def pwd_checker(state):
    all_good = True

    ens_no = len(state._trajs[:-1])
    ens_str = [f'{i:03.0f}' for i in range(ens_no)]
    state_dic = {}
    path_dic = {}
    ens_pwds = []
    locks = [traj0.path_number for traj0, lock0 in
             zip(state._trajs[:-1], state._locks[:-1]) if lock0]

    locks_idx = [i for i, lock0 in enumerate(state._locks[:-1]) if lock0]
    locks_ens = [f'{i:03.0f}' for i, lock0 in enumerate(state._locks[:-1]) if lock0]

    for path_temp in state._trajs[:-1]:
        path_pwds = sorted(set([pp.particles.config[0] for pp in path_temp.phasepoints]))
        ens = next(i for i in path_pwds[0].split('/') if i in ens_str)
        state_dic[ens] = {'pwds': [pwd.split('/')[-1] for pwd in path_pwds]}
        state_dic[ens]['path_number'] = path_temp.path_number
        path_dic[state_dic[ens]['path_number']] = [pwd.split('k')[-1] for pwd in path_pwds]

    for ens in ens_str:
        ens_pwds.append(sorted(os.listdir(f'./{ens}/accepted')))

    # check if state_paths correspond to path_pwds:
    for ens, string1 in zip(ens_str, ens_pwds):
        string0 = state_dic[ens]['pwds']
        if string0 != string1:
            print(string0, string1)
            print('warning! the state_paths does not correspond to the path_pwds!')
            all_good = False

    return all_good


def write_toml(state, ens_sel=(), input_traj=()):
    state.config['current']['active'] = state.live_paths()
    locked_ep = []
    for ens0, path0 in zip(ens_sel, input_traj):
        locked_ep.append((int(ens0 + state._offset), path0.path_number))
    state.config['current']['locked'] = locked_ep

    with open("./infretis_5.toml", "wb") as f:
        tomli_w.dump(state.config, f)  


def prepare_shooting(state, md_items):
    # pwd_checker
    pwd_checker(state)
    # print state:
    state.print_state()
    # write toml:
    ens, input_traj = md_items['ens'], md_items['input_traj']
    write_toml(state, ens, input_traj)
    # chose move and print
    if len(ens) > 1 or ens[0] == -1:
        move = 'sh'
    else:
        move = state.mc_moves[ens[0]+1]

    print('shooting', move, 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in ens]),
          'with paths:', ' '.join([str(trajj.path_number) for trajj in input_traj]),
          'and worker:', md_items['pin'])

    # update pwd
    for ens_num, traj_inp in zip(ens, input_traj):
        ens_num += 1
        md_items['sim'].ensembles[ens_num]['path_ensemble'].last_path = traj_inp.copy()
    



