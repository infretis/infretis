import os
import numpy as np
import time
from pyretis.core.tis import select_shoot
from pyretis.core.retis import retis_swap_zero
from pyretis.setup import create_simulation
from pyretis.inout.settings import parse_settings_file
from pyretis.inout.restart import write_ensemble_restart
from pyretis.inout.common import make_dirs
from infretis.inf_core import REPEX_state
from dask.distributed import dask, Client, as_completed
from pyretis.core.common import compute_weight
dask.config.set({'distributed.scheduler.work-stealing': False})


def run_md(md_items):
    start_time = time.time() 
    ens_nums = md_items['ens_nums']
    ensembles = md_items['ensembles']
    settings = md_items['settings']
    interfaces = settings['simulation']['interfaces']

    if len(ens_nums) == 1:
        start_cond = ensembles[ens_nums[0]+1]['path_ensemble'].start_condition
        tis_settings = settings['ensemble'][ens_nums[0]+1]['tis'] 
        if not md_items['internal']:
            ensembles[ens_nums[0]+1]['engine'].clean_up()
        accept, trials, status = select_shoot(ensembles[ens_nums[0]+1],
                                              tis_settings,
                                              start_cond)
        trials = [trials]
        interfaces = [interfaces] if ens_nums[0] >= 0 else [interfaces[0:1]]

    else:
        ensembles_l = [ensembles[i+1] for i in ens_nums]
        if not md_items['internal']:
            ensembles_l[0]['engine'].clean_up()
            ensembles_l[1]['engine'].clean_up()
        accept, trials, status = retis_swap_zero(ensembles_l, settings, 0)
        interfaces = [interfaces[0:1], interfaces]

    for trial, ens_num, ifaces in zip(trials, ens_nums, interfaces):
        md_items['moves'].append(md_items['mc_moves'][ens_num+1])
        md_items['pnum_old'].append(ensembles[ens_num+1]['path_ensemble'].last_path.path_number)
        if status == 'ACC':
            trial.traj_v = calc_cv_vector(trial, ifaces, md_items['mc_moves'])
            ensembles[ens_num+1]['path_ensemble'].last_path = trial

    end_time = time.time()
    md_items.update({'status': status,
                     'interfaces': interfaces,
                     'time': end_time - start_time,
                     'start_time': start_time,
                     'end_time': end_time})
    return md_items


def treat_output(state, md_items):
    traj_num_dic = state.traj_num_dic
    traj_num = state.config['current']['traj_num']
    ensembles = md_items['ensembles']
    pn_news = []

    # analyse and record worker data
    for ens_num, pn_old in zip(md_items['ens_nums'],
                               md_items['pnum_old']):
        # if path is new: number and save the path:
        out_traj = ensembles[ens_num+1]['path_ensemble'].last_path
        state.ensembles[ens_num+1] = ensembles[ens_num+1]
        if out_traj.path_number == None or md_items['status'] == 'ACC':
            # move to accept:
            ens_save_idx = traj_num_dic[pn_old]['ens_save_idx']
            state.ensembles[ens_save_idx]['path_ensemble'].store_path(out_traj)
            out_traj.path_number = traj_num
            traj_num_dic[traj_num] = {'frac': np.zeros(state.n, dtype="float128"),
                                      'max_op': out_traj.ordermax,
                                      'length': out_traj.length,
                                      'traj_v': out_traj.traj_v,
                                      'ens_save_idx': ens_save_idx}
            if not md_items['internal']:
                traj_num_dic[traj_num]['adress'] = set(kk.particles.config[0].split('salt')[-1] 
                                                       for kk in out_traj.phasepoints)
            traj_num += 1
            
            # NB! Saving can take some time..
            if state.config['output']['store_paths']:
                cycle = {'step': traj_num -1 , 'endcycle': 10,
                         'startcycle': 0, 'stepno': 10, 'steps': 10}
                result = {f'status-{ens_num+1}': 'ACC', 'cycle': cycle,
                          f'path-{ens_num+1}':  out_traj, f'accept-{ens_num+1}': True,
                          f'move-{ens_num+1}': 'sh', 
                          'all-2': {'ensemble_number': ens_num+1, 'mc-move': 'sh',
                                    'status': 'ACC', 'trial': out_traj, 'accept': True},
                          f'pathensemble-{ens_num+1}': state.ensembles[0]['path_ensemble']}
                flipppa = time.time() 
                if md_items['internal'] and state.config['output']['store_paths']:
                    dirname = f'./trajs/{out_traj.path_number}'
                    make_dirs(dirname)
                for task in state.output_tasks:
                    task.output(result)

        if state.config['output']['store_paths']:
            # save ensemble-path_ensemble-rgen (not used) and ensemble-path
            write_ensemble_restart(state.ensembles[ens_num+1], md_items['settings'], save='path')
            # save ensemble-rgen, ensemble-engine-rgen
            write_ensemble_restart(state.ensembles[ens_num+1], md_items['settings'], save=f'e{ens_num+1}')
            
        pn_news.append(out_traj.path_number)
        state.add_traj(ens_num, out_traj, out_traj.traj_v)

        ensembles.pop(ens_num+1)
    state.config['current']['traj_num'] = traj_num
        
    # record weights 
    locked_trajs = state.locked_paths()
    for idx, live in enumerate(state.live_paths()):
        if live not in locked_trajs:
            traj_num_dic[live]['frac'] += state._last_prob[:-1][idx, :]

    # write succ data to infretis_data.txt
    if md_items['status'] == 'ACC':
        write_to_pathens(state, md_items['pnum_old'])

    # print information to screen
    if state.screen > 0 and np.mod(state.cstep, state.screen) == 0:
        print('shooted', ' '.join(md_items['moves']), 'in ensembles:',
              ' '.join([f'00{ens_num+1}' for ens_num in md_items['ens_nums']]),
              'with paths:', ' '.join([str(pn_old) for pn_old in md_items['pnum_old']]),
              '->', ' '.join([str(pn_new) for pn_new in pn_news]),
              'with status:', md_items['status'], 'and worker:',
              md_items['pin'], f"total time: {md_items['time']:.2f}")
        state.print_state()


def write_to_pathens(state, pn_archive):
    traj_num_dic = state.traj_num_dic
    size = state.n

    with open(state.data_file, 'a') as fp:
        for pn in pn_archive:
            string = ''
            string += f'\t{pn:3.0f}\t'
            string += f"{traj_num_dic[pn]['length']:5.0f}" + '\t'
            string += f"{traj_num_dic[pn]['max_op'][0]:8.5f}" + '\t'
            frac = []
            weight = []
            if len(traj_num_dic[pn]['traj_v']) == 1:
                f0 = traj_num_dic[pn]['frac'][0]
                w0 = traj_num_dic[pn]['traj_v'][0]
                frac.append('----' if f0 == 0.0 else f"{f0:8.3f}")
                if weight == 0:
                    print('tortoise', frac, weight)
                    exit('fish')
                weight.append('----' if f0 == 0.0 else f"{w0:8.0f}")
                frac += ['----']*(size-2)
                weight += ['----']*(size-2)
            else:
                frac.append('----')
                weight.append(f'----')
                for w0, f0 in zip(traj_num_dic[pn]['traj_v'][:-1],
                                  traj_num_dic[pn]['frac'][1:-1]):
                    frac.append('----' if f0 == 0.0 else f"{f0:5.3f}")
                    weight.append('----' if f0 == 0.0 else f"{w0:5.0f}")
            fp.write(string + '\t'.join(frac) + '\t' + '\t'.join(weight) + '\t\n')
            traj_num_dic.pop(pn)


def setup_internal(config):
    # setup pyretis
    inp = config['simulation']['pyretis_inp']
    sim_settings = parse_settings_file(inp)
    interfaces = sim_settings['simulation']['interfaces']
    size = len(interfaces)

    data_dir = config['output']['data_dir']
    pattern_file = os.path.join(data_dir, 'pattern.txt')
    data_file = os.path.join(data_dir, 'infretis_data.txt')

    # check if we restart or not 
    if 'current' not in config:
        config['current'] = {}
        config['current']['traj_num'] = 0
        config['current']['cstep'] = 0
        config['current']['active'] = list(range(size))
        config['current']['locked'] = []
        config['current']['size'] = size
        with open(data_file, 'w') as fp:
            fp.write('# ' + '='*(34+8*size)+ '\n')
            ens_str = '\t'.join([f'{i:03.0f}' for i in range(size)])
            fp.write('# ' + f'\txxx\tlen\tmax OP\t\t{ens_str}\n')
            fp.write('# ' + '='*(34+8*size)+ '\n')
    else:
        config['current']['restarted-from'] = config['current']['cstep']
        if config['current']['cstep'] == config['simulation']['steps']:
            print('current step and total steps are equal so we exit ',
                  'without doing anything.')
            return None, None

    # give path to the active paths
    sim_settings['current'] = {'size': size}
    sim_settings['current']['active'] = config['current']['active']

    sim = create_simulation(sim_settings)

    for idx, pn in enumerate(config['current']['active']):
        sim.ensembles[idx]['path_ensemble'].path_number = pn

    # path rng and ensemble rng objects have to be saved separately!
    # create worker/ensemble restart folders
    for worker in range(config['dask']['workers']):
        dirname = f'./trajs/e{worker}'
        make_dirs(dirname)

    sim.set_up_output(sim_settings)
    sim.initiate(sim_settings)

    if config['current']['traj_num'] == 0:
        for i_ens in sim.ensembles:
            i_ens['path_ensemble'].last_path.path_number = config['current']['traj_num']
            config['current']['traj_num'] += 1

    # setup infretis
    state = REPEX_state(size, workers=config['dask']['workers'],
                        minus=True)
    state.tsteps = config['simulation']['steps']
    state.cstep = config['current']['cstep']
    state.screen = config['output']['screen']
    state.pattern = config['output']['pattern']
    state.output_tasks = sim.output_tasks
    state.mc_moves = sim.settings['tis']['shooting_moves']
    traj_num_dic = state.traj_num_dic


    # load acc frac if restart
    for path_num in config['current']['active']:
        if 'frac' in config['current']:
            traj_num_dic[path_num] = {'frac': np.array(config['current']['frac'].get(str(path_num), np.zeros(size+1)), dtype="float128")}
        else:
            traj_num_dic[path_num] = {'frac': np.zeros(size+1, dtype="float128")}

    ## initiate by adding paths from retis sim to repex
    for i in range(size-1):
        # we add all the i+ paths.
        path = sim.ensembles[i+1]['path_ensemble'].last_path
        path.traj_v = calc_cv_vector(path, interfaces, state.mc_moves)
        state.add_traj(ens=i, traj=path, valid=path.traj_v, count=False)
        traj_num_dic[path.path_number].update({'ens_save_idx': i + 1,  
                                               'max_op': path.ordermax,
                                               'length': path.length,  
                                               'traj_v': path.traj_v})  
        if not config['simulation']['internal']:
            traj_num_dic[path.path_number]['adress'] = set(kk.particles.config[0].split('salt')[-1]
                                                           for kk in path.phasepoints)
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    path.traj_v = (1,)
    state.add_traj(ens=-1, traj=path, valid=path.traj_v, count=False)
    traj_num_dic[path.path_number].update({'ens_save_idx': 0,      
                                           'max_op': path.ordermax,
                                           'length': path.length,  
                                           'traj_v': path.traj_v})
    if not config['simulation']['internal']:
        traj_num_dic[path.path_number]['adress'] = set(kk.particles.config[0].split('salt')[-1]
                                                       for kk in path.phasepoints)

    if 'traj_num' not in config['current'].keys():
        config['current']['traj_num'] = max(config['current']['active']) + 1
    state.config = config
    state.pattern_file = pattern_file
    state.data_file = data_file
    if 'restarted-from' in config['current']:
        state.set_rng()

    state.ensembles = {i: sim.ensembles[i] for i in range(len(sim.ensembles))}
    sim.settings['initial-path']['load_folder'] = 'trajs'
    md_items = {'settings': sim.settings,
                'mc_moves': state.mc_moves,
                'ensembles': {}, 
                'internal': config['simulation']['internal']}

    print('start zebra 0')
    for idx, traj in enumerate(state._trajs[:-1]):
        print(idx, traj.path_number, traj.ordermax)
    print('start zebra 1')

    return md_items, state


def setup_dask(workers):
    client = Client(n_workers=workers)
    futures = as_completed(None, with_results=True)
    return client, futures


def pwd_checker(state):
    all_good = True
    ens_str = [f'{i:03.0f}' for i in range(state.n-1)]
    state_dic = {}

    for path_temp in state._trajs[:-1]:
        path_pwds = sorted(set([pp.particles.config[0] for pp in path_temp.phasepoints]))
        ens = next(i for i in path_pwds[0].split('/') if i in ens_str)
        state_dic[ens] = {'pwds': [pwd.split('/')[-1] for pwd in path_pwds]}
        state_dic[ens]['path_number'] = path_temp.path_number

    ens_pwds = []
    for ens in ens_str:
        ens_pwds.append(sorted(os.listdir(f'./{ens}/accepted')))

    # check if state_paths correspond to path_pwds:
    for ens, string1 in zip(ens_str, ens_pwds):
        string0 = state_dic[ens]['pwds']
        if string0 != string1:
            print(string0, string1)
            print('warning! the state_paths does' + \
                  'not correspond to the path_pwds!')
            all_good = False

    return all_good


def prep_pyretis(state, md_items, inp_traj, ens_nums):

    # pwd_checker
    if not md_items['internal']:
        if not pwd_checker(state):
            exit('sumtin fishy goin on here')

    # write toml and rng:
    state.write_toml(ens_nums, inp_traj)

    # update pwd
    if state.worker != state.workers:
        md_items.update({'pin': state.worker})
    if state.worker == state.workers:
        ens_nums_old = md_items.pop('ens_nums')
    md_items.update({'ens_nums': ens_nums})

    for ens_num, traj_inp in zip(ens_nums, inp_traj):
        state.ensembles[ens_num+1]['path_ensemble'].last_path = traj_inp
        md_items['ensembles'][ens_num+1] = state.ensembles[ens_num+1]

    # print state:
    if state.screen > 0 and np.mod(state.cstep, state.screen) == 0:
        if len(ens_nums) > 1 or ens_nums[0] == -1:
            move = 'sh'
        else:
            move = state.mc_moves[md_items['ens_nums'][0]+1]
        ens_p = ' '.join([f'00{ens_num+1}' for ens_num in md_items['ens_nums']])
        pat_p = ' '.join([str(i.path_number) for i in inp_traj])
        print('shooting', move, 'in ensembles:',
              ens_p, 'with paths:', pat_p, 
              'and worker:', md_items['pin'])

    # empty certain md_items:
    md_items['moves'] = []
    md_items['pnum_old'] = []

    # save pattern grid:
    if state.pattern and state.worker == state.workers:
        now0 = time.time()
        with open(state.pattern_file, 'a') as fp:
            for idx, ens_num in enumerate(ens_nums_old):
                fp.write(f"{ens_num+1}\t{state.time_keep[md_items['pin']]:.5f}\t" +
                         f"{now0:.5f}\t{md_items['pin']}\n")
        state.time_keep[md_items['pin']] = now0


def calc_cv_vector(path, interfaces, moves):
    path_max, _ = path.ordermax

    cv = []
    if len(interfaces) == 1:
        return (1. if interfaces[0] <= path_max else 0.,)

    for idx, intf_i in enumerate(interfaces[:-1]):
        if moves[idx+1] == 'wf':
            intfs = [interfaces[0], intf_i, interfaces[-1]]
            cv.append(compute_weight(path, intfs, moves[idx+1]))
        else:
            cv.append(1. if intf_i <= path_max else 0.)
    cv.append(0.)
    return(tuple(cv))