import os
import numpy as np
import time
import tomli_w
from pyretis.core.tis import select_shoot
from pyretis.core.retis import retis_swap_zero
from pyretis.setup import create_simulation
from pyretis.inout.settings import parse_settings_file
from infretis import calc_cv_vector, REPEX_state
from dask.distributed import dask, Client, as_completed
dask.config.config['work-stealing'] = False


def run_md(md_items):
    start_time = time.time() 

    pyretis_mc(md_items)

    # out['traj_vectors'] = []
    md_items['traj_vectors'] = []
    for ens_num, ifaces in zip(md_items['ens_nums'], md_items['interfaces']):
        traj = md_items['ensembles'][ens_num+1]['path_ensemble'].last_path
        cv_vector = calc_cv_vector(traj, ifaces, md_items['mc_moves'])
        md_items['traj_vectors'].append(cv_vector)

    md_items['time'] = time.time() - start_time
    return md_items


def pyretis_mc(md_items):
    ens_nums = md_items['ens_nums']
    ensembles = md_items['ensembles']
    settings = md_items['settings']
    interfaces = settings['simulation']['interfaces']

    # print(ensembles[2]['rgen'].rand()[0],
    #       ensembles[2]['path_ensemble'].rgen.rand()[0],
    #       ensembles[2]['path_ensemble'].last_path.rgen.rand()[0], ens_nums[0]+1)

    pnum_old = []
    # out = {'pnum_old': []}
    for i in ens_nums:
        pnum_old.append(ensembles[i+1]['path_ensemble'].last_path.path_number)

    if len(ens_nums) == 1:
        start_cond = ensembles[ens_nums[0]+1]['path_ensemble'].start_condition
        tis_settings = settings['ensemble'][ens_nums[0]+1]['tis'] 

        accept, trial, status = select_shoot(ensembles[ens_nums[0]+1],
                                             tis_settings,
                                             start_cond)

        if accept: 
            ensembles[ens_nums[0]+1]['path_ensemble'].last_path = trial
        interfaces =  [interfaces] if ens_nums[0] >= 0 else [interfaces[0:1]]
    else:
        ensembles_l = [ensembles[i+1] for i in ens_nums]
        accept, trial, status = retis_swap_zero(ensembles_l, 
                                                settings,
                                                0)
        if accept: 
            ensembles[0]['path_ensemble'].last_path = trial[0]
            ensembles[1]['path_ensemble'].last_path = trial[1]

        interfaces = [interfaces[0:1], interfaces]

    md_items.update({'status': status, 'interfaces': interfaces, 'pnum_old': pnum_old})


def treat_output(state, md_items, save=False):
    traj_num_dic = state.traj_num_dic
    traj_num = state.config['current']['traj_num']
    size = state.config['current']['size']

    ensembles = md_items['ensembles']
    pn_archive = []
    pn_new = []

    # analyse and record worker data
    for ens_num, traj_v, pn_old in zip(md_items['ens_nums'],
                                       md_items['traj_vectors'],
                                       md_items['pnum_old']):
        # if path is new: number and save the path:
        out_traj = ensembles[ens_num+1]['path_ensemble'].last_path
        if out_traj.path_number == None or md_items['status'] == 'ACC':
            out_traj.path_number = traj_num
            pn_new.append(traj_num)
            pn_archive.append(pn_old)
            ens_save_idx = traj_num_dic[pn_old]['ens_idx']
            # print('bear',pn_old, traj_v)
            traj_num_dic[traj_num] = {'frac': np.zeros(size+1),
                                      'ens_idx': ens_save_idx,
                                      'max_op': out_traj.ordermax,
                                      'length': out_traj.length,
                                      'traj_v': traj_v}
            traj_num += 1
            # state.ensembles[ens_save_idx]['path_ensemble'].store_path(out_traj)

            # flamingo0 = [state.ensembles[kk]['path_ensemble'].last_path.path_number for kk in [0, 1, 2]]
            # print('flamingo0', flamingo0)
            # print('flamingo1', state.ensembles[ens_save_idx]['path_ensemble'].last_path.path_number, pn_old)
            # if state.ensembles[ens_save_idx]['path_ensemble'].last_path.path_number != pn_old:
            #     exit('ape1')
            # if len(set(flamingo0)) != len(flamingo0):
            #     exit('ape2')
            
            # cycle = {'step': traj_num -1 , 'endcycle': 10, 'startcycle': 0, 'stepno': 10, 'steps': 10}
            # result = {f'status-{ens_num+1}': 'ACC', 'cycle': cycle, f'path-{ens_num+1}':  out_traj,
            #           f'accept-{ens_num+1}': True, f'move-{ens_num+1}': 'sh', 
            #           'all-2': {'ensemble_number': ens_num+1, 'mc-move': 'sh',
            #                     'status': 'ACC', 'trial': out_traj, 'accept': True},
            #           f'pathensemble-{ens_num+1}': ensembles[0]['path_ensemble']}
            # 
            # if save: # NB! Saving can take some time..
            #     # flipppa = time.time() 
            #     # for task in sim.output_tasks:
            #     #     task.output(result)
            #     print('saving path time:', time.time() - flipppa)
        else:
            pn_new.append(out_traj.path_number)
            
        state.add_traj(ens_num, out_traj, traj_v)
        state.ensembles[ens_num+1] = md_items['ensembles'][ens_num+1]
        md_items['ensembles'].pop(ens_num+1)
        
    # record weights 
    live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
    traj_lock = [traj0.path_number for traj0, lock0 in
                 zip(state._trajs[:-1], state._locks[:-1]) if lock0]
    w_start = 0
    last_prob = True
    # print('all:', [i.path_number for i in state._trajs[:-1]])
    # print(state._last_prob)

    # print('live:', live_trajs)
    # print('locked:', traj_lock)
    for idx, live in enumerate(live_trajs):
        if live not in traj_lock:
            traj_num_dic[live]['frac'] += state._last_prob[:-1][idx, :]
            # for frac in state._last_prob[w_start:-1]:
            #     w_start += 1
            #     print('shark', f'p{live}', frac)
            #     traj_num_dic[live]['frac'] += frac
            #     break
    if not last_prob:
        state._last_prob = None

    # print information to screen
    # print('shooted', 'sh', 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in md_items['ens_nums']]),
    #       'with paths:', ' '.join([str(pn_old) for pn_old in md_items['pnum_old']]), '->', 
    #       ' '.join([str(pn0) for pn0 in pn_new]),
    #       'with status:', md_items['status'], 'and worker:', md_items['pin'], f"total time: {md_items['time']:.2f}")

    state.config['current']['traj_num'] = traj_num

    # print analyzed output
    if'ACC' == md_items['status']:
        write_to_pathens(state, pn_archive)


def write_to_pathens(state, pn_archive):
    traj_num_dic = state.traj_num_dic
    size = state.n

    with open('infretis_data.txt', 'a') as fp:
        for pn in pn_archive:
            string = ''
            string += f'\t{pn:3.0f}\t'
            string += f"{traj_num_dic[pn]['length']:5.0f}" + '\t'
            string += f"{traj_num_dic[pn]['max_op'][0]:8.5f}" + '\t'
            frac = []
            weight = []
            # for pn in pn_archive:
            # print('babbi', pn, traj_num_dic[pn]['traj_v'], traj_num_dic[pn]['length'],traj_num_dic[pn]['max_op'] )
            if len(traj_num_dic[pn]['traj_v']) == 1:
                f0 = traj_num_dic[pn]['frac'][0]
                w0 = traj_num_dic[pn]['traj_v'][0]
                frac.append('----' if f0 == 0.0 else f"{f0:5.3f}")
                if weight == 0:
                    print('tortoise', frac, weight)
                    exit('fish')
                
                weight.append('----' if f0 == 0.0 else f"{w0:5.0f}")
                frac += ['----']*(size-2)
                weight += ['----']*(size-2)
            else:
                frac.append('----')
                weight.append(f'----')
                for w0, f0 in zip(traj_num_dic[pn]['traj_v'][:-1],
                                  traj_num_dic[pn]['frac'][1:-1]):
                    # frac.append(f"{f0:02.3f}")
                    frac.append('----' if f0 == 0.0 else f"{f0:5.3f}")
                    weight.append('----' if f0 == 0.0 else f"{w0:5.0f}")
            # print('babbi 1', string, weight, 'whada')
            fp.write(string + '\t'.join(frac) + '\t' + '\t'.join(weight) + '\t\n')
            traj_num_dic.pop(pn)


def setup_internal(config):
    # setup pyretis
    inp = config['simulation']['pyretis_inp']
    sim_settings = parse_settings_file(inp)
    interfaces = sim_settings['simulation']['interfaces']
    size = len(interfaces)

    # check if we restart or not 
    if 'current' not in config:
        config['current'] = {}
        config['current']['step'] = 0
        config['current']['active'] = []
        config['current']['locked'] = []
        config['current']['dic'] = []
        with open('infretis_data.txt', 'w') as fp:
            fp.write('# ' + '='*(34+8*size)+ '\n')
            ens_str = '\t'.join([f'{i:03.0f}' for i in range(size)])
            fp.write('# ' + f'\txxx\tlen\tmax OP\t\t{ens_str}\n')
            fp.write('# ' + '='*(34+8*size)+ '\n')
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
        traj_v = calc_cv_vector(path, interfaces, state.mc_moves)
        state.add_traj(ens=i, traj=path, valid=traj_v, count=False)
        # print(path.path_number, path.ordermax[0], traj_v)
        traj_num_dic[path.path_number] = {'frac': np.zeros(size+1),
                                          'ens_idx': i + 1,
                                          'max_op': path.ordermax,
                                          'length': path.length,
                                          'traj_v': traj_v}
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    traj_v = (1,)
    state.add_traj(ens=-1, traj=path, valid=traj_v, count=False)
    traj_num_dic[path.path_number] = {'frac': np.zeros(size+1),
                                      'ens_idx': 0,
                                      'max_op': path.ordermax,
                                      'length': path.length, 
                                      'traj_v': traj_v}

    if 'traj_num' not in config['current'].keys():
        config['current']['traj_num'] = max(config['current']['active']) + 1
    state.config = config
    state.ensembles = {i: sim.ensembles[i] for i in range(len(sim.ensembles))}
    md_items = {'settings': sim.settings,
                'mc_moves': state.mc_moves,
                'ensembles': {}}

    return md_items, state


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
            print('warning! the state_paths does' + \
                  'not correspond to the path_pwds!')
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


def prepare_pyretis(state, md_items, input_traj, printing=False):

    # pwd_checker
    # pwd_checker(state)

    # write toml:
    # write_toml(state, ens, input_traj)

    # update pwd
    for ens_num, traj_inp in zip(md_items['ens_nums'], input_traj):
        # md_items['ensembles'][ens_num]['path_ensemble'].last_path = traj_inp.copy()
        state.ensembles[ens_num+1]['path_ensemble'].last_path = traj_inp.copy()
        md_items['ensembles'][ens_num+1] = state.ensembles[ens_num+1]

    # print state:
    if printing:
        # state.print_state()
        if len(md_items['ens_nums']) > 1 or md_items['ens_nums'][0] == -1:
            move = 'sh'
        else:
            move = state.mc_moves[md_items['ens_nums'][0]+1]
        ens_p = ' '.join([f'00{ens_num+1}' for ens_num in md_items['ens_nums']])
        pat_p = ' '.join([str(i.path_number) for i in input_traj])
        print('shooting', move, 'in ensembles:',
              ens_p, 'with paths:', pat_p, 
              'and worker:', md_items['pin'])
