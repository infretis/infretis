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

    out = pyretis_mc(md_items)

    out['traj_vectors'] = []
    for traj, iface in zip(out['accepted_trajs'], out['interfaces']):
        cv_vector = calc_cv_vector(traj, iface, md_items['mc_moves'])
        out['traj_vectors'].append(cv_vector)

    out['time'] = time.time() - start_time
    return out


def pyretis_mc(md_items):
    ens_num = md_items['ens']
    input_traj = md_items['input_traj']
    settings = md_items['sim'].settings
    ensembles = md_items['sim'].ensembles
    interfaces = settings['simulation']['interfaces']
    out = {'pnum_old': []}
    for traj0 in input_traj:
        out['pnum_old'].append(traj0.path_number)

    if len(ens_num) == 1:
        start_cond = ensembles[ens_num[0]+1]['path_ensemble'].start_condition
        tis_settings = settings['ensemble'][ens_num[0]+1]['tis'] 
        move = tis_settings.get('shooting_move', 'sh')
        accept, trial, status = select_shoot(ensembles[ens_num[0]+1],
                                             tis_settings,
                                             start_cond)

        out_traj = [trial if accept else input_traj[0]]
        swap = False
        if ens_num[0] < 0:
            interfaces = [interfaces[0:1]]
        else: 
            interfaces = [interfaces]
    else:
        accept, trial, status = retis_swap_zero(ensembles, 
                                                settings,
                                                0)
        out_traj = trial if accept else input_traj
        swap = True
        move = 'sh'
        interfaces= [[interfaces[0:1]], interfaces]

    out.update({'move': move, 'accepted_trajs': out_traj,
                'status': status, 'interfaces': interfaces,
                'ensembles': list(ens_num), 'pin': md_items['pin']})

    return out


def treat_output(output, state, sim, save=False):
    traj_num_dic = state.traj_num_dic
    traj_num = state.config['current']['traj_num']
    size = state.config['current']['size']

    ensembles = output['ensembles']
    out_trajs = output['accepted_trajs']
    traj_vectors = output['traj_vectors']
    status = output['status']
    pnum_old = output['pnum_old']
    pn_archive = []
    pin = output['pin']
    move = output['move']
    time_spent = output['time']
    
    # analyse and record worker data
    for ens_num, out_traj, traj_v, pn_old in zip(ensembles,
                                                 out_trajs,
                                                 traj_vectors,
                                                 pnum_old):
        # if path is new: number and save the path:
        if out_traj.path_number == None or status == 'ACC':
            out_traj.path_number = traj_num
            pn_archive.append(pn_old)
            ens_save_idx = traj_num_dic[pn_old]['ens_idx']
            traj_num_dic[traj_num] = {'frac': np.zeros(size+1),
                                      'adress': set(kk.particles.config[0].split('salt')[-1] 
                                                    for kk in out_traj.phasepoints),
                                      'ens_idx': ens_save_idx,
                                      'max_op': out_traj.ordermax,
                                      'length': out_traj.length,
                                      'traj_v': traj_v}
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
            
        print('gori0', ensembles, move, traj_v)
        state.add_traj(ens_num, out_traj, traj_v)
        
        # record weights 
        live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
        traj_lock = [traj0.path_number for traj0, lock0 in
                     zip(state._trajs[:-1], state._locks[:-1]) if lock0]
        w_start = 0
        for live in live_trajs:
            if live not in traj_lock:
                for frac in state._last_prob[w_start:-1]:
                    w_start += 1
                    if sum(frac) != 0:
                        traj_num_dic[live]['frac'] += frac
                        break
    
    # print information to screen
    print('shooted', move, 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in ensembles]),
          'with paths:', ' '.join([str(pn_old) for pn_old in pnum_old]), '->', 
          ' '.join([str(trajj.path_number) for trajj in out_trajs]),
          'with status:', status, 'and worker:', pin, f'total time: {time_spent:.2f}')

    state.config['current']['traj_num'] = traj_num

    # print analyzed output
    if'ACC' == status:
        write_to_pathens(state, pn_archive)

    return pin


def write_to_pathens(state, pn_archive):
    traj_num_dic = state.traj_num_dic
    size = state.n

    with open('infretis_data.txt', 'a') as fp:
        for pn in pn_archive:
            string = ''
            string += f'\t{pn:03.0f}\t'
            string += f"{traj_num_dic[pn]['length']:05.0f}" + '\t'
            string += f"{traj_num_dic[pn]['max_op'][0]:05.5f}" + '\t\t'
            frac = []
            weight = []
            # for pn in pn_archive:
            if len(traj_num_dic[pn]['traj_v']) == 1:
                frac.append(f"{traj_num_dic[pn]['frac'][0]:02.3f}")
                weight.append(f"{traj_num_dic[pn]['traj_v'][0]:02.0f}")
                frac += ['----']*(size-2)
                weight += ['01']*(size-2)
            else:
                frac.append('----')
                weight.append(f'{1:02.0f}')
                for w0, f0 in zip(traj_num_dic[pn]['traj_v'][:-1], 
                                  traj_num_dic[pn]['frac'][1:-1]):
                    frac.append(f"{f0:02.3f}")
                    weight.append(f"{w0:02.0f}")
            # for item0 in traj_num_dic[pn]['weight'][:-1]:
            #     w_str = f'{item0:02.2f}' if item0 != 0.0 else '----'
            #     weight.append(w_str)
            print(frac)
            print(weight)
            fp.write(string + '\t'.join(frac) + '\t' + '\t'.join(weight) + '\t\n')


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
        traj_num_dic[path.path_number] = {'frac': np.zeros(size+1),
                                          'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                         for kk in path.phasepoints),
                                          'ens_idx': i + 1,
                                          'max_op': path.ordermax,
                                          'length': path.length,
                                          'traj_v': np.array(list(traj_v) + [0.])}
    
    # add minus path:
    path = sim.ensembles[0]['path_ensemble'].last_path
    state.add_traj(ens=-1, traj=path, valid=(1,), count=False)
    traj_num_dic[path.path_number] = {'frac': np.zeros(size+1),
                                      'adress':  set(kk.particles.config[0].split('salt')[-1]
                                                     for kk in path.phasepoints),
                                      'ens_idx': 0,
                                      'max_op': path.ordermax,
                                      'length': path.length, 
                                      'traj_v': (1.,)}

    if 'traj_num' not in config['current'].keys():
        config['current']['traj_num'] = max(config['current']['active']) + 1
    state.config = config
    return {'sim': sim, 'mc_moves': state.mc_moves}, state


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


def prepare_pyretis(state, md_items, printing=False):

    # pwd_checker
    pwd_checker(state)

    # write toml:
    ens, input_traj = md_items['ens'], md_items['input_traj']
    write_toml(state, ens, input_traj)

    # update pwd
    for ens_num, traj_inp in zip(ens, input_traj):
        ens_num += 1
        md_items['sim'].ensembles[ens_num]['path_ensemble'].last_path = traj_inp.copy()

    # print state:
    if printing:
        state.print_state()
        if len(ens) > 1 or ens[0] == -1:
            move = 'sh'
        else:
            move = state.mc_moves[ens[0]+1]
        ens_p = ' '.join([f'00{ens_num+1}' for ens_num in ens])
        pat_p = ' '.join([str(i.path_number) for i in input_traj])
        print('shooting', move, 'in ensembles:',
              ens_p, 'with paths:', pat_p, 
              'and worker:', md_items['pin'])
