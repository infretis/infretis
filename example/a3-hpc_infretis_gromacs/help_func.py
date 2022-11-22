import os
import numpy as np
import time
from dask.distributed import get_worker
from pyretis.core.tis import shoot, wire_fencing
from pyretis.core.retis import retis_swap_zero # need to disable "add_path_data()"
from infretis import calc_cv_vector

def run_md(ens_num, input_traj, settings, ensembles, cycle, move, pin):
    interfaces = settings['simulation']['interfaces']
    worker = get_worker()
    pin0 = str(worker.name)
    start_time = time.time() 
    out = {'ensembles': [],                                                  
           'accepted_trajs': [],                                        
           'traj_vectors': [],
           'status': None,
           'pin': pin,
           'move': move,
           'pin0': [pin, pin0]}
    path_numbers_old = []
    for traj0 in input_traj:
        path_numbers_old.append(traj0.path_number)
    out['path_numbers_old'] = path_numbers_old
    
    if len(ens_num) == 1:
        start_cond = ensembles[ens_num[0]+1]['path_ensemble'].start_condition
        tis_settings = settings['ensemble'][ens_num[0]+1]['tis'] 
        
        if move == 'sh':
            accept, trial, status = shoot(ensembles[ens_num[0]+1],
                                          tis_settings,
                                          start_cond)
        else:
            accept, trial, status = wire_fencing(ensembles[ens_num[0]+1],
                                                 tis_settings,
                                                 start_cond)
        
        if accept:                                                                  
            out_traj = trial                                                        
        else:                                                                       
            out_traj = input_traj[0]
        if ens_num[0] < 0:
            interfaces = interfaces[0:1]
        cv_vector = calc_cv_vector(out_traj, interfaces, move)
                
        out['ensembles'] = list(ens_num)
        out['accepted_trajs'] = [out_traj]
        out['traj_vectors'] = [cv_vector]
        out['status'] = status
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
        for traj, iface in zip(out_traj, ifaces):
            out['traj_vectors'].append(calc_cv_vector(traj, iface, move))     
        out['status'] = status
    curr_time = time.time() 
    out['time'] = curr_time - start_time
    return out


def print_path_info(state, intfs=False):
    ens_no = len(state._trajs[:-1])
    ens_str = [f'{i:03.0f}' for i in range(ens_no)]
    state_dic = {}
    path_dic = {}
    ens_pwds = []
    locks = [traj0.path_number for traj0, lock0 in
             zip(state._trajs[:-1], state._locks[:-1]) if lock0]

    
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
    
    last_prob = True
    if type(state._last_prob) == type(None):
        state.prob
        last_prob = False
        
    # print current state
    p_num = [state_dic[ens]['path_number'] for ens in ens_str]
    print('===')
    print(' xx |\t', '\t'.join(['e'+i for i in ens_str]))
    print(' -- |     -----------------------------------')
    
    live_trajs = [traj.path_number for traj in state._trajs[:-1]] # state.live_paths()
    w_start = 0
    for live in live_trajs:
        if live not in locks:
            for weight in state._last_prob[w_start:-1]:
                w_start += 1
                if sum(weight) != 0:
                    print(f'p{live:02.0f} |\t',
                          '\t'.join([f'{j:.2f}' if j != 0 else '----' for j in weight[:-1]]))
                    break
        else:
            print(f'p{live:02.0f} |\t', '\t'.join(['----' for j in range(ens_no)]))

    print('===')
    if intfs:
        print(len(state._trajs[:-1]))
        for live in state._trajs[:-1]:
            print(live.path_number, calc_cv_vector(live, interfaces, 'sh'))
    print('===')
            
    if not last_prob:
        state._last_prob = None
        
def set_shooting(sim, ens, input_traj, pin, move):
    print('shooting', move, 'in ensembles:', ' '.join([f'00{ens_num+1}' for ens_num in ens]),
          'with paths:', ' '.join([str(trajj.path_number) for trajj in input_traj]),
          'and worker:', pin)

    # assign gromacs resurces to workers:
    for ens_num, traj_inp in zip(ens, input_traj):
        ens_num += 1
        # not_pin = '1' if pin == '0' else '0'
        # mdrun = sim.ensembles[ens_num]['engine'].mdrun.replace(not_pin, pin)
        # mdrun_c = sim.ensembles[ens_num]['engine'].mdrun_c.replace(not_pin, pin)
        # sim.ensembles[ens_num]['engine'].mdrun = mdrun
        # sim.ensembles[ens_num]['engine'].mdrun_c = mdrun_c
        sim.ensembles[ens_num]['path_ensemble'].last_path = traj_inp.copy()

def treat_output(output, state, sim, traj_num_dic, traj_num, size, save=False):
    ensembles = output['ensembles']
    out_trajs = output['accepted_trajs']
    traj_vectors = output['traj_vectors']
    status = output['status']
    path_numbers_old = output['path_numbers_old']
    pin = output['pin']
    move = output['move']
    pin0 = output['pin0']
    time_spent = output['time']
    
    # analyse and record worker data
    for ens_num, out_traj, traj_v, pn_old in zip(ensembles,
                                                 out_trajs,
                                                 traj_vectors,
                                                 path_numbers_old):
        # if path is new: number and save the path:
        if out_traj.path_number == None or status == 'ACC':
            out_traj.path_number = traj_num
            ens_save_idx = traj_num_dic[pn_old]['ens_idx']
            traj_num_dic[traj_num] = {'weight': np.zeros(size+1),
                                      'adress': set(kk.particles.config[0].split('salt')[-1] 
                                                    for kk in out_traj.phasepoints),
                                      'ens_idx': ens_save_idx}
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
                for task in sim.output_tasks:
                    task.output(result)
            
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
    print('cobra0', pin0)
    return traj_num_dic, traj_num, pin
