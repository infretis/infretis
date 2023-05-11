import os
import numpy as np
import time
import tomli
import logging
from datetime import datetime

from infretis.core.retis import retis_swap_zero
# from infretis.classes.formatter import PathStorage, get_log_formatter
# from infretis.core.tis import select_shoot
from infretis.core.common import write_ensemble_restart, make_dirs
from infretis.core.tis import compute_weight
from infretis.inf_core import REPEX_state

from infretis.newc.formats.formatter import PathStorage, get_log_formatter
from infretis.newc.orderparameter import create_orderparameters
from infretis.newc.ensemble import create_ensembles
from infretis.newc.engine import create_engines
from infretis.newc.path import load_paths

from infretis.newf.tis import select_shoot

from dask.distributed import dask, Client, as_completed, get_worker
dask.config.set({'distributed.scheduler.work-stealing': False})
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
# Define a console logger. This will log to sys.stderr:
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
console.setFormatter(get_log_formatter(logging.WARNING))
logger.addHandler(console)
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"

def run_md2(md_items):
    md_items['wmd_start'] = time.time()
    md_items['ens_nums'] = list(md_items['picked'].keys())
    md_items['pnum_old'] = []
    for key in md_items['picked'].keys():
        md_items['pnum_old'].append(md_items['picked'][key]['traj'].path_number)
    # store info before md run
    picked = md_items['picked']
    # ..
    # ..

    # set the worker hw
    if 'md_worker' in md_items:
        base = md_items['md_worker']
        mdrun = base + ' -s {} -deffnm {} -c {}'
        mdrun_c = base + ' -s {} -cpi {} -append -deffnm {} -c {}'
        # set the worker folder
        w_folder = os.path.join(os.getcwd(), f"worker{md_items['pin']}")
        make_dirs(w_folder)
        for ens in picked.keys():
            picked[ens]['engine'].mdrun = mdrun
            picked[ens]['engine'].mdrun_c = mdrun_c
            picked[ens]['engine'].exe_dir = w_folder

    # perform the hw move:
    accept, trials, status = select_shoot(picked)

    for trial, ens_num in zip(trials, picked.keys()):
        md_items['moves'].append(md_items['mc_moves'][ens_num+1])
        md_items['trial_len'].append(trial.length)
        md_items['trial_op'].append((trial.ordermin[0], trial.ordermax[0]))
        md_items['generated'].append(trial.generated)
        if status == 'ACC':
            trial.weights = calc_cv_vector(trial,
                                           md_items['interfaces'],
                                           md_items['mc_moves'])
            picked[ens_num]['traj'] = trial
            # md_items['out_trajs'].append(trial)

    md_items.update({'status': status,
                     'wmd_end': time.time()})

    return md_items


def run_md(md_items):
    md_items['wmd_start'] = time.time()
    ens_nums = md_items['ens_nums']
    ensembles = md_items['ensembles']
    interfaces = md_items['interfaces']
    start_time = datetime.now()
    exit('down bad')
    logger.info(start_time.strftime(DATE_FORMAT))

    if len(ens_nums) == 1:
        pnum = ensembles[ens_nums[0]+1]['path_ensemble'].last_path.path_number
        enum = f"{ens_nums[0]+1:03.0f}"
        move = md_items['mc_moves'][ens_nums[0]+1]
        logger.info(f"Shooting {move} in ensemble: {enum}"\
                    f" with path: {pnum} and worker: {md_items['pin']}")
        start_cond = ensembles[ens_nums[0]+1]['path_ensemble'].start_condition
        ensembles[ens_nums[0]+1]['engine'].clean_up()
        accept, trials, status = select_shoot(ensembles[ens_nums[0]+1],
                                              md_items['settings'],
                                              start_cond)
        trials = [trials]

    else:
        ensembles_l = [ensembles[i+1] for i in ens_nums]
        pnums = [ensembles_l[0]['path_ensemble'].last_path.path_number,
                 ensembles_l[1]['path_ensemble'].last_path.path_number]
        logger.info(f"Shooting sh sh in ensembles: 000 001"\
                    f" with paths: {pnums} and worker: {md_items['pin']}")
        ensembles_l[0]['engine'].clean_up()
        ensembles_l[1]['engine'].clean_up()
        accept, trials, status = retis_swap_zero(ensembles_l, md_items['settings'], 0)

    for trial, ens_num, ifaces in zip(trials, ens_nums, interfaces):
        md_items['moves'].append(md_items['mc_moves'][ens_num+1])
        md_items['pnum_old'].append(ensembles[ens_num+1]['path_ensemble'].last_path.path_number)
        md_items['trial_len'].append(trial.length)
        md_items['trial_op'].append((trial.ordermin[0], trial.ordermax[0]))
        md_items['generated'] = trial.generated
        logger.info(f'Move finished with trial path lenght of {trial.length}')
        log_mdlogs(f'{ens_num+1:03}/generate/')
        if status == 'ACC':
            trial.traj_v = calc_cv_vector(trial, ifaces, md_items['mc_moves'])
            ensembles[ens_num+1]['path_ensemble'].last_path = trial
            logger.info('The move was accepted!')
        else:
            logger.info('The move was rejected!')

    end_time = datetime.now()
    delta_time = end_time - start_time
    logger.info(end_time.strftime(DATE_FORMAT) +
                f', {delta_time.days} days {delta_time.seconds} seconds' + '\n')
    md_items.update({'status': status,
                     'interfaces': interfaces,
                     'wmd_end': time.time()})
    return md_items

def log_mdlogs(inp):
    logs = [log for log in os.listdir(inp) if 'log' in log]
    speed = []
    for log in logs:
        with open(os.path.join(inp, log), 'r') as read:
            for line in read:
                if 'Performance' in line:
                    logger.info(log + ' '+ line.rstrip().split()[1] + ' ns/day')

def treat_output(state, md_items):
    traj_num_dic = state.traj_num_dic
    traj_num = state.config['current']['traj_num']
    pn_news = []
    md_items['md_end'] = time.time()
    picked = md_items['picked']

    # analyse and record worker data
    # for ens_num, pn_old, out_traj  in zip(md_items['ens_nums'],
    #                            md_items['pnum_old']):
    for ens_num in picked.keys():
        pn_old = picked[ens_num]['pn_old']
        out_traj = picked[ens_num]['traj']
        state.ensembles[ens_num+1] = picked[ens_num]['ens']

        for idx, lock in enumerate(state.locked):
            if str(pn_old) in lock[1]:
                state.locked.pop(idx)

        # if path is new: number and save the path:
        if out_traj.path_number == None or md_items['status'] == 'ACC':
            # move to accept:
            ens_save_idx = traj_num_dic[pn_old]['ens_save_idx']
            out_traj.path_number = traj_num

            # NB! Saving can take some time..
            # add setting where we save .trr file or not (we always save restart)
            # if state.config['output']['store_paths']:
            make_dirs(f'./trajs/{out_traj.path_number}')
            data = {'path': out_traj,
                    'dir': os.path.join(os.getcwd(), state.config['simulation']['load_dir'])}
            # state.pstore.output(state.cstep, out_traj)
            out_traj = state.pstore.output(state.cstep, data)
            traj_num_dic[traj_num] = {'frac': np.zeros(state.n, dtype="float128"),
                                      'max_op': out_traj.ordermax,
                                      'length': out_traj.length,
                                      'weights': out_traj.weights,
                                      'adress': out_traj.adress,
                                      'ens_save_idx': ens_save_idx}
            traj_num += 1

            if state.config['output'].get('delete_old', False) and pn_old > state.n - 2:
                # if pn is larger than ensemble number ...
                for adress in traj_num_dic[pn_old]['adress']:
                    ##### Make checker? so it doesn't do anything super yabai
                    os.remove(adress)

        if state.config['output']['store_paths']:
            # save ens-path_ens-rgen (not used) and ens-path
            # write_ensemble_restart(state.ensembles[ens_num+1], state.pyretis_settings, save='path')
            # save ens-rgen, ens-engine-rgen
            write_ensemble_restart(state.ensembles[ens_num+1], state.config, save=f'e{ens_num+1}')

        pn_news.append(out_traj.path_number)
        state.add_traj(ens_num, out_traj, out_traj.weights)
        
    # record weights 
    locked_trajs = state.locked_paths()
    if state._last_prob is None:
        state.prob
    for idx, live in enumerate(state.live_paths()):
        if live not in locked_trajs:
            traj_num_dic[live]['frac'] += state._last_prob[:-1][idx, :]

    # write succ data to infretis_data.txt
    if md_items['status'] == 'ACC':
        write_to_pathens(state, md_items['pnum_old'])

    state.sort_trajstate()
    state.config['current']['traj_num'] = traj_num
    state.cworker = md_items['pin']
    state.print_shooted(md_items, pn_news)
    # save for possible restart
    state.save_rng()
    state.write_toml()

    return md_items

def setup_internal(input_file):

    # setup logger
    fileh = logging.FileHandler('sim.log', mode='a')
    log_levl = getattr(logging, 'info'.upper(),
                       logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)

    # read input_file.toml
    with open(input_file, mode="rb") as f:
        config = tomli.load(f)
    # if input_file.toml != restart.toml and restart.toml exist:
    if os.path.isfile('restart.toml') and \
       'restart.toml' not in input_file and \
       config['output']['store_paths']:
        with open('./restart.toml', mode="rb") as f:
            restart = tomli.load(f)
        # check if they are similar to use restart over input_file
        equal = True
        for key in ['dask', 'simulation', 'output']:
            if config[key] != restart[key]:
                equal = False
                break
        if equal:
            for act in restart['current']['active']:
                store_p = os.path.join('trajs', str(act), 'ensemble.restart')
                if not os.path.isfile(store_p):
                    equal = False
                    break
        if equal:
            restart['current']['restarted_from'] = restart['current']['cstep']
            config = restart
            logger.info('We use restart.toml instead.')

    # parse retis.rst
    interfaces = config['simulation']['interfaces']
    size = len(interfaces)

    # setup config
    endsim = setup_config(config, size)
    if endsim:
        return None, None, None

    # setup pyretis and infretis
    state = setup_repex(config)
    state.ensembles = create_ensembles(config)
    state.engines = create_engines(config)
    create_orderparameters(state.engines, config)
    paths = load_paths(config)

    # initiate by adding paths from retis sim to repex
    traj_num_dic = state.traj_num_dic
    for i in range(size-1):
        # we add all the i+ paths.
        paths[i+1].weights= calc_cv_vector(paths[i+1], interfaces, state.mc_moves)
        state.add_traj(ens=i, traj=paths[i+1], valid=paths[i+1].weights, count=False)
        pnum = paths[i+1].path_number
        frac = config['current']['frac'].get(str(pnum), np.zeros(size+1))
        traj_num_dic[pnum] = {'ens_save_idx': i + 1,
                              'max_op': paths[i+1].ordermax,
                              'length': paths[i+1].length,
                              'adress': paths[i+1].adress,
                              'weights': paths[i+1].weights,
                              'frac': np.array(frac, dtype='float128')}
    
    # add minus path:
    paths[0].weights = (1.,)
    pnum = paths[0].path_number
    state.add_traj(ens=-1, traj=paths[0], valid=paths[0].weights, count=False)
    frac = config['current']['frac'].get(str(pnum), np.zeros(size+1))
    traj_num_dic[pnum]= {'ens_save_idx': 0,
                         'max_op': paths[0].ordermax,
                         'length': paths[0].length,
                         'weights': paths[0].weights,
                         'adress': paths[0].adress,
                         'frac': np.array(frac, dtype='float128')}

    # state.ensembles = {i: sim.ensembles[i] for i in range(len(sim.ensembles))}
    # sim.settings['initial-path']['load_folder'] = 'trajs'
    # state.pyretis_settings = sim.settings
    md_items = {'mc_moves': state.mc_moves,
                'interfaces': interfaces}

    if state.pattern_file:
        writemode = 'a' if 'restarted_from' in state.config['current'] else 'w'
        with open(state.pattern_file, writemode) as fp:
            fp.write(f"# Worker\tMD_start [s]\t\twMD_start [s]\twMD_end"
                     + f"[s]\tMD_end [s]\t Dask_end [s]\tEnsembles\t{state.start_time}\n")

    if None in (md_items, state):
        exit('None in md_items, state')

    return md_items, state

def setup_dask(state):

    client = Client(n_workers=state.workers)
    for module in state.config['dask'].get('files', []):
        client.upload_file(module)
    futures = as_completed(None, with_results=True)
    # create worker logs
    client.run(set_logger)
    return client, futures

def pwd_checker(state):
    all_good = True
    ens_str = [f'{i:03.0f}' for i in range(state.n-1)]

    tot = []
    for path_temp in state._trajs[:-1]:
        tot += list(path_temp.adress)
    for ppath in tot:
        if not os.path.isfile(ppath):
            print('warning! this path does not exist', ppath)
            all_good = False

    return all_good

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

def setup_config(config, size):

    data_dir = config['output']['data_dir']
    data_file = os.path.join(data_dir, 'infretis_data.txt')
    config['output']['data_file'] = data_file

    # check if we restart or not
    if 'current' not in config:
        config['current'] = {'traj_num': size, 'cstep': 0,
                             'active': list(range(size)),
                             'locked': [], 'size': size, 'frac': {}}
        # write/overwrite infretis_data.txt
        with open(data_file, 'w') as fp:
            fp.write('# ' + '='*(34+8*size)+ '\n')
            ens_str = '\t'.join([f'{i:03.0f}' for i in range(size)])
            fp.write('# ' + f'\txxx\tlen\tmax OP\t\t{ens_str}\n')
            fp.write('# ' + '='*(34+8*size)+ '\n')
    else:
        config['current']['restarted_from'] = config['current']['cstep']
        if config['current']['cstep'] == config['simulation']['steps']:
            print('current step and total steps are equal so we exit ',
                  'without doing anything.')
            return True
    return False

def setup_repex(config):
    state = REPEX_state(n=config['current']['size'],
                        workers=config['dask']['workers'],
                        minus=True)
    state.tsteps = config['simulation']['steps']
    state.cstep = config['current']['cstep']
    state.screen = config['output']['screen']
    state.mc_moves = config['simulation']['shooting_moves']

    state.config = config
    if config['output'].get('pattern', False):
        state.pattern_file = os.path.join('pattern.txt')
    state.data_file = config['output']['data_file']
    if 'restarted_from' in config['current']:
        state.set_rng()
    state.locked0 = list(config['current'].get('locked', []))
    state.locked = list(config['current'].get('locked', []))

    pstore = PathStorage()
    state.pstore = pstore

    return state

def set_logger():
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
            if len(traj_num_dic[pn]['weights']) == 1:
                f0 = traj_num_dic[pn]['frac'][0]
                w0 = traj_num_dic[pn]['weights'][0]
                frac.append('----' if f0 == 0.0 else str(f0))
                if weight == 0:
                    print('tortoise', frac, weight)
                    exit('fish')
                weight.append('----' if f0 == 0.0 else str(w0))
                frac += ['----']*(size-2)
                weight += ['----']*(size-2)
            else:
                frac.append('----')
                weight.append(f'----')
                for w0, f0 in zip(traj_num_dic[pn]['weights'][:-1],
                                  traj_num_dic[pn]['frac'][1:-1]):
                    frac.append('----' if f0 == 0.0 else str(f0))
                    weight.append('----' if f0 == 0.0 else str(w0))
            fp.write(string + '\t'.join(frac) + '\t' + '\t'.join(weight) + '\t\n')
            traj_num_dic.pop(pn)
