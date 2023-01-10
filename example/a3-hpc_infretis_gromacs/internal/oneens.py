import matplotlib.pyplot as plt
from pyretis.core.tis import select_shoot
from pyretis.core.retis import retis_swap_zero
from pyretis.setup import create_simulation
from pyretis.inout.settings import parse_settings_file
from dask.distributed import Client, as_completed


def setup(inp):

    sim_settings = parse_settings_file(inp)
    sim = create_simulation(sim_settings)
    sim.set_up_output(sim_settings)
    sim.initiate(sim_settings)
    return sim



def rand_ens(ensemble):
    print(ensemble['rgen'].rand())


def run_loop(inp, ens, cap=10):
    sim = setup(inp)
    settings = sim.settings
    ensemble = sim.ensembles[ens]
    tis_settings = settings['ensemble'][ens]['tis']
    start_cond = ensemble['path_ensemble'].start_condition

    max_op = []
    it = -1
    retur = {'ens': [], 'path_ens': [], 'path': []}
    for i in range(cap):
        accept, trial, status = select_shoot(ensemble,
                                             tis_settings,
                                             start_cond)
        if accept:
            ensemble['path_ensemble'].last_path = trial
            max_op.append(ensemble['path_ensemble'].last_path.ordermax[0])
            it+=1
        retur['ens'].append(ensemble['rgen'].rand()[0])
        retur['path_ens'].append(ensemble['path_ensemble'].rgen.rand()[0])
        retur['path'].append(ensemble['path_ensemble'].last_path.rgen.rand()[0])
    return retur


def ens_check(inp, ens, cap=10000):
    sim = setup(inp)
    settings = sim.settings
    tis_settings = settings['ensemble'][ens]['tis']
    md_items = {'sim': sim}
    ensembles = md_items['sim'].ensembles

    for i in range(10):
        rand_ens(ensembles[ens])


def shoot(tis_settings, start_cond, dic):
    ensemble = dic['ensemble']

    accept, trial, status = select_shoot(ensemble,
                                         tis_settings,
                                         start_cond)

    retur = dic['retur']

    if accept:
        dic['cnt'] += 1
        ensemble['path_ensemble'].last_path = trial
    # else: 
    #     dic['saved_traj'].rgen = trial.rgen

    # dic['ens_rgen'] = ensemble['rgen'] 
    # dic['pat_rgen'] = ensemble['path_ensemble'].rgen 
    retur['ens'].append(ensemble['rgen'].rand()[0])
    retur['path_ens'].append(ensemble['path_ensemble'].rgen.rand()[0])
    retur['path'].append(ensemble['path_ensemble'].last_path.rgen.rand()[0])
    return dic
    # yo = {'ensemble': ensemble}
    # return yo


def dask_check(inp, ens, cap=10000):
    if __name__ == "__main__":

        client = Client(n_workers=1)
        futures = as_completed(None, with_results=True)
        sim = setup(inp)
        ensemble = sim.ensembles[ens]
        tis_settings = sim.settings['ensemble'][ens]['tis']
        start_cond = ensemble['path_ensemble'].start_condition

        # dic = {'cnt': 0, 'saved_traj': ensemble['path_ensemble'].last_path}
        dic = {'cnt': 0}
        dic['retur'] = {'ens': [], 'path_ens': [], 'path': []}
        dic['ensemble'] = ensemble

        for i in range(10):
            j = client.submit(shoot, tis_settings, start_cond, dic)
            futures.add(j)
            out = next(futures)[1]
            # dic.update(out)
            # print('cat 1', ensemble['rgen'] == ensemble['rgen'])
            # print('cat 2', ensemble['rgen'] == out['ensemble']['rgen'])
            # print('cat 3', ensemble['path_ensemble'].rgen == ensemble['path_ensemble'].rgen)
            # print('cat 4', ensemble['path_ensemble'].rgen == out['ensemble']['path_ensemble'].rgen)
            # print('cat 5', ensemble['path_ensemble'].last_path.rgen == ensemble['path_ensemble'].last_path.rgen)
            # print('cat 6', ensemble['path_ensemble'].last_path.rgen == out['ensemble']['path_ensemble'].last_path.rgen)
            dic['ensemble'] = out['ensemble']
            dic['retur']  = out['retur']
            # print(out['retur'])
            # exit('tiger')
            # ensemble['path_ensemble'].last_path = dic['saved_traj']
            # ensemble['rgen'] = dic['ens_rgen']
            # ensemble['path_ensemble'].rgen = dic['pat_rgen']

            # print(ensemble['path_ensemble'].last_path.length, out['saved_traj'].length)
            # print(dir(dic['saved_traj'].rgen))
            # dic['ensemble']['rgen'] = out['ens_rgen'] 
            # dic['ensemble']['path_ensemble'].last_path = out['saved_traj']
            # ensemble['path_ensemble'].last_path = out['saved_traj']

            # if out['accept']:
            #     printint

        out_dask = dic['retur']
        out_loop = run_loop('./retis_3.rst', 2)
        print(type(out_loop), out_loop.keys() )
        print(type(out_dask), out_dask.keys() )
        # print(out_loop)
        # print(out_dask)
        # exit('na')

        for key in ['ens', 'path_ens', 'path']:
            print(key)
            for i, j in zip(out_loop[key], out_dask[key]):
                print(f'{i:.5f}\t{j:.5f}\t{i==j}')
    
# out_loop = run_loop('./retis_3.rst', 2)
# ens_check('./retis_3.rst', 2)
out_dask = dask_check('./retis_3.rst', 2)

# print('loop\tdask')

# print(type(out_loop), out_loop.keys() )
# print(type(out_dask), out_dask.keys() )
# for key in ['ens', 'path_ens', 'path']:
#     print(out_loop[key])
#     print(out_dask[key])
#     print(' ')
    
    # for i, j in zip(out_loop[key], out_dask[key]):
        # print(f'{i:.5f}\t{j:.5f}')
    





