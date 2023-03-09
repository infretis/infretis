import numpy as np
from infretis.common import run_md, run_bm, treat_output, pwd_checker
from infretis.common import setup_internal, setup_dask, prep_pyretis


def scheduler(input_file):
    # setup pyretis, repex, dask client and futures
    md_items, state, config = setup_internal(input_file)
    if None in (md_items, state):
        return
    client, futures = setup_dask(config, state.workers)

    # submit the first number of workers
    while state.initiate(md_items):
        # chose ens and path for the next job
        ens_nums, input_traj = state.pick_lock()
        prep_pyretis(state, md_items, input_traj, ens_nums)

        # submit job
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

    # main loop
    while state.loop():
        # get and treat worker output
        md_items = next(futures)[1]
        treat_output(state, md_items)

        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            # chose ens and path for the next job
            ens_nums, inp_traj = state.pick()
            prep_pyretis(state, md_items, inp_traj, ens_nums)

            # submit job
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)

    # end client
    client.close()

def bm_scheduler(input_file):
    # setup pyretis, repex, dask client and futures
    md_items, state, config = setup_internal(input_file)
    state.zeroswap = 0.
    if None in (md_items, state):
        return
    stable_a, stable_b = config['simulation']['bm_intfs']
    for key in state.ensembles.keys():
        state.ensembles[key]['interfaces'][0] = stable_a
        state.ensembles[key]['interfaces'][-1] = stable_b
    for idx, path in enumerate(state._trajs[:-1]):
        path.maxlen = int(config['simulation']['bm_steps'])

    client, futures = setup_dask(config, state.workers)

    # submit the first number of workers
    while state.initiate(md_items):
        # chose ens and path for the next job
        ens_nums, input_traj = state.pick_lock()
        prep_pyretis(state, md_items, input_traj, ens_nums)

        fut = client.submit(run_bm, md_items, pure=False)
        futures.add(fut)

    # main loop
    while state.loop():
        # get and treat worker output
        md_items = next(futures)[1]
        treat_output(state, md_items)

        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            # chose ens and path for the next job
            ens_nums, inp_traj = state.pick()
            prep_pyretis(state, md_items, inp_traj, ens_nums)

            # submit job
            fut = client.submit(run_bm, md_items, pure=False)
            futures.add(fut)

    # end client
    client.close()
