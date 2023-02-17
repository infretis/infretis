import tomli
import numpy as np
from infretis.common import run_md, treat_output, pwd_checker
from infretis.common import setup_internal, setup_dask, prep_pyretis


def scheduler(input_file):

    with open(input_file, mode="rb") as f:
        config = tomli.load(f)

    # setup pyretis, repex, dask client and futures
    md_items, state = setup_internal(config)
    if None in (md_items, state):
        return
    client, futures = setup_dask(config, state.workers)
    print('shark a')
    if not config['simulation']['internal']:
        pwd_checker(state)
    print('shark b')

    # submit the first number of workers
    while state.initiate(md_items):
        # chose ens and path for the next job
        # print('chicken a', np.random.get_state()[2], np.random.get_state()[3])
        ens_nums, input_traj = state.pick_lock()
        prep_pyretis(state, md_items, input_traj, ens_nums)

        # submit job
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

    # main loop
    while state.loop():
        # print('chicken b', np.random.get_state()[2], np.random.get_state()[3])
        # get and treat worker output
        if state.cstep == 5:
            break
        #     print('stopp!!!1')
        md_items = next(futures)[1]
        treat_output(state, md_items)
        state.save_rng()
        state.write_toml()

        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            # chose ens and path for the next job
            # vvvv if crash here it is not OK vvvv
            # if state.cstep == 10:
            #     break
            ens_nums, inp_traj = state.pick()
            print('panda', ens_nums)
            # state.write_toml()

            # vvvv if crash here it is OK vvvv
            print('panda', ens_nums)
            # if state.cstep == 10:
            #     break
            prep_pyretis(state, md_items, inp_traj, ens_nums)
            # vvvv if crash here it is OK vvvv
            # if state.cstep == 10:
            #     break

            # submit job
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)
            # vvvv if crash here it is OK vvvv
            # if state.cstep == 10:
            #     break

    # end client
    print('bear a')
    if not config['simulation']['internal']:
        pwd_checker(state)
    print('bear b')
    client.close()
    # print('chicken b', np.random.get_state()[2], np.random.get_state()[3])
