import tomli
from help_func import run_md, treat_output
from help_func import setup_internal, setup_dask, prepare_pyretis
import time

if __name__ == "__main__":

    start_time = time.time() 
    # read config
    with open("./infretis.toml", mode="rb") as f:
        config = tomli.load(f)

    # setup pyretis, repex, dask client and futures
    md_items, state = setup_internal(config)
    flamingo0 = [state.ensembles[kk]['path_ensemble'].last_path.path_number for kk in [0, 1, 2]]
    client, futures = setup_dask(state.workers)

    # print and initiate
    state.print_start()
    for worker in range(state.workers):
        # print(f'------- submit worker {worker} START -------')

        # chose ens and path for the next job
        ens_nums, input_traj = state.pick_lock()

        md_items.update({'ens_nums': ens_nums, 'pin': worker})
        prepare_pyretis(state, md_items, input_traj, printing=True)

        # submit job
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

        # print(f'------- submit worker {worker} END -------\n')

    # main loop

    time1 = time.time() 
    while state.loop():
        # get output from finished worker
        md_items = next(futures)[1]
        # print(f'------- infinity {state.cstep} START -------')

        # analyze & store output
        # time2 = time.time()
        treat_output(state, md_items)
        # time3 = time.time()
        # state.print_state()

        # submit new job:
        if state.cstep < state.steps:
            # chose ens and path for the next job
            ens_nums, input_traj = state.pick()
            # time4 = time.time()
            md_items.update({'ens_nums': ens_nums})
            prepare_pyretis(state, md_items, input_traj, printing=False)
            time5 = time.time()
            # submit job
            print(f'{state.cstep:7.0f}',
                  f'{time5 - time1:2.5f}',
            #       f'{time2 - time1:2.5f}',
            #       f'{time3 - time2:2.5f}',
            #       f'{time4 - time3:2.5f}',
            #       f'{time5 - time4:2.5f}',
                 )
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)
            time1 = time.time() 

        # print(f'------- infinity {state.cstep} END -------')
        state.cstep += 1
        # print()

    state.print_end()
    print('total time spent:', time.time() - start_time)
