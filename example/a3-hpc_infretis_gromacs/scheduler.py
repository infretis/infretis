import dask.distributed
from dask.distributed import Client, as_completed
import tomli
from help_func import run_md, treat_output, print_end
from help_func import setup_internal, setup_dask, prepare_shooting
dask.config.config['work-stealing'] = False

if __name__ == "__main__":

    # read config
    with open("./infretis.toml", mode="rb") as f:
        config = tomli.load(f)

    # setup pyretis, repex, dask client and futures
    md_items, state = setup_internal(config)
    client, futures = setup_dask(state.workers)

    # print and initiate
    state.print_start()
    for worker in range(state.workers):
        print(f'------- submit worker {worker} START -------')

        # chose ens and path for the next job
        ens, input_traj = state.pick_lock()
        md_items.update({'ens': ens, 'input_traj': input_traj,
                         'pin': worker, 'cycle': worker})
        prepare_shooting(state, md_items)

        # submit job
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

        print(f'------- submit worker {worker} END -------\n')

    while state.loop():
        # get output from finished worker
        output = next(futures)[1]
        print(f'------- infinity {state.cstep} START -------')

        # analyze & store output
        pin = treat_output(output, state, md_items['sim'])

        # submit new job:
        if state.cstep < state.steps:
            # chose ens and path for the next job
            ens, input_traj = state.pick()
            md_items.update({'ens': ens, 'input_traj': input_traj,
                             'pin': pin, 'cycle': state.cstep})
            prepare_shooting(state, md_items)

            # submit job
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)
        else: 
            state.print_state()

        print(f'------- infinity {state.cstep} END -------')
        state.cstep += 1
        print()

    state.print_end()
