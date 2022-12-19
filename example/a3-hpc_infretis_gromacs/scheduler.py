import dask.distributed
from dask.distributed import Client, as_completed
import tomli
from help_func import run_md, print_path_info, set_shooting, treat_output
from help_func import print_end, write_to_pathens
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

        ens, input_traj = state.pick_lock()

        md_items.update({'ens': ens, 'input_traj': input_traj,
                         'pin': worker, 'cycle': worker})
        prepare_shooting(state, md_items)
        fut = client.submit(run_md, md_items, pure=False)
        # fut = client.submit(run_md, ens, input_traj,
        #                     sim, worker,
        #                     move, str(worker), pure=False)
        futures.add(fut)

        print(f'------- submit worker {worker} END -------\n')

    exit('mboppi')
    while state.loop():
        # get output from finished worker
        output = next(futures)[1]
        print(f'------- infinity {state.cstep} START -------')

        # analyze & store output
        pin, acc, pn_archive = treat_output(output, state, sim)

        # print analyzed output  # and write toml
        if acc:
            write_to_pathens(state, pn_archive)

        # submit new job:
        if state.cstep < state.steps:
            # chose ens and path for the next job
            ens, input_traj = state.pick()
            move = set_shooting(sim, ens, input_traj, pin)
            print_path_info(state, ens, input_traj)

            # submit job
            fut = client.submit(run_md, ens, input_traj,
                                sim, state.cstep, move, pin, pure=False)
            futures.add(fut)
        else: 
            print_path_info(state)

        print(f'------- infinity {state.cstep} END -------')
        state.cstep += 1
        print()

    live_trajs = state.live_paths()
    traj_num_dic = state.traj_num_dic
    print_end(live_trajs, state.cstep, traj_num_dic)
