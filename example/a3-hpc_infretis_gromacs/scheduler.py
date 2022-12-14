import dask.distributed
from dask.distributed import Client, as_completed
import tomli
from help_func import run_md, print_path_info, set_shooting, treat_output
from help_func import setup_pyretis, setup_repex, print_end, write_to_pathens
dask.config.config['work-stealing'] = False

if __name__ == "__main__":

    # read config and setup pyretis sim
    with open("./infretis.toml", mode="rb") as f:
        config = tomli.load(f)
    steps = config['simulation']['steps']

    # setup pyretis
    sim = setup_pyretis(config)
    moves = sim.settings['tis']['shooting_moves']
    interfaces = sim.settings['simulation']['interfaces']

    # setup repex and dask
    state = setup_repex(sim, config)
    client = Client(n_workers=state.workers)
    futures = as_completed(None, with_results=True)

    # start running
    print('stored ensemble paths:')
    print(' '.join([f'00{i}: {j},' for i, j in enumerate([sim0['path_ensemble'].last_path.path_number for sim0 in sim.ensembles])]))
    print('saved ensemble paths:', state.live_paths())
    print_path_info(state)
    print(' ')

    # initialization: submit the first batch of workers
    for worker in range(state.workers):

        print(f'------- submit worker {worker} START -------')

        # Pick and set ens and trajs

        ens, input_traj = state.pick()
        ## if no locks:
        ##    ens, input_traj = state.pick()
        ## else: if locks:
        ##    ens, input_traj = state.lock_pick()
        move = set_shooting(sim, ens, input_traj, str(worker), moves)

        # submit job
        fut = client.submit(run_md, ens, input_traj,
                            sim, worker,
                            move, str(worker), pure=False)
        futures.add(fut)

        print(f'------- submit worker {worker} END -------')
        print()

    while config['current']['step'] < steps + state.workers:
        step = config['current']['step']

        # get output from finished worker
        output = next(futures)[1]
        print(f'------- infinity {step} START -------')

        # analyze & store output
        pin, acc, pn_archive = treat_output(output, state, sim)

        # print analyzed output  # and write toml
        if acc:
            write_to_pathens(state, pn_archive)

        # submit new job:
        if step < steps:
            # chose ens and path for the next job
            ens, input_traj = state.pick()
            move = set_shooting(sim, ens, input_traj, pin, moves)
            print_path_info(state)

            # submit job
            fut = client.submit(run_md, ens, input_traj,
                                sim, step, move, pin, pure=False)
            futures.add(fut)
        else: 
            print_path_info(state)

        print(f'------- infinity {step} END -------')
        config['current']['step'] += 1

        print()

    live_trajs = state.live_paths()
    traj_num_dic = state.traj_num_dic
    print_end(live_trajs, config['current']['step'], traj_num_dic)
