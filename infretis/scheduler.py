import tomli
from infretis.common import run_md, treat_output
from infretis.common import setup_internal, setup_dask, prep_pyretis


def scheduler(input_file):

    with open(input_file, mode="rb") as f:
        config = tomli.load(f)

    # setup pyretis, repex, dask client and futures
    md_items, state = setup_internal(config)
    client, futures = setup_dask(state.workers)

    print('soppa 0')
    for idx, key in enumerate(state.ensembles.keys()):
        print('pnumber', state.ensembles[key]['path_ensemble'].last_path.path_number)
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].last_path.rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['engine'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['rgen'].get_state()['state'][2])
    print('soppa 1')

    # submit the first number of workers
    while state.initiate():
        # chose ens and path for the next job
        ens_nums, input_traj = state.pick_lock()
        prep_pyretis(state, md_items, input_traj, ens_nums)

        # submit job
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

    # main loop
    while state.loop():
        # get output from finished worker
        md_items = next(futures)[1]

        # analyze & store output
        treat_output(state, md_items)

        print('kaka1')
        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            print('kaka2')
            # chose ens and path for the next job
            ens_nums, inp_traj = state.pick()
            prep_pyretis(state, md_items, inp_traj, ens_nums)

            # submit job
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)

    # print('start horse 0')
    # for idx, traj in enumerate(state._trajs[:-1]):
    #     print(idx, traj.path_number, traj.ordermax, traj.length)
    # print('start horse 1')
    print('boppa 0')
    for idx, key in enumerate(state.ensembles.keys()):
        # print('pnumber', state.ensembles[key]['path_ensemble'].last_path.path_number, state.ensembles[key]['path_ensemble'].last_path.length)
        traj = state._trajs[idx]
        print('pnumber', traj.path_number, traj.length)
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].last_path.rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['engine'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['rgen'].get_state()['state'][2])
    print('boppa 1')
