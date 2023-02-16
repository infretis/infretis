import tomli
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

    print('tori a')
    for idx, key in enumerate(state.ensembles.keys()):
        # print('pnumber', state.ensembles[key]['path_ensemble'].last_path.path_number, state.ensembles[key]['path_ensemble'].last_path.length)
        traj = state._trajs[idx]
        print('pnumber', traj.path_number, traj.length)
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].last_path.rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].rgen.get_state()['state'][2])
        # print(f'00{idx}', state.ensembles[key]['engine'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['rgen'].get_state()['state'][2])
    print('tori b')

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
    print('koppa a')
    for idx, key in enumerate(state.ensembles.keys()):
        # print('pnumber', state.ensembles[key]['path_ensemble'].last_path.path_number, state.ensembles[key]['path_ensemble'].last_path.length)
        traj = state._trajs[idx]
        print('pnumber', traj.path_number, traj.length)
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].last_path.rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['path_ensemble'].rgen.get_state()['state'][2])
        # print(f'00{idx}', state.ensembles[key]['engine'].rgen.get_state()['state'][2])
        print(f'00{idx}', state.ensembles[key]['rgen'].get_state()['state'][2])
    print('koppa b')
    client.close()
