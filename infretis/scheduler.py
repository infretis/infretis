"""The main infretis loop."""
from infretis.core.tis import def_globals, run_md
from infretis.setup import setup_dask, setup_internal


def scheduler(config):
    """Run infretis loop."""
    # setup repex, dask and futures
    md_items, state = setup_internal(config)
    client, futures = setup_dask(state)
    # define global orderparameters and engines
    # print('bunny', config)
    # exit('a')
    # for _ in range(config["dask"]["workers"]):
    #     fut = client.submit(def_globals, config, pure=False)
    #     futures.add(fut)
    # while not futures.is_empty():
    #     _ = next(futures)[1]
    # exit('broski')

    # submit the first number of workers
    while state.initiate():
        # pick and prep ens and path for the next job
        md_items = state.prep_md_items(md_items)

        # submit job to scheduler
        fut = client.submit(run_md, md_items, pure=False)
        futures.add(fut)

    # main loop
    while state.loop():
        # get and treat worker output
        md_items = state.treat_output(next(futures)[1])

        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            # chose ens and path for the next job
            md_items = state.prep_md_items(md_items)

            # submit job to scheduler
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)

    # end client
    client.close()
