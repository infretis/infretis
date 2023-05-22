from infretis.core.setup import setup_internal, setup_dask
from infretis.core.tis import run_md

def scheduler(input_file):
    # setup repex, dask and futures
    md_items, state = setup_internal(input_file)
    client, futures = setup_dask(state)

    # submit the first number of workers
    while state.initiate():
        # pick and prep ens and path for the next job
        md_items = state.prep_md_items(md_items)

        # submit job
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
