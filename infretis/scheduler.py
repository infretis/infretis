import numpy as np
from infretis.common import treat_output, run_md
from infretis.common import setup_internal, setup_dask

def scheduler(input_file):
    # setup repex, dask and futures
    md_items, state = setup_internal(input_file)
    client, futures = setup_dask(state)
    # print('whada')
    # exit('eh')

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
        md_items = treat_output(state, next(futures)[1])

        # submit new job:
        if state.cstep + state.workers <= state.tsteps:
            # chose ens and path for the next job
            md_items = state.prep_md_items(md_items)

            # submit job
            fut = client.submit(run_md, md_items, pure=False)
            futures.add(fut)

    # end client
    client.close()
