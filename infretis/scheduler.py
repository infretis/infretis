"""The main infretis loop."""

import copy
import tarfile

from infretis.setup import setup_internal, setup_runner


def scheduler(config):
    """Run infretis loop."""
    # setup repex, runner and futures
    md_items, state = setup_internal(config)
    runner, futures = setup_runner(state)

    # submit the first number of workers
    while state.initiate():
        # give each worker its own md_items
        worker_md_items = copy.deepcopy(md_items)
        # pick and prep ens and path for the next job
        worker_md_items = state.prep_md_items(worker_md_items)

        # submit job to scheduler
        futures.add(runner.submit_work(worker_md_items))

    # main step loop
    with tarfile.open(state.tar_file, "a") as tar:
        state.tar = tar
        while state.loop():
            # Get futures as they are completed
            future = futures.as_completed()

            if future:
                worker_md_items = state.treat_output(future.result())

            # submit new job
            if state.cstep + state.workers <= state.tsteps:
                # chose ens and path for the next job
                worker_md_items = state.prep_md_items(worker_md_items)

                # submit job to scheduler
                futures.add(runner.submit_work(worker_md_items))

    # end client
    state.tar = None
    runner.stop()
