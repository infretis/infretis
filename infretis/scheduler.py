"""The main infretis loop."""
from infretis.setup import setup_runner, setup_internal
import time


def scheduler(config):
    """Run infretis loop."""
    # setup repex, runner and futures
    md_items, state = setup_internal(config)
    runner, futures = setup_runner(state)

    # submit the first number of workers
    while state.initiate():
        # pick and prep ens and path for the next job
        md_items = state.prep_md_items(md_items)

        # submit job to scheduler
        futures.append(runner.submit_work(md_items))

    time.sleep(1.0)

    # main loop
    while state.loop():
        for fut in list(futures):
            if fut.done():
                # get and treat worker output
                md_items = state.treat_output(fut.result())

                # submit new job:
                if state.cstep + state.workers <= state.tsteps:
                    # chose ens and path for the next job
                    md_items = state.prep_md_items(md_items)

                    # submit job to scheduler
                    futures.append(runner.submit_work(md_items))
                    futures.remove(fut)

        #print("State loop")
        time.sleep(0.2)


    # end client
    runner.stop()
