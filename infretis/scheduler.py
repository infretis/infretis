"""The main infretis loop."""
from infretis.setup import setup_runner, setup_internal
import time
import concurrent.futures


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

    # main step loop
    while state.loop():

        one_future_done = False

        # At each loop iteration, check the status of futures
        while not one_future_done:
            for fut in futures:
                if fut.done():
                    print(" pin: {}, cstep: {}".format(md_items["pin"], state.cstep))
                    # get and treat worker output
                    md_items = state.treat_output(fut.result())

                    # submit new job:
                    if state.cstep + state.workers <= state.tsteps:
                        # chose ens and path for the next job
                        md_items = state.prep_md_items(md_items)

                        # submit job to scheduler
                        futures.append(runner.submit_work(md_items))
                        futures.remove(fut)
                    one_future_done = True
                    break
            #time.sleep(0.1)

        #print("State loop")


    # end client
    runner.stop()
