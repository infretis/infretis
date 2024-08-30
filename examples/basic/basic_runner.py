"""A basic example."""
import numpy as np
import psutil
from infretis.asyncrunner import aiorunner, future_list


def return_data(inp : dict):
    """Return data."""
    inp["0"] += 1
    return inp


if __name__ == "__main__":
    n_workers = 5
    runner = aiorunner({}, n_workers=n_workers)
    runner.set_task(return_data)
    runner.start()
    futures = future_list()

    big_nums = [np.longdouble(np.random.rand()) for _ in range(5000)]
    for i in range(n_workers):
        j = runner.submit_work({"0":0})
        futures.add(j)

    it = 0
    nit = 10000
    while it < nit:
        items = futures.as_completed().result()
        if it < nit-n_workers:
            j = runner.submit_work(items)
            futures.add(j)
        print(it, psutil.Process().memory_info().rss / 10**9)
        it += 1

    runner.stop()
