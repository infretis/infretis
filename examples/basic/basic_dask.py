import numpy as np
import psutil
from dask.distributed import Client, as_completed


def return_data(inp):
    inp[0] += 1
    return inp


if __name__ == "__main__":
    n_workers = 5
    client = Client(n_workers=n_workers)
    futures = as_completed(None, with_results=True)

    big_nums = [np.float128(np.random.rand()) for _ in range(5000)]
    for i in range(1):
        j = client.submit(return_data, big_nums, pure=False)
        futures.add(j)

    it = 0
    while it < 10000:
        items = next(futures)
        j = client.submit(return_data, items[1], pure=False)
        futures.add(j)
        print(it, psutil.Process().memory_info().rss / 10**9)
        it += 1

    while len(futures.futures):
        items = next(futures)[1]
        print(it, psutil.Process().memory_info().rss / 10**9)
        it += 1

    client.close()
