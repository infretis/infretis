import unittest
import tempfile
import numpy as np
from infretis.common import setup_dask
from infretis.inf_core import REPEX_state
from dask.distributed import dask, Client, as_completed


class test_repex_random(unittest.TestCase):
    
    def test_repex_random_1(self):
        state1 = REPEX_state(5, workers=2, minus=True)

        for _ in range(10):
            np.random.rand()
        saved_rng1 = []
        saved_rng2 = []
        with tempfile.TemporaryDirectory(dir='./') as tempdir:
            config = {'simulation': {'save_loc': tempdir}}
            state1.config = config
            state1.save_rng()
            for _ in range(10):
                saved_rng1.append(np.random.rand())
            state1.set_rng()
            for _ in range(10):
                saved_rng2.append(np.random.rand())
        self.assertTrue(all([i == j for i, j in zip(saved_rng1, saved_rng2)]))



        # for _ in range(10):
        #     print(np.random.rand())

        print('hihi')


        w_steal = dask.config.get('distributed.scheduler.work-stealing')
        self.assertFalse(dask.config.get('distributed.scheduler.work-stealing'))
        # scheduler, futures = setup_dask(workers=workers)


if __name__ == '__main__':  
    unittest.main()
