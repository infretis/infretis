import unittest
from infretis.common import setup_dask
from dask.distributed import dask, Client, as_completed


class test_setup_dask(unittest.TestCase):
    
    def test_setup_dask_1(self):
        workers = 5

        w_steal = dask.config.get('distributed.scheduler.work-stealing')
        self.assertFalse(dask.config.get('distributed.scheduler.work-stealing'))
        # scheduler, futures = setup_dask(workers=workers)


if __name__ == '__main__':  
    unittest.main()
