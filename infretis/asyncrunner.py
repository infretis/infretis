import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, List
import time

class light_runner():
    def __init__(self, condig : dict, n_workers):
        self._n_workers = n_workers
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        self._stop_event = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._thread.start()
        self._queue : asyncio.Queue[Any] = asyncio.Queue()
        self._task = None

    def start(self):
        asyncio.run_coroutine_threadsafe(self.start_tasks(), self._loop)

    def _start_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_task(self, worker_task) -> None:
        self._task = worker_task

    async def add_work_to_queue(self, str_in : str) -> asyncio.Future:
        future = asyncio.Future()
        await self._queue.put((str_in, future))
        return future

    def submit_work(self, str_in : str) -> asyncio.Future:
        future = asyncio.run(self.add_work_to_queue(str_in))
        return future

    async def start_tasks(self):
        self._tasks = [ asyncio.create_task(self._task(self._stop_event,
                                                       self._queue,
                                                       self._executor,
                                                       i))
                       for i in range(self._n_workers) ]

    async def wait_for_tasks_to_end(self):
        while len(asyncio.all_tasks(self._loop)) > 0:
            await asyncio.sleep(0.1)

    def stop(self):
        # Make sure there is no more work in the queue
        # before dispatching the task stopping event
        while self._queue.qsize() > 0:
            time.sleep(0.1)

        # Stop ongoing tasks
        self._stop_event.set()

        # Wait until all tasks are done 
        asyncio.run(self.wait_for_tasks_to_end())

        # Close the event loop
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
