import asyncio
import concurrent.futures
import threading
from typing import Any, Callable, List, Dict
import time
import functools

class RunnerError(Exception):
    """Exception class for the runner."""
    pass

class light_runner():
    """A light asynchronuous runner based on asyncio.

    The runner manage an asyncio.queue with a pool of workers.
    Upon instanciation, a dedicated event loop
    is launched in a separate thread. The user can then
    attach a worker function to the runner and start multiple
    instances of that function in the background.
    As work is submitted to the runner, it is picked up by
    worker on-the-fly.
    """
    def __init__(self, config : Dict, n_workers : int = 1):
        """Init function of runner.

        Args:
            config: a dict of parameters
            n_workers: number of workers active in the runner
        """
        self._n_workers : int = n_workers
        self._executor : concurrent.futures.Executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        self._stop_event = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._thread.start()
        self._queue : asyncio.Queue[Any] = asyncio.Queue()
        self._task_f : Callable | None = None
        self._tasks : List[asyncio.Task[Any]] | None = None

    def start(self):
        """Launch background tasks."""
        asyncio.run_coroutine_threadsafe(self._start_tasks(), self._loop)

    def _start_event_loop(self):
        """Start the event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_task(self, task_f : Callable) -> None:
        """Attach the task function to the runner.

        Args:
            task_f: a callable function
        """
        self._task_f = task_f

    async def _task_wrapper(self,
                            stop_event : asyncio.Event(),
                            queue : asyncio.Queue[Any],
                            executor : concurrent.futures.Executor,
                            taskID : int) -> None:
        """An async wrapper to enable running the sync task_f from a dynamic list of tasks.

        Args:
            stop_event: a asyncio event to stop the worker
            queue: an asyncio queue to get work from
            executor: an executor
            taskID : an ID for the long running task
        """
        print("In _task_wrapper")
        while not stop_event.is_set():
            try:
                # Unpack queue element
                md_item, future = queue.get_nowait()
                if md_item is None:
                    break
                print("[{}] Running task in executor".format(taskID))
                #print("Task pin: {}".format(md_item["pin"]))
                loop = asyncio.get_running_loop()
                md_item = await loop.run_in_executor(
                    executor,
                    functools.partial(self._task_f, md_item)
                )
                future.set_result(md_item)
                queue.task_done()
                #print("[{}] Done running task".format(taskID))
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)

    async def _add_work_to_queue(self, work_unit : Dict[str, Any]) -> asyncio.Future:
        """Async function adding work to queue, returns a future.

        Args
            work_unit: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        future = asyncio.Future()
        await self._queue.put((work_unit, future))
        return future

    def submit_work(self, work_unit : Dict[str, Any]) -> asyncio.Future:
        """Submit work to the runner.

        Args:
            task: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        if not self._tasks:
            raise RunnerError("Unable to submit work if the tasks haven't been initiated")
        future = asyncio.run(self._add_work_to_queue(work_unit))
        # Need to wait otherwise some race condition can occur
        time.sleep(0.1)
        return future

    async def _start_tasks(self):
        """Launch the background tasks."""
        if not self._task_f:
            raise RunnerError("Unable to start the background task(s) without a task function.")
        try:
            self._tasks = [ asyncio.create_task(self._task_wrapper(self._stop_event,
                                                                   self._queue,
                                                                   self._executor,
                                                                   i))
                           for i in range(self._n_workers) ]
        except Exception as e:
            print(e)

    async def wait_for_tasks_to_end(self):
        """Async function waiting for tasks to end."""
        while len(asyncio.all_tasks(self._loop)) > 0:
            await asyncio.sleep(0.1)

    def stop(self):
        """Terminate the runner."""
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
