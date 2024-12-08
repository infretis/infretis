"""An asyncio-based task runner for infRETIS."""

import asyncio
import concurrent.futures
import functools
import logging
import multiprocessing
import threading
import time
from collections.abc import Callable
from typing import Any

from infretis.classes.formatter import get_log_formatter

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)


class RunnerError(Exception):
    """Exception class for the runner."""

    pass


class aiorunner:
    """A light asynchronuous runner based on asyncio.

    The runner manage an asyncio.queue with a pool of workers.
    Upon instanciation, a dedicated event loop
    is launched in a separate thread. The user can then
    attach a worker function to the runner and start multiple
    instances of that function in the background.
    As work is submitted to the runner, it is picked up by
    workers on-the-fly.
    """

    def __init__(self, config: dict, n_workers: int = 1) -> None:
        """Init function of runner.

        Args:
            config: the configuration dictionary
            n_workers: number of workers active in the runner
        """
        self._n_workers: int = n_workers
        self._counter = multiprocessing.Value("i", 0)
        self._executor: concurrent.futures.Executor = (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=worker_initializer,
                initargs=(self._counter,),
                mp_context=multiprocessing.get_context("fork"),
            )
        )
        self._stop_event = asyncio.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_event_loop, daemon=True
        )
        self._thread.start()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task_f: Callable | None = None
        self._tasks: list[asyncio.Task[Any]] | None = None

    def start(self) -> None:
        """Launch background tasks."""
        future = asyncio.run_coroutine_threadsafe(
            self._start_tasks(), self._loop
        )
        try:
            # Task startup should be fast
            future.result(5.0)
        except TimeoutError:
            raise RunnerError("Launching background tasks took too long")
        except Exception as e:
            raise (e)

    def _start_event_loop(self) -> None:
        """Start the event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def set_task(self, task_f: Callable) -> None:
        """Attach the task function to the runner.

        Args:
            task_f: a callable function
        """
        self._task_f = task_f

    async def _task_wrapper(
        self,
        stop_event: asyncio.Event,
        queue: asyncio.Queue[Any],
        executor: concurrent.futures.Executor,
        taskID: int,
    ) -> None:
        """Wrap the sync task.

        To enable running the sync task_f
        from a dynamic list of tasks.

        Args:
            stop_event: a asyncio event to stop the worker
            queue: an asyncio queue to get work from
            executor: an executor
            taskID : an ID for the long running task
        """
        while not stop_event.is_set():
            try:
                # Unpack queue element
                md_item, future = queue.get_nowait()

                # Run the task in the event loop
                assert self._task_f
                loop = asyncio.get_running_loop()
                try:
                    md_item = await loop.run_in_executor(
                        executor, functools.partial(self._task_f, md_item)
                    )
                    future.set_result(md_item)
                except Exception as e:
                    # Pass the exception up in the future
                    future.set_exception(e)

                # Mask the task as done
                queue.task_done()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)

    async def _add_work_to_queue(
        self, work_unit: dict[str, Any]
    ) -> asyncio.Future:
        """Async function adding work to queue, returns a future.

        Args
            work_unit: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        future: asyncio.Future = asyncio.Future()
        await self._queue.put((work_unit, future))
        return future

    def submit_work(self, work_unit: dict[str, Any]) -> asyncio.Future:
        """Submit work to the runner.

        Args:
            task: a unit of work encapsulated in a dict

        Return:
            A future wih the results of the work
        """
        if not self._tasks:
            raise RunnerError(
                "Unable to submit work if the tasks haven't been initiated"
            )
        future = asyncio.run(self._add_work_to_queue(work_unit))
        # Need to wait otherwise some race condition can occur
        time.sleep(0.05)
        return future

    async def _start_tasks(self) -> None:
        """Launch the background tasks."""
        if not self._task_f:
            raise RunnerError("Can't start task(s) without a task function.")
        try:
            self._tasks = [
                asyncio.create_task(
                    self._task_wrapper(
                        self._stop_event, self._queue, self._executor, i
                    )
                )
                for i in range(self._n_workers)
            ]
        except Exception as e:
            raise e

    async def wait_for_tasks_to_end(self) -> None:
        """Async function waiting for tasks to end."""
        while len(asyncio.all_tasks(self._loop)) > 0:
            await asyncio.sleep(0.1)

    def n_workers(self) -> int:
        """Return runner number of workers."""
        return self._n_workers

    def stop(self) -> None:
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


def worker_initializer(counter):
    """Initialize function for each worker process."""
    with counter.get_lock():  # Ensure that counter increment is thread-safe
        worker_id = counter.value
        counter.value += 1
    fileh = logging.FileHandler(f"worker{worker_id}.log", mode="a")
    log_levl = getattr(logging, "info".upper(), logging.INFO)
    fileh.setLevel(log_levl)
    fileh.setFormatter(get_log_formatter(log_levl))
    logger.addHandler(fileh)
    logger.info("=============================")
    logger.info("Logging file for worker %s", worker_id)
    logger.info("=============================\n")


class future_list:
    """A managed list of future."""

    def __init__(self) -> None:
        """Initialize future list."""
        self._futures: list[asyncio.Future] = []

    def add(self, future: asyncio.Future) -> None:
        """Add a future to list."""
        self._futures.append(future)

    def as_completed(self) -> asyncio.Future | None:
        """Get future as they are done in order of completion times.

        Return:
            return the first-to-complete future from the list
            or return None when the list is empty.
        """
        future_out = None
        done_futures: list[asyncio.Future] = []
        end_times = []
        while len(self._futures) > 0 and not done_futures:
            for fut in list(self._futures):
                if fut.done():
                    if fut.exception() is None:
                        end_times.append(fut.result().get("wmd_end",0.0))
                        done_futures.append(fut)
                    else:
                        return fut

        # process the first one to finish first
        if len(done_futures) > 0:
            index_min = min(range(len(end_times)), key=end_times.__getitem__)
            future_out = done_futures[index_min]
            self._futures.remove(future_out)

        return future_out
