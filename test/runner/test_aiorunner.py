from time import sleep
from typing import Any

import pytest

from infretis.asyncrunner import aiorunner, future_list


class TaskError(Exception):
    """Exception class for the test task."""

    pass


def sleeping(sleep_dict: dict[str, Any]) -> dict[str, Any]:
    """A simple sleeping task.

    Raising an error if triggered

    Args
        sleep_dict: a dictionary of task params

    Return
        A dictionary containing the task results
    """
    raise_error = sleep_dict.get("raise_err", False)
    if raise_error:
        raise TaskError("Task raise error on purpose")
    else:
        sleeping_time = sleep_dict.get("duration", 0.5)
        sleep(sleeping_time)
        return {"Time slept": sleeping_time}


def test_runner_init():
    """Test bare runner init."""
    runner = aiorunner({})
    assert runner.n_workers() == 1
    runner.stop()


def test_runner_attach_callable():
    """Test attach callable to runner."""
    runner = aiorunner({}, 2)
    runner.set_task(sleeping)
    assert runner.n_workers() == 2
    runner.stop()


def test_runner_fail_start():
    """Test fail runner start missing callable."""
    runner = aiorunner({}, 1)
    with pytest.raises(Exception):
        runner.start()


def test_runner_start_and_stop():
    """Test runner start and stop."""
    runner = aiorunner({}, 1)
    runner.set_task(sleeping)
    runner.start()
    runner.stop()


def test_runner_fail_submit():
    """Test fail runner submit work, need start."""
    runner = aiorunner({}, 2)
    with pytest.raises(Exception):
        runner.submit_work({"duration": 1.0})


def test_runner_check_return_success():
    """Test runner return task result."""
    runner = aiorunner({}, 1)
    futlist = future_list()
    runner.set_task(sleeping)
    runner.start()
    futlist.add(runner.submit_work({"duration": 0.3}))
    fut = futlist.as_completed()
    assert fut.result().get("Time slept") == 0.3
    runner.stop()


def test_runner_task_error():
    """Test fail runner task raised error."""
    runner = aiorunner({}, 1)
    futlist = future_list()
    runner.set_task(sleeping)
    runner.start()
    futlist.add(runner.submit_work({"raise_err": True}))
    fut = futlist.as_completed()
    with pytest.raises(TaskError):
        fut.result()


def test_runner_infretis_mode():
    """Test runner operating in infretis mode."""
    n_workers = 2
    runner = aiorunner({}, n_workers)
    futlist = future_list()
    runner.set_task(sleeping)
    runner.start()
    loop_cnt = 0
    n_loops = 10
    res_dict = {}
    while loop_cnt < n_loops + 2:
        if loop_cnt < n_workers:
            futlist.add(runner.submit_work({}))
            loop_cnt = loop_cnt + 1
        else:
            future = futlist.as_completed()
            if future:
                res_dict[f"l{loop_cnt}"] = future.result()
            if loop_cnt <= n_loops:
                futlist.add(runner.submit_work({}))
            loop_cnt = loop_cnt + 1
    runner.stop()
    assert len(res_dict) == 10
