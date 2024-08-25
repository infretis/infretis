import pytest
from infretis.asyncrunner import aiorunner
from typing import Dict, Any
from time import sleep

def sleeping(sleep_dict : Dict[str, Any]) -> Dict[str, Any]:
    sleeping_time = sleep_dict.get("duration", 0.5)
    time.sleep(sleeping_time)
    return {"Time slept" : sleeping_time}

def test_runner_init():
    runner = aiorunner({})
    assert(runner.n_workers() == 1)

def test_runner_attach_callable():
    runner = aiorunner({}, 2)
    runner.set_task(sleeping)
    assert(runner.n_workers() == 2)

def test_runner_fail_start():
    runner = aiorunner({}, 2)
    with pytest.raises(Exception):
        runner.start()

def test_runner_start_and_stop():
    runner = aiorunner({}, 2)
    runner.set_task(sleeping)
    runner.start()
    runner.stop()
