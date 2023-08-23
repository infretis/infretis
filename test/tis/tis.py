from infretis.classes.engines.factory import create_engine
from infretis.classes.orderparameter import create_orderparameters
from infretis.classes.path import restart_path
from infretis.classes.rgen import MockRandomGenerator
from infretis.core.tis import shoot

ENG_SET = {
    "class": "turtlemd",
    "engine": "turtlemd",
    "timestep": 0.025,
    "temperature": 0.07,
    "boltzmann": 1.0,
    "subcycles": 1,
    "integrator": {
        "class": "LangevinInertia",
        "settings": {"gamma": 0.3, "beta": 14.285714285714285, "seed": 69},
    },
    "potential": {
        "class": "DoubleWell",
        "settings": {"a": 1.0, "b": 2.0, "c": 0.0},
    },
    "particles": {"mass": [1.0], "name": ["Z"], "pos": [[-1.0]]},
    "box": {"periodic": [False]},
}

ENS_SET = {
    "interfaces": (-0.99, -0.3, 1.0),
    "tis_set": {
        "maxlength": 2000,
        "aimless": True,
        "allowmaxlength": False,
        "zero_momentum": False,
        "rescale_energy": False,
    },
    "mc_move": "sh",
    "eng_name": "turtlemd",
    "ens_name": "007",
    "start_cond": "L",
    "rgen": MockRandomGenerator(),
}

ORDP_SET = {"class": "Position", "index": [0, 0], "periodic": False}
TURTLE = create_engine({"engine": ENG_SET})
TURTLE.rgen = ENS_SET["rgen"]
create_orderparameters({"engine": TURTLE}, {"orderparameter": ORDP_SET})
INP_PATH = restart_path("./load/7/path.restart")


def test_yo(tmp_path):
    f1 = tmp_path / "cake"
    f1.mkdir()
    TURTLE.exe_dir = f1
    success, trial_seg, status = shoot(ENS_SET, INP_PATH, TURTLE)
    assert not success
    assert trial_seg.length == 34
    assert status == "BWI"
