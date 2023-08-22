import numpy as np
import psutil
import os
import subprocess
from dask.distributed import Client, as_completed

# GMX = 'mdrun -s topol.tpr -deffnm sim00 -ntomp 4 -pin on -nsteps 50000 -resethway -notunepme -bonded cpu -pinoffset 0 -pinstride 1 -gpu_id 0'
# CMD = [0: '']
NTOMP = 1
GMX = f"gmx mdrun -s topol.tpr -deffnm worker-itnum -ntmpi {NTOMP} -nsteps 25 -pin on -pinoffset xxx -pinstride 1"


def read_log(deffnm):
    with open(f"{deffnm}.log", "r") as read:
        for line in read:
            if "Performance" in line:
                return line.rstrip().split()[1]


def run_gromacs(md_items):
    deffnm = f"sim-{md_items['w_id']}-{md_items['w_it']}"
    run = md_items["run"].replace("worker-itnum", deffnm)
    with open("output.txt", "w") as out:
        return_code = subprocess.Popen(run.split(" "), stderr=out)
    md_items["perf"] = read_log(deffnm)

    return md_items


def write_result(md_items, data="results.txt", mode="a"):
    with open(data, mode) as write:
        if mode == "w":
            write.write("#\tWorker\tPerformance [ns/d]\n")
        else:
            w_id = md_items.get("w_id")
            w_perf = md_items.get("perf", "")
            write.write(f"\t{w_id}\t{w_perf}\n")


if __name__ == "__main__":
    # setup dask
    n_workers = 2
    client = Client(n_workers=n_workers)
    futures = as_completed(None, with_results=True)
    os.system(f" export OMP_NUM_THREADS={NTOMP}")
    it = 0
    write_result({}, mode="w")

    # submit initial workers
    data = {}
    for i in range(n_workers):
        data[i] = []
        w_dic = {"w_id": i, "w_it": 0}
        w_dic["run"] = GMX.replace("xxx", str(i))
        j = client.submit(run_gromacs, w_dic, pure=False)
        futures.add(j)

    # when a worker is done, check data and resubmit worker
    while it < 4:
        items = next(futures)
        write_result(items[1])
        items[1]["w_it"] += 1
        j = client.submit(run_gromacs, items[1], pure=False)
        futures.add(j)
        it += 1

    # obtain the last submitted workers
    while len(futures.futures):
        items = next(futures)
        write_result(items[1])

    client.close()
