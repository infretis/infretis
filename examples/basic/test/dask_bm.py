import numpy as np
import psutil
import os
import subprocess
from dask.distributed import Client, as_completed

# GMX = 'mdrun -s topol.tpr -deffnm sim00 -ntomp 4 -pin on -nsteps 50000 -resethway -notunepme -bonded cpu -pinoffset 0 -pinstride 1 -gpu_id 0'
# CMD = [0: '']
NTOMP = 1
GMX = f'gmx mdrun -s topol.tpr -deffnm worker-itnum -ntmpi {NTOMP} -nsteps 10 -pin on -pinoffset xxx -pinstride 1'

def run_gromacs(md_items):
    deffnm = f"sim-{md_items['w_id']}-{md_items['w_it']}"
    run = md_items['run'].replace('worker-itnum', deffnm)
    with open('output.txt', 'w') as out:
        return_code = subprocess.Popen(run.split(' '), stderr=out)
    return md_items

if __name__ == "__main__":

    # setup dask
    n_workers = 2
    client = Client(n_workers=n_workers)
    futures = as_completed(None, with_results=True)
    os.system(f' export OMP_NUM_THREADS={NTOMP}')
    it = 0

    # submit initial workers
    for i in range(n_workers):
        w_dic = {'w_id': i, 'deffnm': 'sim-worker-itnum'}
        # run = CMD[i] + GMX
        w_dic['run'] = GMX.replace('xxx', str(i))#  + f' > worker{i}.txt'
        w_dic['w_it'] = 0
        # w_dic['run'] = w_dic['run'].replace('worker', str(i))
        # w_dic['run_t'] = w_dic['run']
        # w_dic['run'] = w_dic['run'].replace('num', str(it))
        j = client.submit(run_gromacs, w_dic, pure=False)
        futures.add(j)

    # when a worker is done, check data and resubmit worker
    while it < 4:  
        items = next(futures)
        items[1]['w_it'] += 1
        # items[1]['run'] = items[1]['run_t'].replace('num', str(items[1]['w_it']))
        j = client.submit(run_gromacs, items[1], pure=False)
        futures.add(j)
        it += 1

    # obtain the last submitted workers
    while len(futures.futures):
        items = next(futures)[1]

    client.close()
