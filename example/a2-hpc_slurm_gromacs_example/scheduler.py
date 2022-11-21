from dask.distributed import Client, as_completed
import subprocess
import os
import numpy as np
import time

def print_output(output, cnt, detail=True, start_time=0):
    if detail:
        print(' --- ', 'cycle: ', cnt, ' | START', ' | --- ')
        print('worker:\t', output['worker'])
        print('return_code:\t', output['return_code'])
        print('start_time:\t', output['start_time'])
        print('end_time:\t', output['end_time'])
        print('sim_time:\t', output['sim_time'])
        print(' --- ', 'cycle: ', cnt, ' | END  ', ' | --- ')
        print(' ')
    else: 
        print('------------', f"{output['cnt']:02}", f'{time.time() - start_time:05.02f}',
                output['folder'], output['worker'],
                ' | return code: ', output['return_code'], '------------')


def func0(inp_dic):
    worker = inp_dic['worker']
    randint = inp_dic['randint']
    folder = inp_dic['folder']
    cnt = inp_dic['cnt']
    inp_dic['start_time'] = time.time()
    
    # run nwchem in worker .. 
    stdout = folder + f'stdout_{cnt}.txt'
    stderr = folder + f'stderr_{cnt}.txt'
    cmd1 = ["srun", "--exclusive", "--ntasks", "2", "--mem-per-cpu", "500", "gmx_mpi", "grompp", "-maxwarn","2"]
    # cmd2 = ["srun", "gmx_mpi", "mdrun","-ntomp", "${SLURM_CPUS_PER_TASK}"]
    # cmd2 = ["srun", "gmx_mpi", "mdrun","-ntomp", "1"]
    cmd2 = ["srun", "--exclusive", "--ntasks", "2", "--mem-per-cpu", "500", "gmx_mpi", "mdrun", "-ntomp", "1", '-g', f"md_{cnt}"]
    with open(stdout, 'wb') as out, open(stderr, 'wb') as err:
        exe = subprocess.Popen(cmd1, cwd=folder, stdout=out, stderr=err)
        exe.communicate(input=None)
        exe = subprocess.Popen(cmd2, cwd=folder, stdout=out, stderr=err)
        exe.communicate(input=None)
        inp_dic['return_code'] = exe.returncode
    # inp_dic['return_code'] = 9000 # exe.returncode

    inp_dic['end_time'] = time.time()
    inp_dic['sim_time'] = inp_dic['end_time'] - inp_dic['start_time']
    inp_dic['cnt'] = cnt
    return inp_dic

if __name__ == "__main__":

   n_workers = 1
   for worker in range(n_workers):
       worker_n = f'worker{worker}'
       if not os.path.exists(worker_n):
           exe = subprocess.Popen(['mkdir', worker_n])
       exe = subprocess.Popen(['cp', './gromacs_files/conf.g96', worker_n + '/'])
       exe = subprocess.Popen(['cp', './gromacs_files/grompp.mdp', worker_n + '/'])
       exe = subprocess.Popen(['cp', './gromacs_files/topol.top', worker_n + '/'])

   start_time = time.time()
   cnt = 0
   cnt1 = 0
   maxi = 10
   detail = True
   client = Client(n_workers=n_workers)

   futures = as_completed(None, with_results=True)
   for worker in range(n_workers):
       inp_dic = {'worker': worker,
                  'randint': np.random.randint(1,10),
                  'folder': f'./worker{worker}/',
                  'cnt':cnt1}
       print('workerz', worker, f'./worker{worker}/')
       j = client.submit(func0, inp_dic)
       futures.add(j)
       cnt1+=1
    
   while cnt < maxi:
       output = next(futures)[1]

       print_output(output, cnt, detail, start_time)

       inp_dic = {'worker': output['worker'],
                  'randint': np.random.randint(1,10),
                  'folder': output['folder'],
                  'cnt':cnt1}
       fut = client.submit(func0, inp_dic)
       futures.add(fut)
       cnt+=1
       cnt1+=1

   for i in futures:
       output = i[1]
       print_output(output, cnt, detail, start_time)
       cnt+=1
