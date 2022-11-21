#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=nv-ikj
#SBATCH --time=00:08:30
#SBATCH --nodes=1              
#SBATCH --ntasks=2    
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --job-name="sl_test"
#SBATCH --output=test-srun.out


# srun --account=nv-ikj --nodes=1 --partition=CPUQ --time=00:30:00 --pty bash 
date
rm -r worker0/ worker1/
module load Python/3.8.6-GCCcore-10.2.0
source ~/dask_venv/bin/activate
module load GROMACS/2021-fosscuda-2020b
# module load NWChem/7.0.2-intel-2020b-Python-3.8.6

python3 ./scheduler.py >| out.txt
date
