#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=nv-ikj
#SBATCH --time=00:30:00
#SBATCH --nodes=1              
#SBATCH --ntasks=2    
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --job-name="sl_test"
#SBATCH --output=test-srun.out


# srun --account=nv-ikj --nodes=1 --partition=CPUQ --time=00:30:00 --pty bash 
date
rm -r 00*
module load Python/3.8.6-GCCcore-10.2.0
source ~/dask_venv/bin/activate
module load GROMACS/2021-fosscuda-2020b

python3 ./scheduler.py >| out.txt
date
