import os

# something something hpc here.


def hpc_checker():
    if "SLURM_JOB_ID" in os.environ:
        print("Running in Slurm")
    else:
        print("NOT running in Slurm")
