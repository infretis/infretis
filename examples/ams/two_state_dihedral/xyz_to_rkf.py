# from ase.io.extxyz import read_extxyz
# from ase.io import write, read
# import sys

# # Replace 'input.xyz' and 'output.rkf' with your input and output file paths
# input_xyz = str(sys.argv[1])
# output_rkf = str(sys.argv[2])

# print(input_xyz)
# print(output_rkf)

# # Read the XYZ trajectory using ASE
# atoms_list = read(input_xyz)
# print(atoms_list)
# for i in atoms_list:
#     print(i)

# for i in read_extxyz(input_xyz):
#     print(i)

# !/usr/bin/env amspython
from scm.plams import *
import os, sys
import glob

def main():
    # the first argument needs to be a file readable by ASE
    inp = str(sys.argv[1])
    # out = str(sys.argv[2])
    # print(inp, out)
    inp_type = os.path.join(inp, f'**/*.xyz')
    inp_files = glob.glob(inp_type, recursive=True)
    for inp in inp_files:
        out = os.path.splitext(inp)[0] + '.rkf'
        print(inp, out)
        convert_to_ams_rkf_with_bond_guessing(inp, out)
    #convert_to_ams_rkf_with_bond_guessing('somefile.xyz', 'converted-to-ams.rkf')

def convert_to_ams_rkf_with_bond_guessing(filename, outfile='out.rkf', task='moleculardynamics', timestep=5.):
    temp_traj = 'out.traj'
    file_to_traj(filename, temp_traj)
    traj_to_rkf(temp_traj, outfile, task=task, timestep=timestep)

    # init()
    # #config.log.stdout = 0
    # #config.erase_workdir = True   # to remove workdir, only use this if you're not already inside another PLAMS workflow

    # s = Settings()
    # s.input.ams.task = 'replay'
    # s.input.lennardjones
    # s.input.ams.replay.file = os.path.abspath(outfile)
    # s.input.ams.properties.molecules = 'yes'
    # s.input.ams.properties.bondorders = 'yes'
    # s.runscript.nproc = 1
    # job = AMSJob(settings=s, name='rep')
    # job.run()
    # job.results.wait()
    # cpkf = os.path.expandvars('$AMSBIN/cpkf')
    # os.system(f'sh "{cpkf}" "{job.results.rkfpath()}" "{outfile}" History Molecules')
    # delete_job(job)
    # finish()

    os.remove(temp_traj)

if __name__ == '__main__':
    main()