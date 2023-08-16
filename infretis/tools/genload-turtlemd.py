import os
from infretis.classes.engines.engineparts import (
    read_xyz_file,
    write_xyz_trajectory,
    convert_snapshot
)
import numpy as np
import sys

predir = 'load'
interfaces = [-4.0, -3.5, -3.0, -2.0, 0.0, 4.0]

traj = sys.argv[1] # trajectory  file
order = sys.argv[2] # order file

if order=='nope':
    order = []
    for i, snapshot in enumerate(read_xyz_file(traj)):
        box, xyz, vel, names = convert_snapshot(snapshot)
        order.append([i, xyz[0,0]]) # order function x-position of particle 0
order = np.array(order)

print(interfaces)
n_sht_pts = 10


sorted_idx = np.argsort(order[:,1])

for i in range(len(interfaces)):
    dirname = os.path.join(predir, str(i))
    accepted = os.path.join(dirname, 'accepted')
    trajfile = os.path.join(accepted, 'traj.xyz')
    orderfile = os.path.join(dirname, 'order.txt')
    trajtxtfile = os.path.join(dirname, 'traj.txt')
    print('Making folder: {}'.format(dirname))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print('Making folder: {}'.format(accepted))
    if not os.path.exists(accepted):
        os.makedirs(accepted)
    print('Writing trajectory {} and order {} and trajfile {}'.format(trajfile, orderfile, trajtxtfile))

    # [0^-] ensemble
    if i == 0:
        start = 0
        stop = np.where(order[sorted_idx,1]<interfaces[0])[0][-1]+1
        iterator = [sorted_idx[stop]]+[i for i in sorted_idx[start:stop-1:max(len(sorted_idx[start:stop])//n_sht_pts,1)]] + [sorted_idx[stop]]
        #iterator = [sorted_idx[0]] + [sorted_idx[1]] + [sorted_idx[0]]

    # [0^+ - (N-1)^+] ensemble
    else:
        start = np.where(order[sorted_idx,1]>interfaces[0])[0][0]-1
        stop = np.where(order[sorted_idx,1]>interfaces[i])[0]
        if len(stop)==0:
            stop=-1
        else:
            stop=stop[0]

        iterator = list(sorted_idx[start:stop-1:max(len(sorted_idx[start:stop])//n_sht_pts,1)]) + [sorted_idx[stop]]
        if order[iterator[-1],1]<interfaces[-1]:
            iterator+=[sorted_idx[start]]
        print(interfaces[i],np.max(order[iterator,1]))

    for itr in iterator:
        for i, snapshot in enumerate(read_xyz_file(traj)):
            if i==itr:
                box, xyz, vel, names = convert_snapshot(snapshot)
                write_xyz_trajectory(trajfile, xyz, vel, names, box)
            
    # write order file
    N = len(iterator)
    np.savetxt(orderfile, np.c_[order[:N,0],order[iterator,1]], header=f"{'time':>10} {'orderparam':>15}",fmt=["%10.d","%15.4f"])
    np.savetxt(trajtxtfile, np.c_[[str(i) for i in range(N)],['traj.xyz' for i in range(N)], [str(i) for i in range(N)], [str(1) for i in range(N)]], header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",fmt=["%10s", "%15s","%10s","%5s"])
