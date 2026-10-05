import numpy as np
import argparse
import re


"""
October 12
Merge individual trajectories into a single txt file
Specify the # of sub trajectories that you will get from each trajectory

Assumes each trajectory has the same length, parses the name to learn the temperature
We should probably do sth about PBC

filename : version02_rho0.30_kT1.50_runid10.txt

input data file :
6.604 5.310 3.413
6.656 5.334 3.470
6.704 5.357 3.499
6.729 5.381 3.504
6.732 5.414 3.482
...

output data file :
[traj 1 flattened kT1]
[traj 2 flattened kT2]
.
.
.
[traj N flattened kTN]
"""

parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
non_opt = parser.add_argument_group("mandatory arguments")

non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files", nargs='+',
required=True, help=".cl_data file " )


args = parser.parse_args()
input_files = args.input_files

N_sub_traj_per_traj = 10
d0 = np.loadtxt(input_files[0])
d0 = d0.flatten()
N_dim_sub_traj = len(d0)//N_sub_traj_per_traj

N_total_sub_traj = len(input_files)*N_sub_traj_per_traj

final_data = np.zeros((N_total_sub_traj,N_dim_sub_traj+1))


for i,f in enumerate(input_files):
    d = np.loadtxt(f)

    f = f.split("_")
    kT = re.findall("\d+\.\d+", f[2])
    kT = float(kT[0])

    traj = np.vsplit(d,N_sub_traj_per_traj)
    for j,d_sub in enumerate(traj):
        d_sub = d_sub.flatten()
        final_data[i*10+j,:-1] = d_sub
        final_data[i*10+j,-1] = kT

np.savetxt('dataset_kirk03.txt',final_data,fmt='%.3f')



print("end")













exit()
