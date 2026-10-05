import numpy as np
import argparse
import re
import sys,os
np.set_printoptions(suppress=True,threshold=sys.maxsize)
"""
October 21
Take indiviual trajectories from raw_txt files
Apply pbc
spit them back

Dividing into sub trajectories is to be done at the model training scripts
as it will be treated as an hyperparameter.

Assumes each trajectory has the same length, parses the name to learn the temperature

filename : version02_rho0.30_kT1.50_runid10.txt

input data file :
6.604 5.310 3.413
6.656 5.334 3.470
6.704 5.357 3.499
6.729 5.381 3.504
6.732 5.414 3.482
...

"""

def apply_pbc(d,Lx):
    d = d - ((d - Lx*0.5)//Lx + 1)*Lx
    return d


parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
non_opt = parser.add_argument_group("mandatory arguments")

non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files", nargs='+',
required=True, help=".cl_data file " )

args = parser.parse_args()
input_files = args.input_files

if not os.path.exists('./pbc_txt/'):
    os.makedirs('./pbc_txt/')

for i,f in enumerate(input_files):
    d1 = np.loadtxt(f)
    d2 = np.zeros_like(d1)
    d2[1:,:] = d1[0:-1,:]
    d2[0,:] = d1[0,:]
    d3 = d1 - d2
    d3 = apply_pbc(d3,Lx=15.0)
    d3 = np.cumsum(d3,axis=0)
    outname = './pbc_txt/'+f[8:]
    np.savetxt(outname,d3,fmt='%.3f')
    



print("end")













exit()
