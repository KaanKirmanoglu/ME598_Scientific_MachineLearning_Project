from sys import argv
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import argparse
import numpy as np
import gsd
import gsd.hoomd
import re

parser = argparse.ArgumentParser(description="-")
non_opt = parser.add_argument_group("mandatory arguments")

non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, nargs='+', dest="input_files",
required=True, help=".gsd files " )

args = parser.parse_args()
input_files = args.input_files

for kk,input_file in enumerate(input_files):
    print('%d/%d' %(kk,len(input_files)))
    in_file=gsd.fl.GSDFile(name=input_file, mode='rb',application="hoomd",schema="hoomd", schema_version=[1,0])
    N_frames = in_file.nframes
    trajectory  = gsd.hoomd.HOOMDTrajectory(in_file)
    all_pos = np.zeros((N_frames,3))
    for i in range(N_frames):
        all_pos[i,:] = trajectory[i].particles.position[0]

    if not os.path.exists('./txt_data/'):
        os.makedirs('./txt_data/')
    all_pos = np.array(all_pos)

    outname = './txt_data/' + input_file[:-3] + 'txt'
    np.savetxt(outname,all_pos,fmt='%.3f')



exit()
