import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
October 7 - 2022

#timestep	potential_energy	kinetic_energy	temperature
0	7.650086346	573.7683219	0.9562805365
1	7.569654135	573.8973595	0.9564955991
2	7.354712095	574.212395	0.9570206583

"""


parser = argparse.ArgumentParser(description="just plots one line of data from.txt ")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files", required=True, nargs='+' ,help=".log file which is analyze output of hoomd" )
args = parser.parse_args()
input_files = args.input_files

data = np.loadtxt(input_files[0])

plt.figure(1)
plt.plot(data[:,0],data[:,1],label='Potential Energy')
plt.title(input_files[0])
plt.legend()
plt.show()

plt.figure(2)
plt.plot(data[:,0],data[:,2],label='Kinetic Energy')
plt.title(input_files[0])
plt.legend()
plt.show()

plt.figure(3)
plt.plot(data[:,0],data[:,3],label='Temperature')
plt.title(input_files[0])
plt.legend()
plt.show()



exit()
