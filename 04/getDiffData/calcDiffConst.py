import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import sys,os
from scipy.stats import linregress

arr1 = np.arange(0, 5)
print(5 > 0)
print(int(0 > 5))
path_to_txt = "raw_txt/version02_rho0.30_kT0.10_runid10.txt"
path_base = "pbc_txt/version02_rho0.30_kT0.10_runid"
print(path_base + str(14) + ".txt")
L = 15
temp_arr = np.linspace(0.1, 3, 30)

id_list = np.arange(10, 28)
#x_ms = np.zeros([100000, 3])
temp_diff = np.zeros([30, 2])

pathto = "pbc_txt/version02_rho0.30_kT"
i = 0
fig_id = 1

for temp in temp_arr:
    print(temp)
    filename = pathto + str(format(temp, '.2f')) + "_runid"
    x_msi = np.zeros([2000, 3])
    for r_id in id_list:
        path_to_pbc = filename + str(r_id) + ".txt"
        x_i = np.loadtxt(path_to_pbc)
        t_arr = np.arange(0, len(x_i))
        xsp = np.split(x_i, 50)
        t_arr_sp = np.split(t_arr, 50)
        ts0 = t_arr_sp[0]
        delta_t = ts0[-1] - ts0[0]
        for t in range(len(xsp)):
            xsi = xsp[t]
            tsi = t_arr_sp[t]
            #plt.plot(tsi, xsi[:, 0], 'bo')
            del_x = xsi[:, :] - xsi[0, :]
            x_msi = x_msi + np.power(del_x, 2)
    x_ms3 = x_msi / (len(id_list)*50)
    x_ms3 = np.mean(x_ms3, axis=1)
    t_arr3 = np.arange(0, len(x_ms3))
    plt.figure(fig_id)
    plt.plot(x_ms3, 'bo')
    plt.xlabel('$\Delta t$')
    plt.ylabel('$<\Delta x^2>$')
    plt.show()
    fig_id = fig_id + 1
    (slope, intercept, rvalue, pvalue, stderr) = linregress(t_arr3, x_ms3)
    temp_diff[i] = [temp, slope]
    print(i)
    i = i+1
#
#
plt.figure(fig_id)
plt.plot(temp_diff[:, 0], temp_diff[:, 1], 'bo')
plt.show()

path_to_pbc = path_base + str(15) + ".txt"
x_i = np.loadtxt(path_to_pbc)
#
# plt.figure(1)
# plt.plot(x_i[:, 0], 'bo')
# plt.show()
t_arr = np.arange(0, len(x_i))
xsp = np.split(x_i, 50)
x_ms = np.zeros(xsp[0].shape)
t_arr_sp = np.split(t_arr, 50)
#
ts0 = t_arr_sp[0]
delta_t = ts0[-1] - ts0[0]
#
for i in range(len(xsp)):
    xsi = xsp[i]
    tsi = t_arr_sp[i]
    #plt.plot(tsi, xsi[:, 0], 'bo')
    del_x = xsi[:, :] - xsi[0, :]
    x_ms = x_ms + np.power(del_x, 2)
#
x_ms = x_ms/len(xsp)
x_ms = np.mean(x_ms, axis=1)
#plt.figure(2)
#plt.plot(x_ms, 'bo')
#plt.show()

# for r_id in id_list:
#     path_to_pbc = path_base + str(r_id) + ".txt"
#     x_i = np.loadtxt(path_to_pbc)
#     x_2 = np.power(x_i,2)
#     x_ms = x_ms + x_2
# #
# x_ms = x_ms/len(id_list)
# t_arr = np.arange(0, len(x_ms))
# # A_mat*t_arr = x_ms
# plt.figure(1)
# plt.plot(x_ms[:, 0], 'bo')
# plt.show()
# plt.figure(2)
# plt.plot(x_ms[:, 1], 'bo')
# plt.show()
#
# plt.figure(3)
# plt.plot(x_ms[:, 2], 'bo')
# plt.show()
#
# x_ms3 = np.mean(x_ms, axis=1)
#
# (slope, intercept, rvalue, pvalue, stderr) = linregress(t_arr, x_ms3)
# x_ms_pred = intercept + slope*t_arr
# plt.figure(4)
# plt.plot(x_ms3, 'bo')
# plt.plot(x_ms_pred, 'r')
# plt.show()
#
#
#
#
#

#plt.figure(2)
#plt.plot(r_t_pbc[:, 0], 'bo')
#plt.show()


#parser = argparse.ArgumentParser(description="Cluster analysis of a gsd file for a single frame")
#non_opt = parser.add_argument_group("mandatory arguments")

#non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_files", nargs='+',
#required=True, help=".cl_data file " )

#args = parser.parse_args()
#input_files = args.input_files
