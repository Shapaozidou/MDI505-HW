# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 11:08 AM
# @Author  : Sophia
# @FileName: MDI505-HW-1(1).py
# @Software: PyCharm

# command + ,  set up 3.6/2.7(preference)
# command + left  move to the very left
# reshape (-1( count by the computer itself, 1 (list))
# matrix computation/reshape , use  . values, otherwise, ignore it
# command + delete : delete the whole line
# r:   transfer markdown
# $  formula-----  text form
# decrease the margin :::::     bbox_inches="tight"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("specific-heat-and-atomic-mass-metals.csv", index_col=0, header=0)

X = 1 / data.iloc[:, 0].values
Y = data.iloc[:, 1].values

phi = np.mat(np.zeros([data.shape[0], 2]))
phi[:, 0] = np.ones(data.shape[0]).reshape(-1, 1)
phi[:, 1] = X.reshape(-1, 1)

theta_ls = np.linalg.lstsq(phi, Y, rcond=None)[0]

print(theta_ls)

x_fit = np.linspace(min(X), max(X), 100)
y_fit = theta_ls[0]+theta_ls[1]*x_fit

plt.figure(figsize=[10, 10])
plt.scatter(1/data.iloc[:, 0], data.iloc[:, 1])
plt.plot(x_fit, y_fit, color="red")
plt.xscale('log')
plt.yscale('log')
plt.title(r"$c_s$ vs $1/M_{at}$")
plt.xlabel(r"$1/M_{at}$")
plt.ylabel(r"$c_s$")
plt.savefig("Q1-Heat Capacity.png", bbox_inches="tight")







