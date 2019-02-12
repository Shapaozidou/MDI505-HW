# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 3:52 PM
# @Author  : Sophia
# @FileName: MDI505-HW-1(2).py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Sr-resistivity-vs-temp.txt", sep="\t", header=0, index_col=None)
print(data)

X_temperature = data.iloc[:, 0].values
Y_resistivity = data.iloc[:, 1].values
X_temperature_log = np.log10(X_temperature)
Y_resistivity_log = np.log10(Y_resistivity)

phi = np.mat(np.zeros([data.shape[0], 2]))
phi[:, 0] = np.ones(data.shape[0]).reshape(-1, 1)
phi[:, 1] = X_temperature_log.reshape(-1, 1)

theta_ls = np.linalg.lstsq(phi, Y_resistivity_log, rcond=None)[0]
print("theta_ls = " + str(theta_ls))

x_fit = np.linspace(min(X_temperature_log), max(X_temperature_log), 100)
y_fit = theta_ls[0]+theta_ls[1]*x_fit

plt.figure(figsize=[10, 10])
plt.scatter(X_temperature_log, Y_resistivity_log)
plt.plot(x_fit, y_fit, color="red")
plt.xscale('log')
plt.yscale('log')
plt.title("Resistivity vs Temperature")
plt.xlabel(r"$T$")
plt.ylabel(r"$\rho$")
plt.savefig("Q2-Resistivity vs Temperature")

print("n =" + str(theta_ls[1]))
