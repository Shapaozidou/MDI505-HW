# -*- coding: utf-8 -*-
# @Time    : 2019-02-09 18:33
# @Author  : Sophia
# @FileName: MDI505-HW-1(3).py
# @Software: PyCharm

# sep="\t" ： tab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Cu-Zn-alloy-Nordheim.csv", header=0, index_col=None)
print(data)

X = data.iloc[:, 0].values/100
Y = data.iloc[:, 1].values

X = X*(1-X)

phi = np.mat(np.zeros([data.shape[0], 2]))
phi[:, 0] = np.ones(data.shape[0]).reshape(-1, 1)
phi[:, 1] = X.reshape(-1, 1)

theta_ls = np.linalg.lstsq(phi, Y, rcond=None)[0]

# print(phi.shape)
print(theta_ls)

x_fit = np.linspace(min(X), max(X), 100)
y_fit = theta_ls[0]+theta_ls[1]*x_fit

plt.figure(figsize=[10, 10])
plt.scatter(X, Y)
plt.plot(x_fit, y_fit, color="red")
plt.title("Resistivity vs percent of Zn")
plt.xlabel(r"$Zn percent$")
plt.ylabel(r"$\rho$")
plt.savefig(r"Q3-percent of Zn")

